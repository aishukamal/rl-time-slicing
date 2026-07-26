# Copyright 2026
# Licensed under the Apache License, Version 2.0
"""Wikipedia search tool for multi-turn RL rollout.

Replacement for the removed in-tree ``verl.tools.search_tool.SearchTool``.
Talks to the local pyserini BM25 Wikipedia server (Serper-style request/response
schema, see benchmark-deepresearch local_search_server.py):

    POST {search_url}  body: {"q": <query>, "num": <topk>}
    resp: {"organic": [{"title": ..., "snippet": ...}, ...]}

Deployed by copying this file into the cloned verl tree as
``verl/tools/wiki_search_tool.py`` so ``class_name:
verl.tools.wiki_search_tool.WikiSearchTool`` resolves inside every Ray worker.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class WikiSearchTool(BaseTool):
    """Search tool backed by a local Serper-format Wikipedia retrieval server."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.search_url = config.get("search_url", "http://localhost:8877/search")
        self.topk = int(config.get("topk", 3))
        self.timeout = float(config.get("timeout", 30))
        self._instance_dict: dict[str, dict] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        # create_kwargs (ground_truth/question/data_source) arrive here; keep for debugging.
        self._instance_dict[instance_id] = {"create_kwargs": kwargs, "num_calls": 0}
        return instance_id, ToolResponse()

    def _search_one(self, query: str) -> str:
        """Blocking HTTP call, run in a thread. Returns formatted doc block."""
        import requests

        resp = requests.post(
            self.search_url,
            data=json.dumps({"q": query, "num": self.topk}),
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        organic = resp.json().get("organic", [])
        if not organic:
            return f'Query: "{query}"\nNo results found.'
        docs = []
        for i, hit in enumerate(organic[: self.topk]):
            title = hit.get("title", "")
            snippet = hit.get("snippet", "")
            docs.append(f"Doc {i + 1} (Title: {title})\n{snippet}")
        return f'Query: "{query}"\n' + "\n\n".join(docs)

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        query_list = parameters.get("query_list") or []
        if isinstance(query_list, str):
            query_list = [query_list]
        query_list = [str(q) for q in query_list if q]
        if not query_list:
            return (
                ToolResponse(text="Error: no query provided. Pass query_list=[...]"),
                0.0,
                {"num_queries": 0, "error": "empty_query"},
            )

        loop = asyncio.get_event_loop()
        blocks = []
        errors = 0
        for q in query_list:
            try:
                block = await loop.run_in_executor(None, self._search_one, q)
            except Exception as e:  # noqa: BLE001 - tool must not crash the agent loop
                logger.warning("search failed for %r: %s", q, e)
                block = f'Query: "{q}"\nSearch error: {e}'
                errors += 1
            blocks.append(block)

        if instance_id in self._instance_dict:
            self._instance_dict[instance_id]["num_calls"] += 1

        text = "\n\n".join(blocks)
        metrics = {"num_queries": len(query_list), "errors": errors}
        return ToolResponse(text=text), 0.0, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
