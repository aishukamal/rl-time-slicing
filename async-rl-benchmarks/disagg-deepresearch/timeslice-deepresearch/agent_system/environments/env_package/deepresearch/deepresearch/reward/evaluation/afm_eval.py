# Vendored from cxcscmu/verl-agent-deepresearch (commit 9c311053).
# Changes: dotenv dropped (plain env var), OpenAI client created lazily so a
# missing `openai` package or missing key degrades to score 0 instead of
# crashing the env actor at import time. With the dummy OPENAI_API_KEY used
# for GPU-trace benchmarks all judged calls fail -> score 0 for every row,
# matching the colocated baseline behavior.

import json
import os
import time

_client = None
_client_failed = False


def _get_client():
    global _client, _client_failed
    if _client is not None or _client_failed:
        return _client
    try:
        from openai import OpenAI
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=180)
    except Exception as e:  # pragma: no cover
        import sys
        print(f"[afm_eval] OpenAI client unavailable ({e}); all judge scores will be 0",
              file=sys.stderr)
        _client_failed = True
    return _client


llm_prompt = """
Please determine if the predicted answer is SEMANTICALLY equivalent to the labeled answer.
Question:  {question}
Labeled Answer:  {gt_answer}
Predicted Answer: {pred_answer}

{{
"rationale": "your rationale for the judgement, as a text",
"judgement": "your judgement result, can only be 'correct' or 'incorrect'
}}
"""


def retry_predict(model, prompt, developer_prompt=None):
    client = _get_client()
    if client is None:
        return ""
    messages = []
    if developer_prompt:
        messages.append({
            "role": "system",
            "content": developer_prompt
        })
    messages.append({
        "role": "user",
        "content": prompt
    })
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=8192,
                top_p=0.9,
                frequency_penalty=0,
                presence_penalty=0
            )
            content = response.choices[0].message.content.strip()
            if content:
                return content
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1 * (2 ** attempt))  # Exponential backoff
    return ""


def decode_response(response):
    try:
        if isinstance(response, str):
            return json.loads(response)
        return response
    except Exception:
        return {"judgement": "incorrect"}


def evaluate_afm_answer(question, answer, ground_truth):
    llm_evaluation_prompt = llm_prompt.format(
        question=question,
        gt_answer=ground_truth,
        pred_answer=answer
    )
    output = retry_predict(
        "gpt-4o-mini",
        llm_evaluation_prompt,
        developer_prompt="You are an evaluation assistant."
    )
    json_output = decode_response(output)
    if (json_output and isinstance(json_output, dict) and
            "judgement" in json_output and
            json_output['judgement'].lower() == "correct"):
        score = 1
    else:
        score = 0
    return score
