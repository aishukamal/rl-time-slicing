#!/usr/bin/env python3
"""Prepare a small search-R1-style multi-hop QA parquet for verl ToolAgentLoop.

Source: HF dataset PeterJinGo/nq_hotpotqa_train (the Search-R1 training mix).
Filters to hotpotqa rows (multi-hop -> variable turn counts), samples
TRAIN_SIZE train / VAL_SIZE val prompts, and writes verl RLHFDataset-compatible
parquet with agent_name=tool_agent + tools_kwargs for the `search` tool.

Reward: data_source `searchR1_hotpotqa` routes to
verl.utils.reward_score.search_r1_like_qa_em (EM on <answer>...</answer> vs
ground_truth["target"]) -- no custom reward function needed.
"""

import argparse
import os

import pandas as pd
from huggingface_hub import hf_hub_download

SYSTEM_CONTENT = "You are a helpful and harmless assistant."

# Search-R1 style instructions adapted for hermes/native function calling:
# the model calls the `search` tool (schema injected via chat template) instead
# of emitting free-form <tool_call> query text.
USER_CONTENT_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and "
    "</think> first every time you get new information. After reasoning, if you "
    "find you lack some knowledge, you can call the search tool with a list of "
    "queries; it will return the top results. You can search as many times as "
    "you want. If you find no further external knowledge needed, provide the "
    "answer inside <answer> and </answer> without detailed illustrations. For "
    "example, <answer> Beijing </answer>. Question: "
)


def to_list(x):
    """golden_answers may be a numpy array from parquet."""
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return [str(a) for a in list(x)]


def process_row(row, split_name, idx):
    question = str(row.get("question", "")).strip()
    golden = to_list(row.get("golden_answers"))

    rm = row.get("reward_model")
    if isinstance(rm, dict) and isinstance(rm.get("ground_truth"), dict) and "target" in rm["ground_truth"]:
        ground_truth = {"target": to_list(rm["ground_truth"]["target"])}
    elif isinstance(rm, dict) and "ground_truth" in rm:
        ground_truth = {"target": to_list(rm["ground_truth"])}
    else:
        ground_truth = {"target": golden}

    data_source = "searchR1_" + str(row.get("data_source", "hotpotqa"))
    prompt = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": USER_CONTENT_PREFIX + question},
    ]
    tools_kwargs = {
        "search": {
            "create_kwargs": {
                "ground_truth": ground_truth,
                "question": question,
                "data_source": data_source,
            }
        }
    }
    return pd.Series(
        {
            "data_source": data_source,
            "prompt": prompt,
            "ability": "multihop-qa",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "index": idx,
                "split": split_name,
                "question": question,
                "need_tools_kwargs": True,
                "tools_kwargs": tools_kwargs,
            },
            "agent_name": "tool_agent",
        }
    )


def prep_split(repo_id, filename, split_name, size, out_path, seed):
    local = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    df = pd.read_parquet(local)
    print(f"{split_name}: loaded {len(df)} rows, sources: {df['data_source'].value_counts().to_dict()}")

    hotpot = df[df["data_source"] == "hotpotqa"]
    if len(hotpot) >= size:
        df = hotpot
    else:
        print(f"WARNING: only {len(hotpot)} hotpotqa rows; using full mix")
    df = df.sample(n=min(size, len(df)), random_state=seed).reset_index(drop=True)

    out = df.apply(lambda r: process_row(r, split_name, r.name), axis=1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"{split_name}: wrote {len(out)} rows -> {out_path}")
    print(f"{split_name}: sample question: {out.iloc[0]['extra_info']['question'][:120]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", default="PeterJinGo/nq_hotpotqa_train")
    ap.add_argument("--out_dir", default="/workspace/data/hotpotqa_search")
    ap.add_argument("--train_size", type=int, default=512)
    ap.add_argument("--val_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    prep_split(args.repo_id, "train.parquet", "train", args.train_size,
               os.path.join(args.out_dir, "train.parquet"), args.seed)
    try:
        prep_split(args.repo_id, "test.parquet", "test", args.val_size,
                   os.path.join(args.out_dir, "val.parquet"), args.seed)
    except Exception as e:  # test split may be absent; fall back to a train slice
        print(f"WARNING: test split failed ({e}); sampling val from train with different seed")
        prep_split(args.repo_id, "train.parquet", "test", args.val_size,
                   os.path.join(args.out_dir, "val.parquet"), args.seed + 1)


if __name__ == "__main__":
    main()
