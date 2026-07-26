"""Standalone port of CMU's examples/data_preprocess/deep_research_data_prepare.py.

Converts the deepresearch env JSON question files to the parquet format
verl main's RLHFDataset expects. The parquet only feeds batch sizing +
raw_prompt/data_source — the multi-turn loop pulls the actual questions from
the env's own JSON dataset (DeepResearchEnvironmentManager._load_dataset).

Usage:
  python3 prepare_deepresearch_data.py \
      --train_json .../data/deepresearch_mhqa/train.json \
      --val_json   .../data/deepresearch_mhqa/val.json \
      --out_dir    /workspace/data \
      --train_data_size 128 --val_data_size 64
"""

import argparse
import json
import os


def _map_to_rl_sample(example: dict, split: str) -> dict:
    question = example.get("question", "")
    return {
        "data_source": "webwalker",
        "prompt": [
            {
                "role": "user",
                "content": question,
            }
        ],
        "ability": "agent",
        "extra_info": {
            "split": split,
            "id": str(example.get("id")),
            # Keep answer in extra_info for potential evaluation usage
            "answer": str(example.get("answer")),
        },
    }


def _convert(json_path: str, split: str, cap: int | None) -> list[dict]:
    with open(json_path) as f:
        rows = json.load(f)
    if cap is not None:
        rows = rows[: min(cap, len(rows))]
    return [_map_to_rl_sample(r, split) for r in rows]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--val_json", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="dummy_data/text")
    parser.add_argument("--train_data_size", type=int, default=None)
    parser.add_argument("--val_data_size", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    import pandas as pd

    train_rows = _convert(args.train_json, "train", args.train_data_size)
    val_rows = _convert(args.val_json, "validation", args.val_data_size)

    train_out = os.path.join(args.out_dir, "train.parquet")
    val_out = os.path.join(args.out_dir, "val.parquet")
    pd.DataFrame(train_rows).to_parquet(train_out)
    pd.DataFrame(val_rows).to_parquet(val_out)

    print(f"Train size: {len(train_rows)} -> {train_out}")
    print(f"Val size: {len(val_rows)} -> {val_out}")
