#!/usr/bin/env python3
"""Prepare DAPO-Math-17k subsets for the async long-CoT experiment.

Source: HF dataset BytedTsinghua-SIA/DAPO-Math-17k (data/dapo-math-17k.parquet),
which is ALREADY in verl RLHFDataset format:
  - data_source = "math_dapo"  -> default reward routing hits
    verl.utils.reward_score.math_dapo.compute_score (Answer: <ans> extraction,
    matching the instruction baked into every prompt)
  - prompt = [{"role": "user", "content": "...Remember to put your answer on its
    own line after \"Answer:\"..."}]
  - reward_model = {"style": "rule", "ground_truth": "<answer string>"}

We only subsample (512 train / 64 val, disjoint) and sanity-check the schema.
No agent_name column -> single_turn_agent (default), single-turn long CoT.
"""

import argparse
import os

import pandas as pd
from huggingface_hub import hf_hub_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", default="BytedTsinghua-SIA/DAPO-Math-17k")
    ap.add_argument("--filename", default="data/dapo-math-17k.parquet")
    ap.add_argument("--out_dir", default="/workspace/data/dapo_math")
    ap.add_argument("--train_size", type=int, default=512)
    ap.add_argument("--val_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    local = hf_hub_download(repo_id=args.repo_id, filename=args.filename, repo_type="dataset")
    df = pd.read_parquet(local)
    print(f"loaded {len(df)} rows, columns: {list(df.columns)}")

    # ---- schema sanity checks (fail fast, before burning GPU time) ----
    row = df.iloc[0]
    prompt = row["prompt"]
    assert row["data_source"] == "math_dapo", f"unexpected data_source {row['data_source']!r}"
    p0 = prompt[0] if not isinstance(prompt, dict) else prompt
    p0 = dict(p0)
    assert p0.get("role") == "user" and isinstance(p0.get("content"), str), f"bad prompt row: {p0}"
    assert "Answer:" in p0["content"], "prompt lacks the Answer: instruction the reward fn expects"
    rm = dict(row["reward_model"])
    gt = rm.get("ground_truth")
    assert isinstance(gt, str) and gt, f"bad ground_truth: {gt!r}"
    print("schema OK; sample prompt (first 400 chars):")
    print(p0["content"][:400])
    print("sample ground_truth:", gt[:100])

    # Deduplicate on prompt text so train/val are truly distinct problems.
    df = df.assign(_ptext=df["prompt"].map(lambda p: dict(list(p)[0])["content"]))
    df = df.drop_duplicates("_ptext").drop(columns="_ptext").reset_index(drop=True)
    print(f"{len(df)} unique problems after dedup")

    n = args.train_size + args.val_size
    sub = df.sample(n=min(n, len(df)), random_state=args.seed).reset_index(drop=True)
    train = sub.iloc[: args.train_size]
    val = sub.iloc[args.train_size : args.train_size + args.val_size]

    os.makedirs(args.out_dir, exist_ok=True)
    train.to_parquet(os.path.join(args.out_dir, "train.parquet"), index=False)
    val.to_parquet(os.path.join(args.out_dir, "val.parquet"), index=False)
    print(f"wrote {len(train)} train / {len(val)} val -> {args.out_dir}")


if __name__ == "__main__":
    main()
