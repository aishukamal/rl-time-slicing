#!/usr/bin/env python3
"""Prepare the Eurus-2-RL-Data CODE subset for the async code-RLVR experiment.

Source: HF dataset PRIME-RL/Eurus-2-RL-Data (train.parquet), ALREADY in verl
RLHFDataset format:
  - data_source in {codecontests, apps, codeforces, taco} -> default reward routing
    hits verl.utils.reward_score.prime_code.compute_score (local in-pod test-case
    execution; extracts the LAST ```python fenced block)
  - prompt = [{system (Eurus action protocol)}, {user: problem + "Write Python code
    ... ```python ... ``` at the end"}]
  - reward_model = {"style": "rule", "ground_truth": '{"inputs": [...], "outputs": [...]}'}

Transformations (documented in NOTES.md):
  1. keep only ability == "code" rows with a data_source that routes to prime_code
  2. DROP the Eurus system message (R1-Distill guidance: no system prompt; the
     reward-relevant ```python instruction lives in the user message, kept verbatim)
  3. truncate test suites to first 8 in/out pairs (bounds worst-case reward latency),
     preserving fn_name; drop rows with unusable/oversized ground truth
  4. filter to prompts <= --max_prompt_tokens under the actual chat template
  5. subsample 512 train / 64 val, disjoint, dedup'd on user text
"""

import argparse
import json
import os

import pandas as pd
from huggingface_hub import hf_hub_download

CODE_SOURCES = {"codecontests", "apps", "codeforces", "taco"}


def truncate_ground_truth(gt_str, max_tests, max_bytes):
    """Return truncated JSON ground truth, or None if the row is unusable."""
    try:
        gt = json.loads(gt_str)
    except Exception:
        return None
    if not isinstance(gt, dict):
        return None
    ins, outs = gt.get("inputs"), gt.get("outputs")
    if not isinstance(ins, list) or not isinstance(outs, list):
        return None
    n = min(len(ins), len(outs), max_tests)
    if n < 1:
        return None
    new_gt = {"inputs": ins[:n], "outputs": outs[:n]}
    if gt.get("fn_name"):
        new_gt["fn_name"] = gt["fn_name"]
    try:
        s = json.dumps(new_gt)
    except Exception:
        return None
    if len(s) > max_bytes:
        return None
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", default="PRIME-RL/Eurus-2-RL-Data")
    ap.add_argument("--filename", default="train.parquet")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out_dir", default="/workspace/data/eurus_code")
    ap.add_argument("--train_size", type=int, default=512)
    ap.add_argument("--val_size", type=int, default=64)
    ap.add_argument("--max_tests", type=int, default=8)
    ap.add_argument("--max_gt_bytes", type=int, default=200_000)
    ap.add_argument("--max_prompt_tokens", type=int, default=1900)
    ap.add_argument("--candidates", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    local = hf_hub_download(repo_id=args.repo_id, filename=args.filename, repo_type="dataset")
    df = pd.read_parquet(local)
    print(f"loaded {len(df)} rows, columns: {list(df.columns)}")

    df = df[df["ability"] == "code"].reset_index(drop=True)
    print(f"{len(df)} code rows; data_source counts:\n{df['data_source'].value_counts()}")
    df = df[df["data_source"].isin(CODE_SOURCES)].reset_index(drop=True)
    print(f"{len(df)} rows with prime_code-routable data_source")

    # ---- rebuild rows: user-only prompt + truncated ground truth ----
    rows = []
    n_bad_gt, n_bad_prompt = 0, 0
    for _, r in df.iterrows():
        msgs = [dict(m) for m in r["prompt"]]
        user = [m for m in msgs if m.get("role") == "user"]
        if len(user) != 1 or "```python" not in user[0].get("content", ""):
            n_bad_prompt += 1
            continue
        gt = truncate_ground_truth(dict(r["reward_model"])["ground_truth"], args.max_tests, args.max_gt_bytes)
        if gt is None:
            n_bad_gt += 1
            continue
        rows.append(
            {
                "data_source": r["data_source"],
                "prompt": [{"role": "user", "content": user[0]["content"]}],
                "ability": "code",
                "reward_model": {"style": "rule", "ground_truth": gt},
                "extra_info": dict(r["extra_info"]) if r["extra_info"] is not None else {},
                "_ptext": user[0]["content"],
            }
        )
    print(f"kept {len(rows)} rows (dropped {n_bad_prompt} bad-prompt, {n_bad_gt} bad-ground-truth)")

    out = pd.DataFrame(rows).drop_duplicates("_ptext").reset_index(drop=True)
    print(f"{len(out)} unique problems after dedup")

    # cheap char prefilter, then exact token filter on a sampled candidate pool
    out = out[out["_ptext"].str.len() <= 7000].reset_index(drop=True)
    print(f"{len(out)} rows after 7000-char prefilter")
    cand = out.sample(n=min(args.candidates, len(out)), random_state=args.seed).reset_index(drop=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    keep_idx, tok_lens = [], []
    for i, r in cand.iterrows():
        ntok = len(tok.apply_chat_template(list(r["prompt"]), add_generation_prompt=True))
        if ntok <= args.max_prompt_tokens:
            keep_idx.append(i)
            tok_lens.append(ntok)
    cand = cand.loc[keep_idx].reset_index(drop=True)
    print(
        f"{len(cand)} candidates <= {args.max_prompt_tokens} prompt tokens "
        f"(mean {sum(tok_lens) / max(len(tok_lens), 1):.0f})"
    )

    n = args.train_size + args.val_size
    assert len(cand) >= n, f"only {len(cand)} candidates for {n} needed"
    sub = cand.sample(n=n, random_state=args.seed + 1).reset_index(drop=True)
    train = sub.iloc[: args.train_size].drop(columns="_ptext")
    val = sub.iloc[args.train_size : n].drop(columns="_ptext")

    # ---- final schema sanity checks (fail fast, before burning GPU time) ----
    row = train.iloc[0]
    assert row["data_source"] in CODE_SOURCES
    p0 = dict(row["prompt"][0])
    assert p0["role"] == "user" and "```python" in p0["content"]
    gt = json.loads(dict(row["reward_model"])["ground_truth"])
    assert len(gt["inputs"]) >= 1 and len(gt["inputs"]) == len(gt["outputs"])
    ntests = [len(json.loads(dict(rm)["ground_truth"])["inputs"]) for rm in train["reward_model"]]
    n_fn = sum(1 for rm in train["reward_model"] if "fn_name" in json.loads(dict(rm)["ground_truth"]))
    print(f"tests/problem: mean {sum(ntests) / len(ntests):.1f} max {max(ntests)}; call-based (fn_name): {n_fn}")
    print("sample prompt (first 400 chars):")
    print(p0["content"][:400])
    print("sample prompt (last 200 chars):")
    print(p0["content"][-200:])
    print("sample ground_truth (first 200 chars):", dict(row["reward_model"])["ground_truth"][:200])

    os.makedirs(args.out_dir, exist_ok=True)
    train.to_parquet(os.path.join(args.out_dir, "train.parquet"), index=False)
    val.to_parquet(os.path.join(args.out_dir, "val.parquet"), index=False)
    print(f"wrote {len(train)} train / {len(val)} val -> {args.out_dir}")


if __name__ == "__main__":
    main()
