# Vendored from cxcscmu/verl-agent-deepresearch (commit 9c311053).
# The upstream file loads a hardcoded CMU cluster path
# (/data/group_data/cx_group/behavior_priming/Qwen3/Qwen3-8B). This copy loads
# the tokenizer lazily from DEEPRESEARCH_TOKENIZER_PATH (set to the training
# model path in the job script) and falls back to a chars/4 estimate if no
# tokenizer can be loaded — tokenize() is only used for context-length
# accounting (summary reminder at 8000 tokens), not for training tensors.

import os

_tokenizer = None
_tokenizer_failed = False


def _get_tokenizer():
    global _tokenizer, _tokenizer_failed
    if _tokenizer is not None or _tokenizer_failed:
        return _tokenizer
    model_name = os.environ.get("DEEPRESEARCH_TOKENIZER_PATH")
    if not model_name:
        _tokenizer_failed = True
        return None
    try:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:  # pragma: no cover
        import sys
        print(f"[deepresearch.utils] failed to load tokenizer from {model_name}: {e}; "
              f"falling back to len//4 estimate", file=sys.stderr)
        _tokenizer_failed = True
    return _tokenizer


def tokenize(input):
    tok = _get_tokenizer()
    if tok is None:
        return len(input) // 4
    return len(tok.encode(input))
