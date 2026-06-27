"""reward.py — rule-based GSM8K reward and GRPO advantage computation (JAX/TPU port)

Uses willccbb's validated XML format:
  <reasoning>...chain of thought...</reasoning>
  <answer>42</answer>

Reward structure (matching willccbb's gist):
  2.0 = correct answer
  0.5 = correct XML format (even if answer wrong)
  0.0 = wrong answer, no format

This file is pure Python (no JAX dependency) — identical logic to the GPU impl.
"""

import re
import statistics
from typing import List, Optional, Tuple


_STOP_TOKENS = ("<|im_end|>", "</s>")


# -- Answer extraction --------------------------------------------------------


def extract_xml_answer(text: str) -> Optional[str]:
    """Extract answer from <answer>...</answer> tags."""
    m = re.search(r"<answer>\s*([\s\S]*?)\s*</answer>", text)
    if not m:
        return None
    raw = m.group(1).strip()
    nums = re.findall(r"[-]?\d+(?:,\d{3})*(?:\.\d+)?", raw.replace(",", ""))
    return nums[-1] if nums else raw


def extract_answer(text: str) -> Optional[str]:
    """Extract final answer.

    Priority:
      1. <answer>N</answer>  (XML format)
      2. \\boxed{N}           (Qwen native)
      3. #### N              (GSM8K format)
      4. Last number — only if not truncated
    """
    # 1. XML answer tag
    ans = extract_xml_answer(text)
    if ans:
        return ans.replace(",", "")
    # 2. \boxed{N}
    m = re.search(r"boxed\{([^}]+)\}", text)
    if m:
        val = m.group(1).strip().lstrip("$").strip()
        nums = re.findall(r"[-]?\d+(?:,\d{3})*(?:\.\d+)?", val)
        return nums[-1].replace(",", "") if nums else None
    # 3. #### N
    m = re.search(r"####\s*\$?\s*([-\d][,\d]*(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    # 4. Last number fallback (only if not truncated)
    if not is_truncated(text):
        nums = re.findall(r"[-]?\d+(?:,\d{3})*(?:\.\d+)?", text)
        return nums[-1].replace(",", "") if nums else None
    return None


def is_truncated(text: str) -> bool:
    """True if completion was cut off before finishing."""
    t = text.rstrip()
    if any(t.endswith(tok) for tok in _STOP_TOKENS):
        return False
    if re.search(r"</answer>", t):
        return False
    if re.search(r"boxed\{[^}]+\}", t):
        return False
    if re.search(r"####\s*\$?\s*[-\d]", t):
        return False
    if re.search(r"\*\*\s*\\?\$?\s*[-\d][,\d]*(?:\.\d+)?\s*\*\*", t):
        return False
    if re.search(r"\d[.)!]?\s*$", t):
        return False
    if re.search(r"[.!?]\s*$", t):
        return False
    if re.search(r"</reasoning>", t):
        return False
    return True


def has_xml_format(text: str) -> bool:
    """True if completion has the XML answer structure."""
    return bool(
        re.search(r"</reasoning>", text)
        and re.search(r"<answer>[\s\S]*</answer>", text)
    )


def normalize(ans: str) -> str:
    try:
        f = float(str(ans).replace(",", ""))
        return str(int(f)) if f == int(f) else f"{f:.6f}".rstrip("0").rstrip(".")
    except Exception:
        return str(ans).strip().lower()


# -- Reward computation -------------------------------------------------------


def compute_rewards(
    completions: List[str], ground_truths: List[str]
) -> Tuple[List[float], dict]:
    """Reward function matching willccbb's validated setup:

      2.0 = correct answer
      0.5 = correct XML format, wrong answer
      0.0 = wrong answer, no format
    """
    rewards = []
    n_correct = n_format = n_truncated = 0

    for comp, gt in zip(completions, ground_truths):
        has_fmt = has_xml_format(comp)
        truncated = is_truncated(comp)
        pred = extract_answer(comp)
        correct = bool(pred and gt and normalize(pred) == normalize(str(gt)))

        if has_fmt:
            n_format += 1
        if truncated:
            n_truncated += 1
        if correct:
            n_correct += 1

        if correct:
            reward = 2.0
        elif has_fmt:
            reward = 0.5
        else:
            reward = 0.0

        rewards.append(reward)

    n = max(len(completions), 1)
    stats = {
        "correct_rate": round(n_correct / n, 4),
        "format_rate": round(n_format / n, 4),
        "truncated_rate": round(n_truncated / n, 4),
    }
    return rewards, stats


def compute_advantages(
    rewards: List[float], group_size: int
) -> Tuple[List[float], dict]:
    """GRPO: normalise rewards within each group of G completions.

    With the 0/0.5/2.0 reward scale, groups will have meaningful std as long as
    the model doesn't always get all correct or all wrong.
    """
    n = len(rewards) // group_size
    advantages = []
    all_stats = {}
    for i in range(n):
        grp = rewards[i * group_size : (i + 1) * group_size]
        mean = statistics.mean(grp)
        std = statistics.stdev(grp) if len(grp) > 1 else 1.0
        for r in grp:
            advantages.append((r - mean) / (std + 1e-8))
        all_stats[i] = {"mean": round(mean, 4), "std": round(std, 4)}
    return advantages, all_stats
