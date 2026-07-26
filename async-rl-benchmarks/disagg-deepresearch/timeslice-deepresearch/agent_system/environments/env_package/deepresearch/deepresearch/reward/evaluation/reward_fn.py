# Vendored from cxcscmu/verl-agent-deepresearch (commit 9c311053).
# Only the qa mode (AFM LLM-judge) survives — the report-mode evaluators
# (eval_kpr_async / eval_quality_async) and the webwalker evaluator were
# stripped: they pull the openai/langchain dependency trees and are unused by
# the MHQA benchmark.

from .afm_eval import evaluate_afm_answer


def evaluation_reward_fn(query_id, question, answer, mode, ground_truth=None, options=None):
    if mode == 'report':
        raise NotImplementedError(
            "report mode evaluators were stripped from this vendored copy")
    elif mode == "qa":
        score = evaluate_afm_answer(question, answer, ground_truth)
        return score
