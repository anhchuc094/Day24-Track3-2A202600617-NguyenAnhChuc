from __future__ import annotations

"""Module 4: RAGAS evaluation and diagnostic failure analysis."""

import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH
from src.llm_client import configure_ragas_environment


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _overlap(a: str, b: str) -> float:
    left = _tokens(a)
    right = _tokens(b)
    if not left:
        return 0.0
    return len(left & right) / len(left)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _evaluate_lexical(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> list[EvalResult]:
    per_question = []
    for question, answer, ctxs, ground_truth in zip(questions, answers, contexts, ground_truths):
        context_text = " ".join(ctxs)
        per_question.append(
            EvalResult(
                question=question,
                answer=answer,
                contexts=ctxs,
                ground_truth=ground_truth,
                faithfulness=min(1.0, _overlap(answer, context_text)),
                answer_relevancy=min(1.0, _overlap(question, answer)),
                context_precision=min(1.0, _overlap(question, context_text)),
                context_recall=min(1.0, _overlap(ground_truth, context_text)),
            )
        )
    return per_question


def evaluate_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """Run RAGAS evaluation, falling back to transparent lexical proxies."""
    configure_ragas_environment()
    if os.getenv("RAG_SKIP_RAGAS") == "1":
        per_question = _evaluate_lexical(questions, answers, contexts, ground_truths)
        return {
            "faithfulness": _mean([r.faithfulness for r in per_question]),
            "answer_relevancy": _mean([r.answer_relevancy for r in per_question]),
            "context_precision": _mean([r.context_precision for r in per_question]),
            "context_recall": _mean([r.context_recall for r in per_question]),
            "per_question": per_question,
        }

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        df = result.to_pandas()
        per_question = [
            EvalResult(
                question=row["question"],
                answer=row["answer"],
                contexts=list(row["contexts"]),
                ground_truth=row["ground_truth"],
                faithfulness=float(row.get("faithfulness", 0.0) or 0.0),
                answer_relevancy=float(row.get("answer_relevancy", 0.0) or 0.0),
                context_precision=float(row.get("context_precision", 0.0) or 0.0),
                context_recall=float(row.get("context_recall", 0.0) or 0.0),
            )
            for _, row in df.iterrows()
        ]
    except Exception as exc:
        print(f"  RAGAS evaluation failed, using lexical fallback: {exc}")
        per_question = _evaluate_lexical(questions, answers, contexts, ground_truths)

    results = {
        "faithfulness": _mean([r.faithfulness for r in per_question]),
        "answer_relevancy": _mean([r.answer_relevancy for r in per_question]),
        "context_precision": _mean([r.context_precision for r in per_question]),
        "context_recall": _mean([r.context_recall for r in per_question]),
        "per_question": per_question,
    }
    if any(math.isnan(value) for key, value in results.items() if key != "per_question"):
        print("  RAGAS returned NaN, using lexical fallback.")
        per_question = _evaluate_lexical(questions, answers, contexts, ground_truths)
        results = {
            "faithfulness": _mean([r.faithfulness for r in per_question]),
            "answer_relevancy": _mean([r.answer_relevancy for r in per_question]),
            "context_precision": _mean([r.context_precision for r in per_question]),
            "context_recall": _mean([r.context_recall for r in per_question]),
            "per_question": per_question,
        }
    return results


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using a diagnostic tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten the prompt, cite context, and lower temperature."),
        "answer_relevancy": ("Answer does not match the question", "Improve the answer prompt and add query rewriting."),
        "context_precision": ("Too many irrelevant chunks", "Add reranking, metadata filters, or smaller child chunks."),
        "context_recall": ("Missing relevant chunks", "Improve chunking, enrich context, or expand hybrid search."),
    }
    rows = []
    for result in eval_results:
        metrics = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        }
        avg_score = _mean(list(metrics.values()))
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        rows.append(
            {
                "question": result.question,
                "worst_metric": worst_metric,
                "score": avg_score,
                "diagnosis": diagnosis,
                "suggested_fix": suggested_fix,
            }
        )
    rows.sort(key=lambda row: row["score"])
    return rows[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON."""
    per_question = results.get("per_question", [])
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(per_question),
        "per_question": [asdict(row) if isinstance(row, EvalResult) else row for row in per_question],
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
