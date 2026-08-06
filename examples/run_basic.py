"""
End-to-end calibration experiment.

Loads the toy question set, runs each question through a DSPy pipeline that
returns an answer + self-reported confidence, scores correctness against the
gold answer, and prints calibration metrics.

Usage:
    python examples/run_basic.py
    python examples/run_basic.py --strategy reasoned
    python examples/run_basic.py --model anthropic/claude-sonnet-4-5
"""

import argparse
import os
import sys
from pathlib import Path

# Allow running as `python examples/run_basic.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dspy
from dotenv import load_dotenv

from src.confidence import Prediction, brier_score, expected_calibration_error, reliability_breakdown
from src.data import load_jsonl
from src.pipeline import get_pipeline


def is_correct(model_answer: str, gold_answer: str) -> bool:
    """Very naive substring match. Good enough for the toy dataset — swap
    for something smarter (embedding similarity, exact-match on normalized
    labels, etc.) once you're using real evaluation data.
    """
    return gold_answer.strip().lower() in model_answer.strip().lower()


def main():
    load_dotenv()

    # Windows consoles default to cp1252, which chokes on non-ASCII
    # characters that can show up in LM output (accents, symbols, etc.).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy", default="direct", choices=["direct", "reasoned"],
        help="Which pipeline strategy to run"
    )
    parser.add_argument(
        "--model", default="anthropic/claude-sonnet-4-5",
        help="LM identifier passed to dspy.LM, e.g. 'openai/gpt-4o' or 'anthropic/claude-sonnet-4-5'"
    )
    parser.add_argument(
        "--data", default="data/sample_questions.jsonl",
        help="Path to a JSONL dataset with 'question' and 'gold_answer' fields"
    )
    args = parser.parse_args()

    if not any(
        os.environ.get(var)
        for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")
    ):
        print(
            "Warning: no ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY found in "
            "environment. Set one before running, e.g.:\n  export GEMINI_API_KEY='your-key'\n"
        )

    lm = dspy.LM(args.model)
    dspy.configure(lm=lm)

    pipeline = get_pipeline(args.strategy)
    dataset = load_jsonl(Path(__file__).resolve().parent.parent / args.data)

    predictions: list[Prediction] = []

    print(f"Running {len(dataset)} questions through '{args.strategy}' pipeline...\n")

    for row in dataset:
        question = row["question"]
        gold = row["gold_answer"]

        result = pipeline(question=question)

        # Confidence sometimes comes back as a string from the LM; coerce defensively.
        try:
            confidence = float(result.confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        correct = is_correct(result.answer, gold)

        predictions.append(
            Prediction(question=question, answer=result.answer, confidence=confidence, correct=correct)
        )

        status = "[OK]" if correct else "[X]"
        print(f"  {status} conf={confidence:.2f}  Q: {question[:60]}")
        print(f"      A: {result.answer[:100]}")

    print("\n--- Calibration Summary ---")
    print(f"Strategy:      {args.strategy}")
    print(f"N questions:   {len(predictions)}")
    print(f"Accuracy:      {sum(p.correct for p in predictions) / len(predictions):.2%}")
    print(f"Brier score:   {brier_score(predictions):.4f}  (lower is better, 0 = perfect)")
    print(f"ECE:           {expected_calibration_error(predictions):.4f}  (lower is better, 0 = perfect)")

    print("\n--- Reliability by confidence bucket ---")
    for row in reliability_breakdown(predictions):
        if row["count"] == 0:
            continue
        print(
            f"  conf {row['range']}: n={row['count']:<3} "
            f"avg_conf={row['avg_confidence']:<6} accuracy={row['accuracy']}"
        )


if __name__ == "__main__":
    main()
