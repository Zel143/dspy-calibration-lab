"""Persist run outputs to results/ so strategy/model/dataset comparisons
survive past the terminal instead of only ever being printed once.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path

from src.confidence import Prediction, brier_score, expected_calibration_error

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"
SUMMARY_FIELDS = [
    "run_id", "timestamp", "strategy", "model", "dataset",
    "n", "accuracy", "brier", "ece",
]


def _slug(text: str) -> str:
    return text.replace("/", "-").replace(" ", "_")


def save_run(
    strategy: str,
    model: str,
    dataset: str,
    predictions: list[Prediction],
) -> str:
    """Write per-run predictions.csv + report.txt under results/runs/, and
    append a summary row to results/summary.csv. Returns the run_id.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    runs_dir = RESULTS_DIR / "runs"
    runs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{_slug(strategy)}_{_slug(model)}_{_slug(Path(dataset).stem)}"
    run_dir = runs_dir / run_id
    run_dir.mkdir()

    n = len(predictions)
    accuracy = sum(p.correct for p in predictions) / n if n else float("nan")
    brier = brier_score(predictions)
    ece = expected_calibration_error(predictions)

    with open(run_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "answer", "confidence", "correct"])
        for p in predictions:
            writer.writerow([p.id, p.question, p.answer, p.confidence, p.correct])

    with open(run_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write(f"Strategy:      {strategy}\n")
        f.write(f"Model:         {model}\n")
        f.write(f"Dataset:       {dataset}\n")
        f.write(f"N questions:   {n}\n")
        f.write(f"Accuracy:      {accuracy:.2%}\n")
        f.write(f"Brier score:   {brier:.4f}\n")
        f.write(f"ECE:           {ece:.4f}\n")

    is_new = not SUMMARY_CSV.exists()
    with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "run_id": run_id,
            "timestamp": timestamp,
            "strategy": strategy,
            "model": model,
            "dataset": dataset,
            "n": n,
            "accuracy": round(accuracy, 4),
            "brier": round(brier, 4),
            "ece": round(ece, 4),
        })

    return run_id
