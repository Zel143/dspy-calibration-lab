"""
Calibration metrics for comparing self-reported confidence against actual
correctness. This is the core measurement layer for the Overtrust Index
question: is confidence framing informative, or decorative?
"""

from dataclasses import dataclass


@dataclass
class Prediction:
    question: str
    answer: str
    confidence: float  # 0.0 - 1.0
    correct: bool       # ground truth, filled in after scoring


def brier_score(predictions: list[Prediction]) -> float:
    """Mean squared error between confidence and correctness (0 = perfect).

    Lower is better. A model that says 0.9 confidence and is right scores
    well; a model that says 0.9 confidence and is wrong is penalized hard.
    """
    if not predictions:
        return float("nan")
    total = sum(
        (p.confidence - (1.0 if p.correct else 0.0)) ** 2 for p in predictions
    )
    return total / len(predictions)


def expected_calibration_error(
    predictions: list[Prediction], n_bins: int = 10
) -> float:
    """Expected Calibration Error (ECE).

    Buckets predictions by confidence, then measures the gap between average
    confidence and actual accuracy within each bucket, weighted by bucket
    size. A perfectly calibrated model (confidence == accuracy in every
    bucket) scores 0.
    """
    if not predictions:
        return float("nan")

    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    total_error = 0.0
    n = len(predictions)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        bucket = [
            p for p in predictions
            if lo <= p.confidence < hi or (i == n_bins - 1 and p.confidence == hi)
        ]
        if not bucket:
            continue
        avg_confidence = sum(p.confidence for p in bucket) / len(bucket)
        accuracy = sum(1 for p in bucket if p.correct) / len(bucket)
        total_error += (len(bucket) / n) * abs(avg_confidence - accuracy)

    return total_error


def reliability_breakdown(
    predictions: list[Prediction], n_bins: int = 5
) -> list[dict]:
    """Human-readable per-bucket breakdown: confidence range, count,
    avg confidence, actual accuracy. Print this to see calibration behavior
    at a glance rather than just a single summary number.
    """
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    rows = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        bucket = [
            p for p in predictions
            if lo <= p.confidence < hi or (i == n_bins - 1 and p.confidence == hi)
        ]
        if not bucket:
            rows.append({
                "range": f"{lo:.1f}-{hi:.1f}",
                "count": 0,
                "avg_confidence": None,
                "accuracy": None,
            })
            continue
        rows.append({
            "range": f"{lo:.1f}-{hi:.1f}",
            "count": len(bucket),
            "avg_confidence": round(sum(p.confidence for p in bucket) / len(bucket), 3),
            "accuracy": round(sum(1 for p in bucket if p.correct) / len(bucket), 3),
        })

    return rows
