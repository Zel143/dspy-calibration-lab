import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.confidence import Prediction, brier_score, expected_calibration_error


def test_brier_score_perfect_confidence():
    predictions = [
        Prediction(question="q1", answer="a1", confidence=1.0, correct=True),
        Prediction(question="q2", answer="a2", confidence=0.0, correct=False),
    ]
    assert brier_score(predictions) == 0.0


def test_brier_score_worst_case():
    predictions = [
        Prediction(question="q1", answer="a1", confidence=1.0, correct=False),
    ]
    assert brier_score(predictions) == 1.0


def test_ece_perfect_calibration():
    # 10 predictions all at confidence 0.7, and exactly 70% of them correct
    predictions = [
        Prediction(question=f"q{i}", answer=f"a{i}", confidence=0.7, correct=(i < 7))
        for i in range(10)
    ]
    ece = expected_calibration_error(predictions, n_bins=10)
    assert abs(ece) < 1e-9


def test_ece_detects_overconfidence():
    # All predictions claim high confidence but are all wrong
    predictions = [
        Prediction(question=f"q{i}", answer=f"a{i}", confidence=0.95, correct=False)
        for i in range(5)
    ]
    ece = expected_calibration_error(predictions, n_bins=10)
    assert ece > 0.9


def test_empty_predictions_do_not_crash():
    assert brier_score([]) != brier_score([])  # NaN != NaN, sanity check
    assert expected_calibration_error([]) != expected_calibration_error([])
