"""
DSPy Modules that wire signatures into runnable pipelines.

A Module is where you decide *how* a signature gets executed — direct
prediction, chain-of-thought, retrieval-augmented, etc. Keeping this
separate from signatures.py means you can benchmark multiple strategies
against the same evaluation harness.
"""

import dspy

from src.signatures import AnswerWithConfidence, AnswerWithReasonedConfidence


class DirectConfidencePipeline(dspy.Module):
    """Asks for an answer and confidence in a single step, no explicit reasoning."""

    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(AnswerWithConfidence)

    def forward(self, question: str):
        result = self.predict(question=question)
        return result


class ReasonedConfidencePipeline(dspy.Module):
    """Asks for chain-of-thought reasoning before answer and confidence.

    Comparing this against DirectConfidencePipeline is the first useful
    experiment: does forcing reasoning before the confidence estimate
    produce better-calibrated confidence, or just longer, more convincing
    -sounding wrong answers?
    """

    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(AnswerWithReasonedConfidence)

    def forward(self, question: str):
        result = self.predict(question=question)
        return result


def get_pipeline(strategy: str = "direct") -> dspy.Module:
    """Factory so examples/run_basic.py can switch strategies with a flag."""
    strategies = {
        "direct": DirectConfidencePipeline,
        "reasoned": ReasonedConfidencePipeline,
    }
    if strategy not in strategies:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from: {list(strategies)}"
        )
    return strategies[strategy]()
