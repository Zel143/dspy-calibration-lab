"""
DSPy Signatures for the calibration-lab experiments.

A Signature is a typed contract: given these inputs, produce these outputs.
DSPy uses the field descriptions to build the underlying prompt, and you can
swap Signatures in and out of a Module without touching the rest of the
pipeline — that's the whole point of doing this in DSPy instead of raw
prompt strings.
"""

import dspy


class AnswerWithConfidence(dspy.Signature):
    """Answer the question, then state how confident you are in your own answer.

    Confidence should reflect genuine uncertainty about correctness, not just
    how fluently you can phrase the answer. Report it as a float between 0
    and 1, where 1.0 means you are certain the answer is correct.
    """

    question: str = dspy.InputField(desc="The question to answer")
    answer: str = dspy.OutputField(desc="A concise, direct answer to the question")
    confidence: float = dspy.OutputField(
        desc="Self-reported confidence in the answer's correctness, from 0.0 to 1.0"
    )


class AnswerWithReasonedConfidence(dspy.Signature):
    """Answer the question with explicit step-by-step reasoning, then give a
    confidence score grounded in that reasoning.

    This variant exists to test whether forcing chain-of-thought reasoning
    before the confidence estimate changes calibration compared to
    AnswerWithConfidence, which asks for confidence directly.
    """

    question: str = dspy.InputField(desc="The question to answer")
    reasoning: str = dspy.OutputField(
        desc="Step-by-step reasoning toward the answer"
    )
    answer: str = dspy.OutputField(desc="A concise, direct answer to the question")
    confidence: float = dspy.OutputField(
        desc="Self-reported confidence in the answer's correctness, from 0.0 to 1.0"
    )
