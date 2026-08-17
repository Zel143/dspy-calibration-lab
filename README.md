# calibration-lab

*A DSPy playground testing whether LLM confidence framing tracks actual correctness (the Overtrust Index question), comparing direct vs. reasoned confidence-elicitation strategies via Brier score and ECE.*

A small DSPy playground for exploring the **Overtrust Index (OTI)** question:
does an LLM's linguistic confidence framing ("I'm certain that...", "It's possible
that...") track its actual correctness, or are the two independent?

This repo is deliberately minimal. The goal isn't a production pipeline — it's a
place to run small experiments and build intuition for DSPy before extending
this into the full calibration-lab / OTI thesis work.

## Why DSPy specifically

Instead of hand-writing prompts and eyeballing outputs, DSPy treats the prompt
itself as an optimizable parameter. For this project that matters because:

- You can swap prompting strategies (direct answer vs. chain-of-thought vs.
  explicit confidence elicitation) without rewriting everything by hand.
- DSPy's `Signature` objects force you to be explicit about what "confidence"
  means as a typed output field, instead of parsing it out of free text.
- Later, if you plug in retrieval over your stylometry / biblical NLP corpora,
  DSPy's `Module` composition makes that a natural extension rather than a
  rewrite.

## Repo structure

```
calibration-lab/
├── src/
│   ├── signatures.py    # DSPy Signatures: question -> (answer, confidence)
│   ├── pipeline.py       # DSPy Module wiring the signatures together
│   ├── confidence.py     # Calibration metrics: ECE, Brier score, reliability data
│   └── data.py            # Dataset loading utilities
├── examples/
│   └── run_basic.py      # End-to-end: ask questions, log confidence vs. correctness
├── data/
│   └── sample_questions.jsonl   # Toy seed dataset (replace with real data)
├── tests/
│   └── test_confidence.py
├── requirements.txt
└── .gitignore
```

## Setup

```bash
git clone <your-new-repo-url>
cd calibration-lab
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You'll need an LLM API key. DSPy supports OpenAI, Anthropic, Gemini, and
others via `litellm` under the hood. Set whichever you're using:

```bash
export ANTHROPIC_API_KEY="your-key-here"
# or
export OPENAI_API_KEY="your-key-here"
# or
export GEMINI_API_KEY="your-key-here"
```

Gemini has a free tier — grab a key at https://aistudio.google.com/apikey.

## Running the basic experiment

```bash
python examples/run_basic.py
# with Gemini:
python examples/run_basic.py --model gemini/gemma-4-26b-a4b-it
```

Note: new Gemini API keys sometimes have zero free-tier quota for the
`gemini-2.x`/`gemini-1.5` models (429/404 errors), but do have quota for
Gemma models. If a `gemini-*` model fails, try a `gemma-*` one instead.

This will:
1. Load the toy question set in `data/sample_questions.jsonl`
2. Run each question through the DSPy pipeline, which returns both an answer
   and a self-reported confidence score
3. Compare confidence against actual correctness
4. Print calibration metrics (Expected Calibration Error, Brier score) and a
   reliability breakdown by confidence bucket

## Extending toward the actual OTI thesis

Some directions once you're comfortable with the basics:

- **Swap in your stylometry classifiers as seed data.** Replace
  `data/sample_questions.jsonl` with cases where you already have ground-truth
  labels (e.g. disputed-authorship classification results) and see whether
  confidence framing shifts independent of correctness on domain-specific
  questions.
- **Try different Signature designs.** Right now confidence is a single
  0–1 float. Try categorical confidence ("high/medium/low") vs. numeric, and
  see if the calibration behavior changes — that's a finding in itself.
- **Add a DSPy optimizer.** `dspy.BootstrapFewShot` or similar can be pointed
  at a metric function (e.g. minimizing ECE) to see whether the model can be
  taught to be *better calibrated*, not just more accurate.
- **Log raw completions, not just parsed fields.** Worth keeping the full
  transcript around for qualitative review — the OTI thesis cares about
  *how* confidence is expressed linguistically, not just the number.

## Reference

DSPy: https://github.com/stanfordnlp/dspy
