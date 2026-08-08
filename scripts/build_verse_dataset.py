"""Cross-repo bridge, step 1: convert the outer repo's labeled KJV verses
into a JSONL question set this harness can run, so the LLM's self-reported
confidence and the outer repo's calibrated classifier `predict_proba` can be
compared on the *same* verses.

Source: G:/prophetic-register-classifier/results/seed_set_features.csv
(510 verses, 3-class: prophetic / narrative / law-wisdom). Only `text`,
`label`, and `ref` are used here — the 130 stylometric feature columns are
irrelevant to an LLM given the raw verse text.

Draws a class-stratified subsample (default 60, matching the source's
~36/35/29% class split) rather than all 510, to keep wall-clock time and
free-tier API usage bounded -- see calibration-lab (outer repo) context
notes for why: a 50-question `reasoned` run already takes a long time on
the free tier.

Usage:
    python scripts/build_verse_dataset.py
    python scripts/build_verse_dataset.py --n 60 --seed 42
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SOURCE_CSV = "G:/prophetic-register-classifier/results/seed_set_features.csv"
CLASSES = ["prophetic", "narrative", "law-wisdom"]

QUESTION_TEMPLATE = (
    'What register is this King James Version verse: prophetic, narrative, '
    'or law-wisdom? Answer with exactly one of those three words. Verse: "{text}"'
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=SOURCE_CSV)
    parser.add_argument("--n", type=int, default=60, help="Total sample size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out", default="data/verse_register.jsonl",
        help="Output path, relative to repo root"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.source)

    # Proportional per-class sample size, rounded, with rounding drift
    # corrected on the largest class so the total is exactly args.n.
    counts = df["label"].value_counts()
    proportions = counts / counts.sum()
    per_class_n = (proportions * args.n).round().astype(int)
    drift = args.n - per_class_n.sum()
    if drift != 0:
        per_class_n[per_class_n.idxmax()] += drift

    sampled = pd.concat([
        df[df["label"] == label].sample(n=n, random_state=args.seed)
        for label, n in per_class_n.items()
    ]).sample(frac=1, random_state=args.seed)  # shuffle across classes

    out_path = Path(__file__).resolve().parent.parent / args.out
    out_path.parent.mkdir(exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in sampled.iterrows():
            record = {
                "id": row["ref"],
                "question": QUESTION_TEMPLATE.format(text=row["text"]),
                "gold_answer": row["label"],
            }
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(sampled)} verses to {out_path}")
    print(sampled["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
