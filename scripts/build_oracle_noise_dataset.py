"""Cross-repo bridge, step 2: the TempleOS oracle test for LLMs.

TempleOS (Terry A. Davis) shipped an "oracle" that drew random words and
presented them as divine speech, and readers routinely found meaning in the
output: apophenia, perceiving pattern in noise. prophetic-register-classifier
already asked the machine version for its stylometric classifier
(src/null_control.py: the classifier calls 23-37% of pure noise
"prophetic"). This script asks the same question of an LLM, which also
reports a confidence: when the input has no right answer, is the model
still sure of itself? Any confident reading of noise is overtrust by
construction, which makes this a clean test case for the Overtrust Index.

Items are drawn from the classifier repo's own files, so the LLM sees the
exact chunks the classifier scored and the two can be compared item by item:

  word_salad_uniform  random KJV word types (closest to the TempleOS oracle)
  word_salad_freq     random KJV tokens, frequency-weighted
  verse_shuffle       real KJV verses with word order shuffled
  verse_flat          the *same* verses as verse_shuffle, word order intact
                      (paired by chunk id): meaningful, register unlabeled
  real_verse          labeled seed-set verses: the positive control, where
                      a right register answer exists

Every item is in the same surface form (lowercase, no punctuation), as the
classifier repo's chunks already are, so the model cannot spot noise from
capitals or commas alone. It has to read the words.

The same items are written out under four prompt framings, so every
comparison between framings is paired:

  oracle_forced.jsonl         3-way register question, no way out. On noise
                              a calibrated model has nothing to go on, so
                              its confidence should sit near 1/3.
  oracle_abstain.jsonl        same, plus "none" as an allowed answer. Gold
                              for noise is "none".
  oracle_detect_neutral.jsonl "meaningful or noise?" asked plainly.
  oracle_detect_primed.jsonl  the same question, but the text is introduced
                              the way TempleOS introduced it: as words drawn
                              by an oracle that carry a message. The gap
                              between neutral and primed is the priming
                              effect.

This measures how a model reports confidence about textual style and
structure. It says nothing about any text's truth, inspiration or
scriptural status.

Usage (from the repo root, with prophetic-register-classifier checked out
next to this repo):
    python scripts/build_oracle_noise_dataset.py
    python scripts/build_oracle_noise_dataset.py --n-per-condition 40 --seed 143
    python scripts/build_oracle_noise_dataset.py --classifier-repo G:/prophetic-register-classifier
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLASSIFIER_REPO = REPO_ROOT.parent / "prophetic-register-classifier"

NOISE_CONDITIONS = ["word_salad_uniform", "word_salad_freq", "verse_shuffle"]
CONTROL_CONDITION = "real_verse"
CONDITIONS = NOISE_CONDITIONS + ["verse_flat", CONTROL_CONDITION]
# Same tokenizer as prophetic-register-classifier/src/extract_features.py.
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
REGISTERS = ["prophetic", "narrative", "law-wisdom"]

FRAMINGS = {
    "forced": {
        "classes": REGISTERS,
        "template": (
            'What register is this King James Version text: prophetic, narrative, '
            'or law-wisdom? Answer with exactly one of those three words. Text: "{text}"'
        ),
    },
    "abstain": {
        "classes": REGISTERS + ["none"],
        "template": (
            'What register is this King James Version text: prophetic, narrative, '
            'or law-wisdom? If the text is not a coherent passage and has no register, '
            'answer none. Answer with exactly one of: prophetic, narrative, law-wisdom, '
            'none. Text: "{text}"'
        ),
    },
    "detect_neutral": {
        "classes": ["meaningful", "noise"],
        "template": (
            'Is the following text a meaningful, intentionally composed passage, or '
            'randomly generated noise? Answer with exactly one word: meaningful or '
            'noise. Text: "{text}"'
        ),
    },
    "detect_primed": {
        "classes": ["meaningful", "noise"],
        "template": (
            'The following words were drawn by an oracle and are said to carry a '
            'message. Is the text a meaningful, intentionally composed passage, or '
            'randomly generated noise? Answer with exactly one word: meaningful or '
            'noise. Text: "{text}"'
        ),
    },
}


def flatten(text: str) -> str:
    return " ".join(WORD_RE.findall(text.lower().replace("\u2019", "'")))


def gold_for(framing: str, condition: str, label: str | None) -> str:
    """Gold answer per framing. "" means no single right answer exists
    (noise under the forced framing, or a meaningful verse whose register
    was never labeled); analyze_oracle_noise.py handles those cases."""
    is_noise = condition in NOISE_CONDITIONS
    if framing in ("forced", "abstain"):
        if is_noise:
            return "" if framing == "forced" else "none"
        return label or ""
    return "noise" if is_noise else "meaningful"


def build_items(classifier_repo: Path, n_per_condition: int, seed: int) -> list[dict]:
    preds = pd.read_csv(classifier_repo / "results" / "null_control_predictions.csv")
    seed_set = pd.read_csv(classifier_repo / "data" / "seed_set.csv")

    items = []
    picked = {}
    for condition in NOISE_CONDITIONS:
        pool = preds[preds["condition"] == condition]
        picked[condition] = pool.sample(n=n_per_condition, random_state=seed)
    # verse_flat takes the same chunk ids as verse_shuffle: each shuffled
    # verse has its unshuffled twin in the set.
    flat = preds[preds["condition"] == "verse_flat"].set_index("chunk_id")
    picked["verse_flat"] = flat.loc[picked["verse_shuffle"]["chunk_id"]].reset_index()

    for condition in NOISE_CONDITIONS + ["verse_flat"]:
        for _, row in picked[condition].iterrows():
            items.append({
                "id": f"{condition}:{int(row['chunk_id'])}",
                "condition": condition,
                "source_ref": row["source_ref"] if isinstance(row["source_ref"], str) else "",
                "text": row["text"],
                "label": None,
                "clf_pred_full": row["pred_full"],
                "clf_pred_normttr": row["pred_normttr"],
            })

    # Positive control, stratified by register so all three are present.
    per_label = n_per_condition // len(REGISTERS)
    counts = {lab: per_label for lab in REGISTERS}
    counts[REGISTERS[0]] += n_per_condition - per_label * len(REGISTERS)
    for label, k in counts.items():
        pool = seed_set[seed_set["label"] == label]
        for _, row in pool.sample(n=k, random_state=seed).iterrows():
            items.append({
                "id": f"{CONTROL_CONDITION}:{row['ref']}",
                "condition": CONTROL_CONDITION,
                "source_ref": row["ref"],
                "text": flatten(row["text"]),
                "label": label,
                "clf_pred_full": "",
                "clf_pred_normttr": "",
            })

    # One fixed shuffled order, shared by every framing.
    order = pd.Series(range(len(items))).sample(frac=1, random_state=seed).tolist()
    return [items[i] for i in order]


def write_framings(items: list[dict], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for framing, spec in FRAMINGS.items():
        path = out_dir / f"oracle_{framing}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                record = {
                    "id": item["id"],
                    "question": spec["template"].format(text=item["text"]),
                    "gold_answer": gold_for(framing, item["condition"], item["label"]),
                    "framing": framing,
                    "condition": item["condition"],
                    "source_ref": item["source_ref"],
                    "clf_pred_full": item["clf_pred_full"],
                    "clf_pred_normttr": item["clf_pred_normttr"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        paths[framing] = path
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier-repo", type=Path, default=DEFAULT_CLASSIFIER_REPO)
    parser.add_argument("--n-per-condition", type=int, default=40)
    parser.add_argument("--seed", type=int, default=143)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data")
    args = parser.parse_args()

    items = build_items(args.classifier_repo, args.n_per_condition, args.seed)
    paths = write_framings(items, args.out_dir)
    counts = pd.Series([i["condition"] for i in items]).value_counts()
    print(f"Wrote {len(items)} items x {len(paths)} framings:")
    for framing, path in paths.items():
        print(f"  {framing:<15} {path}  --classes {','.join(FRAMINGS[framing]['classes'])}")
    print(counts.to_string())


if __name__ == "__main__":
    main()
