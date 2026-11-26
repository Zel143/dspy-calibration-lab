"""Analyze runs on the oracle-noise datasets (see build_oracle_noise_dataset.py).

run_basic.py's own summary line treats every item alike, which is the wrong
reading here: under the forced framing a noise item has no right answer at
all. This script joins each run's predictions.csv back to its dataset by id
and reports per condition:

  forced          label mix, mean confidence, and the overtrust gap on noise
                  (mean confidence minus 1/3, the most a calibrated model can
                  justify when three answers are equally unsupported);
                  prophetic share against the classifier's noise floor;
                  agreement with the classifier on the same noise chunks;
                  shuffled-vs-unshuffled agreement for the same verses
  abstain         how often the model says "none" on noise, and how often it
                  wrongly says "none" on real verses
  detect_*        accuracy, Brier, ECE, and the confident-apophenia rate:
                  share of noise items called "meaningful" at confidence
                  >= 0.8
  primed vs.      paired change in how often noise is called meaningful
  neutral         when the text is introduced as an oracle's message

Usage:
    python scripts/analyze_oracle_noise.py                # latest run per framing
    python scripts/analyze_oracle_noise.py --runs results/runs/<run_id> ...
    python scripts/analyze_oracle_noise.py --model gemini/gemma-4-31b-it --strategy direct
"""

import argparse
import csv
import json
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.confidence import Prediction, brier_score, expected_calibration_error  # noqa: E402
from src.scoring import classify_answer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "results" / "runs"
REPORT_PATH = REPO_ROOT / "results" / "oracle_noise_report.md"

FRAMING_CLASSES = {
    "forced": ["prophetic", "narrative", "law-wisdom"],
    "abstain": ["prophetic", "narrative", "law-wisdom", "none"],
    "detect_neutral": ["meaningful", "noise"],
    "detect_primed": ["meaningful", "noise"],
}
NOISE = ["word_salad_uniform", "word_salad_freq", "verse_shuffle"]
CONDITIONS = NOISE + ["verse_flat", "real_verse"]
CONFIDENT = 0.8

# From prophetic-register-classifier/results/null_control_summary.txt
# (full model, n=1000 per condition): the classifier's prophetic share on
# the noisiest-looking condition it was fooled by.
CLASSIFIER_FLOOR_FULL = 0.345


def load_dataset(framing: str) -> dict[str, dict]:
    path = REPO_ROOT / "data" / f"oracle_{framing}.jsonl"
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in (json.loads(line) for line in f if line.strip())}


def load_run(run_dir: Path) -> list[dict]:
    with open(run_dir / "predictions.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def framing_of(run_dir: Path) -> str | None:
    for framing in FRAMING_CLASSES:
        if run_dir.name.endswith(f"_oracle_{framing}"):
            return framing
    return None


def latest_runs(model: str | None, strategy: str | None) -> dict[str, Path]:
    found: dict[str, Path] = {}
    if not RUNS_DIR.exists():
        return found
    for run_dir in sorted(RUNS_DIR.iterdir()):  # run ids start with a timestamp
        framing = framing_of(run_dir)
        if framing is None:
            continue
        if strategy and f"_{strategy}_" not in run_dir.name:
            continue
        if model and model.replace("/", "-") not in run_dir.name:
            continue
        found[framing] = run_dir
    return found


def join(framing: str, run_dir: Path) -> list[dict]:
    dataset = load_dataset(framing)
    classes = FRAMING_CLASSES[framing]
    rows = []
    for p in load_run(run_dir):
        item = dataset.get(p["id"])
        if item is None:
            continue
        rows.append({
            **item,
            "answer": p["answer"],
            "parsed": classify_answer(p["answer"], classes),
            "confidence": float(p["confidence"]),
        })
    return rows


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant-pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def share(rows, pred):
    return mean(1.0 if pred(r) else 0.0 for r in rows)


def fmt(x, pct=False):
    if x != x:  # NaN
        return "-"
    return f"{x:.1%}" if pct else f"{x:.3f}"


def by_condition(rows):
    return {c: [r for r in rows if r["condition"] == c] for c in CONDITIONS}


def label_mix_table(rows, classes):
    groups = by_condition(rows)
    header = "| condition | n | " + " | ".join(classes) + " | unparsed | mean conf |"
    lines = [header, "|" + "---|" * (len(classes) + 4)]
    for c, rs in groups.items():
        if not rs:
            continue
        cells = [fmt(share(rs, lambda r, k=k: r["parsed"] == k), pct=True) for k in classes]
        lines.append(
            f"| {c} | {len(rs)} | " + " | ".join(cells)
            + f" | {fmt(share(rs, lambda r: r['parsed'] is None), pct=True)}"
            + f" | {fmt(mean(r['confidence'] for r in rs))} |"
        )
    return lines


def section_forced(rows):
    out = ["## Forced choice (no way out)", ""]
    out += label_mix_table(rows, FRAMING_CLASSES["forced"]) + [""]
    noise = [r for r in rows if r["condition"] in NOISE]
    gap = mean(r["confidence"] for r in noise) - 1 / 3
    out.append(
        f"- **Overtrust gap on noise:** mean confidence {fmt(mean(r['confidence'] for r in noise))} "
        f"vs. 0.333 justified, gap **{gap:+.3f}**. "
        f"{fmt(share(noise, lambda r: r['confidence'] >= CONFIDENT), pct=True)} of noise answers "
        f"come at confidence >= {CONFIDENT}."
    )
    for c in NOISE:
        rs = [r for r in rows if r["condition"] == c]
        if not rs:
            continue
        agree = share(rs, lambda r: r["parsed"] == r["clf_pred_full"])
        out.append(
            f"- {c}: prophetic share {fmt(share(rs, lambda r: r['parsed'] == 'prophetic'), pct=True)} "
            f"(classifier floor {CLASSIFIER_FLOOR_FULL:.1%}); agrees with the classifier's label "
            f"on {fmt(agree, pct=True)} of the same chunks."
        )
    # Twins share a chunk id: verse_shuffle:<k> is verse_flat:<k> shuffled.
    shuf = {r["id"].split(":", 1)[1]: r for r in rows if r["condition"] == "verse_shuffle"}
    flat = {r["id"].split(":", 1)[1]: r for r in rows if r["condition"] == "verse_flat"}
    pairs = [(shuf[k], flat[k]) for k in shuf.keys() & flat.keys()]
    if pairs:
        same = mean(1.0 if a["parsed"] == b["parsed"] else 0.0 for a, b in pairs)
        out.append(
            f"- Shuffled vs. unshuffled twin agree on {fmt(same, pct=True)} of {len(pairs)} pairs "
            f"(classifier full model: 85.3%). Lower means the LLM reads word order."
        )
    real = [r for r in rows if r["condition"] == "real_verse"]
    if real:
        preds = [Prediction(r["question"], r["answer"], r["confidence"], r["parsed"] == r["gold_answer"]) for r in real]
        out.append(
            f"- Positive control (labeled verses): accuracy "
            f"{fmt(mean(p.correct for p in preds), pct=True)}, Brier {fmt(brier_score(preds))}, "
            f"ECE {fmt(expected_calibration_error(preds))}."
        )
    return out + [""]


def section_abstain(rows):
    out = ["## Abstention allowed (\"none\")", ""]
    out += label_mix_table(rows, FRAMING_CLASSES["abstain"]) + [""]
    noise = [r for r in rows if r["condition"] in NOISE]
    meaningful = [r for r in rows if r["condition"] in ("verse_flat", "real_verse")]
    answered = [r for r in noise if r["parsed"] not in (None, "none")]
    out.append(f"- Says \"none\" on {fmt(share(noise, lambda r: r['parsed'] == 'none'), pct=True)} of noise "
               f"(ideal 100%) and on {fmt(share(meaningful, lambda r: r['parsed'] == 'none'), pct=True)} "
               f"of real verses (ideal 0%).")
    out.append(f"- When it does give noise a register anyway, mean confidence is "
               f"{fmt(mean(r['confidence'] for r in answered))} (n={len(answered)}).")
    return out + [""]


def section_detect(rows, framing):
    title = "Detection, primed as an oracle's message" if framing == "detect_primed" else "Detection, neutral wording"
    out = [f"## {title}", ""]
    out += label_mix_table(rows, FRAMING_CLASSES[framing]) + [""]
    preds = [Prediction(r["question"], r["answer"], r["confidence"], r["parsed"] == r["gold_answer"]) for r in rows]
    noise = [r for r in rows if r["condition"] in NOISE]
    out.append(f"- Overall accuracy {fmt(mean(p.correct for p in preds), pct=True)}, "
               f"Brier {fmt(brier_score(preds))}, ECE {fmt(expected_calibration_error(preds))}.")
    out.append(f"- **Confident apophenia:** {fmt(share(noise, lambda r: r['parsed'] == 'meaningful' and r['confidence'] >= CONFIDENT), pct=True)} "
               f"of noise items called meaningful at confidence >= {CONFIDENT}.")
    return out + [""]


def section_priming(neutral, primed):
    n = {r["id"]: r for r in neutral}
    p = {r["id"]: r for r in primed}
    ids = [i for i in n.keys() & p.keys() if n[i]["condition"] in NOISE]
    if not ids:
        return []
    to_meaning = sum(1 for i in ids if n[i]["parsed"] != "meaningful" and p[i]["parsed"] == "meaningful")
    to_noise = sum(1 for i in ids if n[i]["parsed"] == "meaningful" and p[i]["parsed"] != "meaningful")
    rate_n = share([n[i] for i in ids], lambda r: r["parsed"] == "meaningful")
    rate_p = share([p[i] for i in ids], lambda r: r["parsed"] == "meaningful")
    return [
        "## Priming effect (same noise items, two framings)", "",
        f"- Noise called meaningful: {fmt(rate_n, pct=True)} neutral -> {fmt(rate_p, pct=True)} primed "
        f"(n={len(ids)} paired items).",
        f"- Flips: {to_meaning} noise items became \"meaningful\" when primed, {to_noise} went the other way "
        f"(exact McNemar p = {mcnemar_exact(to_meaning, to_noise):.3g}).",
        "",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", type=Path, help="Run directories to analyze")
    parser.add_argument("--model", help="Filter auto-picked runs by model id")
    parser.add_argument("--strategy", help="Filter auto-picked runs by strategy (direct/reasoned)")
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    if args.runs:
        runs = {framing_of(r): r for r in args.runs if framing_of(r)}
    else:
        runs = latest_runs(args.model, args.strategy)
    if not runs:
        sys.exit("No oracle-noise runs found. Run examples/run_basic.py on data/oracle_*.jsonl first "
                 "(see docs/oracle-noise.md).")

    joined = {framing: join(framing, run_dir) for framing, run_dir in runs.items()}
    lines = ["# Oracle-noise results", ""]
    lines += [f"- `{framing}`: `{run_dir.name}`" for framing, run_dir in runs.items()] + [""]
    if "forced" in joined:
        lines += section_forced(joined["forced"])
    if "abstain" in joined:
        lines += section_abstain(joined["abstain"])
    for framing in ("detect_neutral", "detect_primed"):
        if framing in joined:
            lines += section_detect(joined[framing], framing)
    if "detect_neutral" in joined and "detect_primed" in joined:
        lines += section_priming(joined["detect_neutral"], joined["detect_primed"])

    report = "\n".join(lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
