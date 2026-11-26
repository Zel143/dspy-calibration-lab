import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_oracle_noise import mcnemar_exact  # noqa: E402
from build_oracle_noise_dataset import FRAMINGS, NOISE_CONDITIONS, flatten, gold_for  # noqa: E402
from src.scoring import classify_answer  # noqa: E402


def load(framing):
    with open(ROOT / "data" / f"oracle_{framing}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_framings_share_items_in_same_order():
    ids = [[r["id"] for r in load(f)] for f in FRAMINGS]
    assert all(i == ids[0] for i in ids)
    assert len(set(ids[0])) == len(ids[0])


def test_every_shuffled_verse_has_its_unshuffled_twin():
    rows = load("forced")
    shuf = {r["id"].split(":", 1)[1]: r for r in rows if r["condition"] == "verse_shuffle"}
    flat = {r["id"].split(":", 1)[1]: r for r in rows if r["condition"] == "verse_flat"}
    assert shuf.keys() == flat.keys()
    for k in shuf:
        assert shuf[k]["source_ref"] == flat[k]["source_ref"]


def test_surface_form_matches_across_conditions():
    # No capitals or punctuation anywhere, so noise can't be spotted by format.
    for r in load("forced"):
        text = r["question"].rsplit('Text: "', 1)[1][:-1]
        assert text == flatten(text), r["id"]


def test_gold_answers():
    assert gold_for("forced", "word_salad_uniform", None) == ""
    assert gold_for("abstain", "verse_shuffle", None) == "none"
    assert gold_for("abstain", "real_verse", "prophetic") == "prophetic"
    assert gold_for("detect_primed", "word_salad_freq", None) == "noise"
    assert gold_for("detect_neutral", "verse_flat", None) == "meaningful"
    for r in load("detect_primed"):
        assert (r["gold_answer"] == "noise") == (r["condition"] in NOISE_CONDITIONS)


def test_classify_answer_uses_whole_words():
    classes = FRAMINGS["abstain"]["classes"]
    assert classify_answer("none", classes) == "none"
    assert classify_answer("Nonetheless, prophetic", classes) == "prophetic"
    assert classify_answer("Law-wisdom", classes) == "law-wisdom"
    assert classify_answer("prophetic or narrative", classes) is None


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert abs(mcnemar_exact(5, 5) - 1.0) < 1e-12
    assert mcnemar_exact(10, 0) < 0.002
