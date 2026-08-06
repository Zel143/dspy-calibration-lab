"""
Dataset loading utilities.

Expected format: JSONL, one object per line, with at minimum
"question" and "gold_answer" keys. This is intentionally generic so you
can later point it at exported cases from your stylometry classifiers
(question = "who wrote this passage", gold_answer = the labeled author)
instead of the toy trivia set.
"""

import json
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No dataset found at {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_num} of {path}: {e}")

    return records
