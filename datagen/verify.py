"""
Independent verification of the shipped CiteCast task file.

Structural checks run offline against data/citecast_tasks.jsonl. Live checks
re-fetch a seeded sample from the sources and verify GUARANTEES, not bytes:
citation counts drift upward between the snapshot day and the verify day (and
occasionally downward on Semantic Scholar merge events), so the live check is
a tolerance band, never equality. A rebuild will not reproduce the shipped
file byte for byte, and cannot — see DATA_UPLOAD.md.

    python verify.py                # structural only
    python verify.py --sample 60    # + live re-checks against arXiv and S2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import (  # noqa: E402
    BAND_COUNTS,
    SNAPSHOT,
    SPLIT_SALT,
    TEST_TOTAL,
    TOTAL_TASKS,
    TRAIN_TOTAL,
    VERIFY_SEED,
)
from grading import band_of  # noqa: E402

import common  # noqa: E402

DATA_FILE = Path(__file__).parent.parent / "data" / "citecast_tasks.jsonl"

_ROW_KEYS = (
    "task_id", "split", "band", "arxiv_id", "announced", "title", "abstract",
    "authors", "author_count", "primary_category", "true_citations",
    "s2_paper_id", "snapshot_utc", "provenance",
)

# Fields that must NOT exist anywhere in a row: they mutate on arXiv without a
# version bump and can carry post-cutoff information.
FORBIDDEN_KEYS = {"comment", "comments", "journal_ref", "doi", "categories"}


def load_rows() -> list[dict]:
    with open(DATA_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]


def check_structure(rows: list[dict]) -> list[str]:
    problems: list[str] = []

    if len(rows) != TOTAL_TASKS:
        problems.append(f"row count {len(rows)} != {TOTAL_TASKS}")

    for i, row in enumerate(rows):
        if tuple(row.keys()) != _ROW_KEYS:
            problems.append(f"row {i}: key set/order mismatch")
            break
    for row in rows:
        bad = FORBIDDEN_KEYS & set(row.keys())
        if bad:
            problems.append(f"{row['task_id']}: forbidden keys {bad}")

    band_counts = Counter(r["band"] for r in rows)
    for band, want in BAND_COUNTS.items():
        if band_counts.get(band, 0) != want:
            problems.append(f"band {band}: {band_counts.get(band, 0)} != {want}")

    split_counts = Counter(r["split"] for r in rows)
    if split_counts.get("train", 0) != TRAIN_TOTAL or split_counts.get("test", 0) != TEST_TOTAL:
        problems.append(f"split counts {dict(split_counts)} != {TRAIN_TOTAL}/{TEST_TOTAL}")
    test_bands = {r["band"] for r in rows if r["split"] == "test"}
    if test_bands != set(BAND_COUNTS):
        problems.append(f"test split missing bands: {set(BAND_COUNTS) - test_bands}")

    for field in ("task_id", "arxiv_id", "s2_paper_id"):
        values = [r[field] for r in rows]
        if len(set(values)) != len(values):
            problems.append(f"duplicate {field}")
    titles = [common.normalise_title(r["title"]) for r in rows]
    if len(set(titles)) != len(titles):
        problems.append("duplicate normalised titles")

    for row in rows:
        if row["task_id"] != f"cc_{row['arxiv_id']}":
            problems.append(f"{row['task_id']}: id does not match arxiv_id")
        if not row["arxiv_id"].startswith("2601."):
            problems.append(f"{row['task_id']}: not a January 2026 announcement")
        if band_of(row["true_citations"]) != row["band"]:
            problems.append(f"{row['task_id']}: count {row['true_citations']} outside band {row['band']}")
        if row["snapshot_utc"][:10] != SNAPSHOT:
            problems.append(f"{row['task_id']}: snapshot {row['snapshot_utc']} != {SNAPSHOT}")
        # Real abstracts can be one sentence (e.g. a 94-char abstract on a
        # streamlined Kakeya proof); only flag genuinely degenerate ones.
        if not row["abstract"] or len(row["abstract"]) < 60:
            problems.append(f"{row['task_id']}: abstract too short")
        if not row["title"] or not row["authors"]:
            problems.append(f"{row['task_id']}: missing title/authors")
        if row["author_count"] < len(row["authors"]):
            problems.append(f"{row['task_id']}: author_count < listed authors")
        if not row["announced"].startswith("2026-01"):
            problems.append(f"{row['task_id']}: announced {row['announced']} outside January 2026")

    # Split membership is a pure function of SPLIT_SALT + arxiv_id within each
    # band: rows in the same band must be split by the hash order.
    def split_hash(arxiv_id: str) -> int:
        return int(hashlib.sha256(f"{SPLIT_SALT}:{arxiv_id}".encode()).hexdigest(), 16)

    by_band: dict[str, list[dict]] = {}
    for row in rows:
        by_band.setdefault(row["band"], []).append(row)
    for band, members in by_band.items():
        members = sorted(members, key=lambda r: split_hash(r["arxiv_id"]))
        n_test = sum(1 for m in members if m["split"] == "test")
        if [m["split"] for m in members] != ["test"] * n_test + ["train"] * (len(members) - n_test):
            problems.append(f"band {band}: split is not the hash-order prefix")

    return problems


def check_live(rows: list[dict], sample: int) -> list[str]:
    problems: list[str] = []
    rng = random.Random(VERIFY_SEED)
    chosen = rng.sample(rows, min(sample, len(rows)))

    # Semantic Scholar drift check (cache off — this is the whole point).
    ids = [r["arxiv_id"] for r in chosen]
    live = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        for arxiv_id, paper in zip(chunk, common.s2_batch(chunk, use_cache=False)):
            live[arxiv_id] = None if paper is None else paper.get("citationCount")
    for row in chosen:
        frozen = row["true_citations"]
        current = live.get(row["arxiv_id"])
        if current is None:
            problems.append(f"{row['task_id']}: S2 record vanished")
            continue
        # Counts mostly grow after the snapshot; merges can shrink them a bit.
        if current < 0.7 * frozen - 2:
            problems.append(
                f"{row['task_id']}: live count {current} collapsed vs frozen {frozen}"
            )

    # v1 metadata is immutable: the shipped prompt text must match a re-fetch.
    v1 = common.arxiv_metadata_by_id([f"{r['arxiv_id']}v1" for r in chosen])
    by_id = {e["arxiv_id"]: e for e in v1}
    for row in chosen:
        entry = by_id.get(row["arxiv_id"])
        if entry is None:
            problems.append(f"{row['task_id']}: v1 metadata no longer retrievable")
            continue
        if common.normalise_title(entry["title"]) != common.normalise_title(row["title"]):
            problems.append(f"{row['task_id']}: v1 title changed (should be impossible)")
        if entry["abstract"] != row["abstract"]:
            problems.append(f"{row['task_id']}: v1 abstract mismatch")

    print(f"live-checked {len(chosen)} rows")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=0, help="live re-check this many rows")
    args = parser.parse_args()

    rows = load_rows()
    problems = check_structure(rows)
    print(f"structural: {len(problems)} problems")
    if args.sample:
        problems += check_live(rows, args.sample)

    for p in problems:
        print(f"  PROBLEM: {p}")
    if problems:
        raise SystemExit(1)
    print("verify: all checks passed")


if __name__ == "__main__":
    main()
