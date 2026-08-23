"""
CiteCast dataset builder. Staged, parts-based, resumable:

    python build.py --harvest    # January 2026 cohort from the arXiv API
    python build.py --labels     # Semantic Scholar citation snapshot (one day!)
    python build.py --census     # band pools + farmability math for BAND_COUNTS
    python build.py --enrich     # v1 metadata + OpenAlex for sampled candidates
    python build.py --assemble   # exclusions, sampling, split, shuffle, ship

Each stage writes datagen/parts/<stage>.jsonl and is idempotent (sources are
disk-cached). The labels stage must complete within one calendar day — the
snapshot date it stamps is the day the live fetches ran, and rerunning later
against a stale cache would mislabel it (delete datagen/cache to re-snapshot).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import (  # noqa: E402
    BAND_COUNTS,
    BAND_NAMES,
    BUILD_SEED,
    CITATION_BANDS,
    FARM_GATE,
    SNAPSHOT,
    SPLIT_SALT,
    TEST_TOTAL,
    TOTAL_TASKS,
    TRAIN_TOTAL,
)
from grading import band_of, best_constant  # noqa: E402

import common  # noqa: E402

PARTS_DIR = Path(__file__).parent / "parts"
DATA_FILE = Path(__file__).parent.parent / "data" / "citecast_tasks.jsonl"

COHORT_PART = PARTS_DIR / "cohort.jsonl"
LABELS_PART = PARTS_DIR / "labels.jsonl"
V1_PART = PARTS_DIR / "v1_meta.jsonl"
OPENALEX_PART = PARTS_DIR / "openalex.json"

JANUARY_DAYS = [f"2026-01-{d:02d}" for d in range(1, 32)]

# Candidate slack over the per-band target, to absorb enrichment-time drops
# (missing v1 metadata, dedup casualties). Tight bands take the whole pool.
ENRICH_SLACK = 1.3

# Every shipped row carries every one of these keys, in this order — no .get()
# presence tests downstream.
_ROW_KEYS = (
    "task_id",
    "split",
    "band",
    "arxiv_id",
    "announced",
    "title",
    "abstract",
    "authors",
    "author_count",
    "primary_category",
    "true_citations",
    "s2_paper_id",
    "snapshot_utc",
    "provenance",
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)
    print(f"wrote {len(rows)} rows -> {path}")


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------
# Stage 1: harvest
# --------------------------------------------------------------------------


def harvest() -> None:
    """Every paper submitted in January 2026, filtered to 2601.* ids.

    The submittedDate window also catches papers announced in February (2602.*
    ids) whose abs pages did not exist before the cutoff; those are dropped.
    The metadata kept here is the LATEST version's — used for withdrawal
    detection and dedup only, never for prompts (prompts use v1, see --enrich).
    """
    seen: dict[str, dict] = {}
    for day in JANUARY_DAYS:
        entries = common.arxiv_day_slice(day)
        kept = 0
        for e in entries:
            if not e["arxiv_id"].startswith("2601."):
                continue
            if e["arxiv_id"] not in seen:
                seen[e["arxiv_id"]] = {
                    "arxiv_id": e["arxiv_id"],
                    "latest_version": e["version"],
                    "latest_title": e["title"],
                    "latest_abstract": e["abstract"],
                    "authors": e["authors"],
                    "primary_category": e["primary_category"],
                    "published": e["published"],
                }
                kept += 1
        print(f"{day}: {len(entries)} submitted, {kept} new 2601.* papers (total {len(seen)})")
    _write_jsonl(COHORT_PART, sorted(seen.values(), key=lambda r: r["arxiv_id"]))


# --------------------------------------------------------------------------
# Stage 2: labels
# --------------------------------------------------------------------------


def labels() -> None:
    """Semantic Scholar citation counts for the whole cohort, one snapshot day."""
    cohort = _read_jsonl(COHORT_PART)
    ids = [r["arxiv_id"] for r in cohort]
    snapshot_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows: list[dict] = []
    missing = 0
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        results = common.s2_batch(chunk)
        for arxiv_id, paper in zip(chunk, results):
            if paper is None:
                missing += 1
                rows.append(
                    {"arxiv_id": arxiv_id, "citations": None, "s2_paper_id": None,
                     "s2_publication_date": None, "snapshot_utc": snapshot_utc}
                )
            else:
                rows.append(
                    {
                        "arxiv_id": arxiv_id,
                        "citations": paper.get("citationCount"),
                        "s2_paper_id": paper.get("paperId"),
                        "s2_publication_date": paper.get("publicationDate"),
                        "snapshot_utc": snapshot_utc,
                    }
                )
        print(f"labels: {len(rows)}/{len(ids)} ({missing} missing so far)")
    _write_jsonl(LABELS_PART, rows)
    print(f"snapshot_utc: {snapshot_utc} — constants.SNAPSHOT must match its date part")


# --------------------------------------------------------------------------
# Pool assembly shared by census / enrich / assemble
# --------------------------------------------------------------------------


def _eligible_pool() -> dict[str, list[dict]]:
    """Cohort joined with labels, exclusions applied, grouped by band.

    Exclusions, in order: no S2 record; citation count null; withdrawn latest
    version; duplicate s2_paper_id (two arXiv ids, one underlying work);
    duplicate normalised title.
    """
    cohort = {r["arxiv_id"]: r for r in _read_jsonl(COHORT_PART)}
    label_rows = _read_jsonl(LABELS_PART)

    dropped = Counter()
    seen_s2: set[str] = set()
    seen_titles: set[str] = set()
    pool: dict[str, list[dict]] = defaultdict(list)

    for lab in label_rows:
        paper = cohort.get(lab["arxiv_id"])
        if paper is None:
            dropped["not_in_cohort"] += 1
            continue
        if lab["s2_paper_id"] is None or lab["citations"] is None:
            dropped["s2_missing"] += 1
            continue
        if common.looks_withdrawn(
            {"title": paper["latest_title"], "abstract": paper["latest_abstract"]}
        ):
            dropped["withdrawn"] += 1
            continue
        if lab["s2_paper_id"] in seen_s2:
            dropped["dup_s2_paper_id"] += 1
            continue
        title_key = common.normalise_title(paper["latest_title"])
        if title_key in seen_titles:
            dropped["dup_title"] += 1
            continue
        seen_s2.add(lab["s2_paper_id"])
        seen_titles.add(title_key)
        merged = {**paper, **lab}
        pool[band_of(lab["citations"])].append(merged)

    for band in pool:
        pool[band].sort(key=lambda r: r["arxiv_id"])
    print("exclusions:", dict(dropped))
    return pool


# --------------------------------------------------------------------------
# Stage 3: census
# --------------------------------------------------------------------------


def census() -> None:
    """Band pools, BAND_COUNTS feasibility, and the farmability math."""
    pool = _eligible_pool()
    total_eligible = sum(len(v) for v in pool.values())
    print(f"\neligible pool: {total_eligible}")
    print(f"{'band':>8} {'pool':>7} {'target':>7} {'pool*0.8':>9} feasible")
    for name in BAND_NAMES:
        have = len(pool.get(name, []))
        want = BAND_COUNTS[name]
        print(f"{name:>8} {have:>7} {want:>7} {int(have * 0.8):>9} {'YES' if want <= have * 0.8 or want <= have else 'NO':>8}")

    # Farmability of the TARGET mix: simulate the shipped label multiset by
    # sampling each band's target count of true labels from its pool.
    rng = random.Random(BUILD_SEED)
    simulated: list[int] = []
    for name in BAND_NAMES:
        rows = pool.get(name, [])
        take = min(BAND_COUNTS[name], len(rows))
        simulated.extend(r["citations"] for r in rng.sample(rows, take))
    const, reward = best_constant(simulated)
    print(f"\nbest constant on simulated mix: predict {const} -> mean reward {reward:.4f} "
          f"(gate {FARM_GATE}) {'PASS' if reward <= FARM_GATE else 'FAIL'}")


# --------------------------------------------------------------------------
# Stage 4: enrich
# --------------------------------------------------------------------------


def _candidates(pool: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Per-band candidate sample with slack, deterministic under BUILD_SEED."""
    rng = random.Random(BUILD_SEED)
    out: dict[str, list[dict]] = {}
    for name in BAND_NAMES:
        rows = pool.get(name, [])
        want = min(len(rows), int(BAND_COUNTS[name] * ENRICH_SLACK) + 5)
        out[name] = rng.sample(rows, want) if want < len(rows) else list(rows)
    return out


def enrich() -> None:
    """v1 metadata (prompt text) + OpenAlex secondary counts for candidates."""
    pool = _eligible_pool()
    candidates = _candidates(pool)
    flat = [r for name in BAND_NAMES for r in candidates[name]]
    print(f"enriching {len(flat)} candidates "
          f"({ {n: len(candidates[n]) for n in BAND_NAMES} })")

    v1_entries = common.arxiv_metadata_by_id([f"{r['arxiv_id']}v1" for r in flat])
    by_id = {e["arxiv_id"]: e for e in v1_entries}
    rows = []
    missing_v1 = 0
    for r in flat:
        v1 = by_id.get(r["arxiv_id"])
        if v1 is None or not v1["title"] or not v1["abstract"]:
            missing_v1 += 1
            continue
        rows.append(
            {
                "arxiv_id": r["arxiv_id"],
                "v1_title": v1["title"],
                "v1_abstract": v1["abstract"],
                "v1_authors": v1["authors"],
                "v1_primary_category": v1["primary_category"],
                "v1_published": v1["published"],
            }
        )
    print(f"v1 metadata: {len(rows)} ok, {missing_v1} missing/empty")
    _write_jsonl(V1_PART, rows)

    openalex = common.openalex_counts([r["arxiv_id"] for r in rows])
    OPENALEX_PART.write_text(json.dumps(openalex))
    print(f"openalex secondary counts: {len(openalex)} of {len(rows)}")


# --------------------------------------------------------------------------
# Stage 5: assemble
# --------------------------------------------------------------------------


def _test_allocation() -> dict[str, int]:
    """Per-band test counts by largest remainder, summing exactly TEST_TOTAL."""
    frac = TEST_TOTAL / TOTAL_TASKS
    raw = {name: BAND_COUNTS[name] * frac for name in BAND_NAMES}
    alloc = {name: int(raw[name]) for name in BAND_NAMES}
    remainder = TEST_TOTAL - sum(alloc.values())
    by_frac = sorted(BAND_NAMES, key=lambda n: raw[n] - alloc[n], reverse=True)
    for name in by_frac[:remainder]:
        alloc[name] += 1
    assert sum(alloc.values()) == TEST_TOTAL
    return alloc


def _split_hash(arxiv_id: str) -> int:
    return int(hashlib.sha256(f"{SPLIT_SALT}:{arxiv_id}".encode()).hexdigest(), 16)


def assemble() -> None:
    pool = _eligible_pool()
    candidates = _candidates(pool)
    v1 = {r["arxiv_id"]: r for r in _read_jsonl(V1_PART)}
    openalex = json.loads(OPENALEX_PART.read_text()) if OPENALEX_PART.exists() else {}

    # Survivors: candidates with v1 metadata, deduped again on v1 title (the
    # first dedup used latest titles; a revision can change one and not the
    # other).
    seen_v1_titles: set[str] = set()
    survivors: dict[str, list[dict]] = defaultdict(list)
    for name in BAND_NAMES:
        for r in candidates[name]:
            meta = v1.get(r["arxiv_id"])
            if meta is None:
                continue
            key = common.normalise_title(meta["v1_title"])
            if key in seen_v1_titles:
                continue
            seen_v1_titles.add(key)
            survivors[name].append({**r, **meta})

    rng = random.Random(BUILD_SEED)
    test_alloc = _test_allocation()
    rows: list[dict] = []
    for name in BAND_NAMES:
        have = survivors[name]
        want = BAND_COUNTS[name]
        if len(have) < want:
            raise SystemExit(
                f"band {name}: only {len(have)} survivors for target {want} — "
                f"raise ENRICH_SLACK and rerun --enrich"
            )
        chosen = rng.sample(have, want) if want < len(have) else list(have)
        # Deterministic within-band split: the lowest split-hashes go to test.
        chosen.sort(key=lambda r: _split_hash(r["arxiv_id"]))
        n_test = test_alloc[name]
        for i, r in enumerate(chosen):
            split = "test" if i < n_test else "train"
            authors = r["v1_authors"]
            rows.append(
                {
                    "task_id": f"cc_{r['arxiv_id']}",
                    "split": split,
                    "band": name,
                    "arxiv_id": r["arxiv_id"],
                    "announced": (r["v1_published"] or r["published"])[:10],
                    "title": r["v1_title"],
                    "abstract": r["v1_abstract"],
                    "authors": authors,
                    "author_count": len(authors),
                    "primary_category": r["v1_primary_category"] or r["primary_category"],
                    "true_citations": r["citations"],
                    "s2_paper_id": r["s2_paper_id"],
                    "snapshot_utc": r["snapshot_utc"],
                    "provenance": {
                        "s2_publication_date": r["s2_publication_date"],
                        "openalex_citations": openalex.get(r["arxiv_id"]),
                        "harvest": "arxiv_api submittedDate 2026-01",
                        "prompt_metadata": "arxiv_api id_list v1",
                    },
                }
            )

    # Project through the fixed key tuple and shuffle.
    rows = [{k: row[k] for k in _ROW_KEYS} for row in rows]
    rng.shuffle(rows)

    # Final self-checks before shipping.
    assert len(rows) == TOTAL_TASKS
    split_counts = Counter(r["split"] for r in rows)
    assert split_counts["train"] == TRAIN_TOTAL and split_counts["test"] == TEST_TOTAL
    assert all(r["snapshot_utc"][:10] == SNAPSHOT for r in rows), (
        "snapshot_utc does not match constants.SNAPSHOT — update the constant "
        "to the labels run's actual date"
    )
    train_labels = [r["true_citations"] for r in rows if r["split"] == "train"]
    const, reward = best_constant(train_labels)
    print(f"best constant on shipped train labels: {const} -> {reward:.4f} (gate {FARM_GATE})")
    if reward > FARM_GATE:
        raise SystemExit("farmability gate FAILED — adjust BAND_COUNTS")

    _write_jsonl(DATA_FILE, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harvest", action="store_true")
    parser.add_argument("--labels", action="store_true")
    parser.add_argument("--census", action="store_true")
    parser.add_argument("--enrich", action="store_true")
    parser.add_argument("--assemble", action="store_true")
    args = parser.parse_args()
    if args.harvest:
        harvest()
    if args.labels:
        labels()
    if args.census:
        census()
    if args.enrich:
        enrich()
    if args.assemble:
        assemble()
    if not any(vars(args).values()):
        parser.print_help()


if __name__ == "__main__":
    main()
