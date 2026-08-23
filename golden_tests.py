"""
Golden tests for CiteCast.

Offline tests assert the shipped dataset's invariants and the reward
arithmetic; they need no network and no keys. Tests marked `grader` exercise
the live backdated web service (they need OPENREWARD_API_KEY and spend search
quota) — deselect with `-m "not grader"`.

    pytest golden_tests.py -v -m "not grader"
    OPENREWARD_API_KEY=... pytest golden_tests.py -v -m grader
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import urllib.request
from collections import Counter

import pytest

from constants import (
    ALPHA,
    BAND_COUNTS,
    CITATION_BANDS,
    CUTOFF,
    DATA_FILE,
    FARM_GATE,
    RESOLUTION_FLOOR,
    SNAPSHOT,
    TEST_TOTAL,
    TOTAL_TASKS,
    TRAIN_TOTAL,
)
from grading import band_of, best_constant, compute_reward
from prompts import render_prompt


def load_rows() -> list[dict]:
    with open(DATA_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]


ROWS = load_rows()
TRAIN = [r for r in ROWS if r["split"] == "train"]
TEST = [r for r in ROWS if r["split"] == "test"]


# --------------------------------------------------------------------------
# Dataset invariants (offline)
# --------------------------------------------------------------------------


def test_task_count():
    assert len(ROWS) == TOTAL_TASKS
    assert len(TRAIN) == TRAIN_TOTAL
    assert len(TEST) == TEST_TOTAL


def test_band_composition():
    counts = Counter(r["band"] for r in ROWS)
    assert dict(counts) == BAND_COUNTS


def test_every_band_in_test_split():
    assert {r["band"] for r in TEST} == set(BAND_COUNTS)


def test_ids_unique():
    for field in ("task_id", "arxiv_id", "s2_paper_id"):
        values = [r[field] for r in ROWS]
        assert len(set(values)) == len(values), f"duplicate {field}"


def test_titles_unique():
    keys = [re.sub(r"\W+", " ", r["title"]).strip().casefold() for r in ROWS]
    assert len(set(keys)) == len(keys)


def test_counts_within_bands():
    for row in ROWS:
        assert band_of(row["true_citations"]) == row["band"]


def test_single_snapshot():
    assert {r["snapshot_utc"][:10] for r in ROWS} == {SNAPSHOT}


def test_task_id_encodes_no_band():
    # Ids are cc_<arxiv_id> and nothing else: a band marker in the id would
    # leak the label into every trajectory.
    for row in ROWS:
        assert re.fullmatch(r"cc_2601\.\d{4,5}", row["task_id"]), row["task_id"]
        assert row["task_id"] == f"cc_{row['arxiv_id']}"


def test_cohort_is_january_2026():
    for row in ROWS:
        assert row["arxiv_id"].startswith("2601.")
        assert row["announced"].startswith("2026-01")


def test_no_forbidden_metadata_fields():
    # These arXiv fields mutate without a version bump and can carry
    # post-cutoff information ("accepted at ICML 2026", the journal DOI).
    forbidden = {"comment", "comments", "journal_ref", "doi", "categories"}
    for row in ROWS:
        assert not forbidden & set(row.keys())


def test_prompt_contains_no_label():
    # For the most-cited rows (where a leak would pay most), the true count
    # must not appear in the prompt as a citation figure, and no band name may
    # appear at all.
    top = sorted(ROWS, key=lambda r: r["true_citations"], reverse=True)[:20]
    band_names = [name for name, _, _ in CITATION_BANDS if "-" in name or name.endswith("+")]
    for row in top:
        prompt = render_prompt(row)
        true = row["true_citations"]
        assert not re.search(rf"\b{true}\s+citations?\b", prompt)
        # Band names as raw substrings occur accidentally in dates
        # ("2026-01-28" contains "1-2"), so check citation-adjacent context
        # and the vocabulary that would disclose the stratification.
        assert "band" not in prompt.lower()
        for band in band_names:
            assert not re.search(rf"{re.escape(band)}\s+citations?\b", prompt)


def test_prompt_carries_paper_card():
    row = ROWS[0]
    prompt = render_prompt(row)
    for needle in (row["arxiv_id"], row["title"], row["announced"],
                   row["primary_category"], SNAPSHOT, CUTOFF):
        assert str(needle) in prompt


# --------------------------------------------------------------------------
# Reward arithmetic (offline)
# --------------------------------------------------------------------------


def test_reward_exact_is_full():
    for value in (0, 1, 7, 250):
        assert compute_reward(value, value) == 1.0


def test_reward_symmetric_in_log_space():
    # (4+1)/(9+1) and (9+1)/(4+1) are reciprocal ratios: same reward.
    assert compute_reward(4, 9) == compute_reward(9, 4)


def test_reward_monotone_in_error():
    truth = 10
    rewards = [compute_reward(p, truth) for p in (10, 15, 25, 50, 120, 500)]
    assert rewards == sorted(rewards, reverse=True)
    assert all(0.0 <= r <= 1.0 for r in rewards)


def test_reward_documented_values():
    # The README's worked examples, pinned.
    assert compute_reward(9, 4) == pytest.approx(0.4706, abs=1e-4)   # 2x error
    assert compute_reward(99, 9) == pytest.approx(0.02, abs=1e-2)    # 10x error
    assert compute_reward(5, 0) == pytest.approx(0.0555, abs=1e-4)
    assert compute_reward(0, 1) == pytest.approx(0.4706, abs=1e-4)


def test_reward_rounding():
    assert compute_reward(3, 7) == round(compute_reward(3, 7), 4)


def test_resolution_floor():
    assert compute_reward(9, 4, ALPHA) >= RESOLUTION_FLOOR


def test_farmability_floor():
    """The central farmability gate: no constant prediction may clear FARM_GATE.

    A policy that reads nothing can always play the best constant, so this
    number is the floor an actual forecaster has to beat — and it must be low.
    """
    labels = [r["true_citations"] for r in TRAIN]
    constant, reward = best_constant(labels)
    assert reward <= FARM_GATE, f"constant {constant} farms {reward:.4f}"


# --------------------------------------------------------------------------
# Environment contract (offline)
# --------------------------------------------------------------------------


def test_web_as_of_pinned():
    from citecast import CiteCast, CiteCastBackSearch
    from openreward.toolsets import BackSearchToolset

    assert CiteCast.web_as_of == CUTOFF
    assert CiteCast.toolsets == [CiteCastBackSearch]
    assert issubclass(CiteCastBackSearch, BackSearchToolset)


def test_agent_cannot_pass_as_of():
    # Pins the SDK contract: if a future SDK exposes as_of/corpus as tool
    # params, the point-in-time guarantee is gone and this env must not ship.
    from openreward.toolsets._web_common import WebFetchParams, WebSearchParams

    for model in (WebSearchParams, WebFetchParams):
        assert "as_of" not in model.model_fields
        assert "corpus" not in model.model_fields


def test_corpus_pinned_to_arxiv():
    import constants  # noqa: F401  (the setdefault runs at import)

    assert "cc_arxiv" in os.environ.get("OPENREWARD_WEB_CORPUS", "")


def test_list_tasks_returns_only_task_id():
    from citecast import CiteCast

    for split, want in (("train", TRAIN_TOTAL), ("test", TEST_TOTAL)):
        specs = CiteCast.list_tasks(split)
        assert len(specs) == want
        assert all(set(spec.keys()) == {"task_id"} for spec in specs)


def test_splits_typed():
    from citecast import CiteCast

    splits = CiteCast.list_splits()
    assert [(s.name, s.type) for s in splits] == [("train", "train"), ("test", "test")]


# --------------------------------------------------------------------------
# Live backdated-web checks (grader-marked: spend search quota)
# --------------------------------------------------------------------------


def _api_key() -> str:
    key = os.environ.get("OPENREWARD_API_KEY", "").strip()
    if not key:
        pytest.skip("OPENREWARD_API_KEY not set")
    return key


@pytest.mark.grader
def test_search_reaches_cc_arxiv():
    """The corpus pin works end to end: a cohort paper's exact title, searched
    through the same engine call the toolset makes, must surface its abs page."""
    from openreward.tools.web import run_search
    from openreward.web_service import WebServiceConfig

    _api_key()
    row = max(ROWS, key=lambda r: r["true_citations"])
    result = asyncio.run(
        run_search(query=row["title"][:200], as_of=CUTOFF,
                   config=WebServiceConfig.from_env(), include_snippets=True)
    )
    text = result.text or ""
    assert f"arxiv.org/abs/{row['arxiv_id']}" in text, text[:500]


@pytest.mark.grader
def test_fetch_is_pre_cutoff():
    """The served crawl of a cohort abs page must predate the cutoff."""
    key = _api_key()
    row = max(ROWS, key=lambda r: r["true_citations"])
    request = urllib.request.Request(
        "https://search.openreward.ai/fetch",
        data=json.dumps({"url": f"https://arxiv.org/abs/{row['arxiv_id']}",
                         "as_of": CUTOFF}).encode(),
        headers={"x-api-key": key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read())
    assert payload.get("crawl_date", "9999")[:10] <= CUTOFF
    assert payload.get("text")


@pytest.mark.grader
def test_fetch_output_fits_harness_cap():
    """A full-paper fetch must fit under the 32,768-byte harness tool-output cap.

    The SDK mirrors the fetched text into metadata["content"]; harnesses that
    cap tool outputs drop the WHOLE output when metadata alone exceeds the cap,
    so the model sees a placeholder instead of the paper. CiteCastBackSearch
    strips the mirror and caps text at FETCH_MAX_CHARS — this test replays the
    fetch that originally surfaced the bug (a ~55k-char capture)."""
    import json as _json

    from openreward.toolsets._web_common import WebFetchParams

    from citecast import CiteCastBackSearch
    from constants import FETCH_MAX_CHARS

    _api_key()
    toolset = CiteCastBackSearch(None, as_of=CUTOFF)
    params = WebFetchParams(
        url="https://arxiv.org/abs/2601.09028", prompt="Extract the author names."
    )
    out = asyncio.run(toolset.web_fetch(params))
    text = out.blocks[0].text
    assert "OpenDecoder" in text
    assert len(text) <= FETCH_MAX_CHARS + 500  # envelope headroom
    assert not (out.metadata and "content" in out.metadata)
    wire = _json.dumps({"blocks": [{"text": text}], "metadata": out.metadata})
    assert len(wire.encode()) < 32_768


@pytest.mark.grader
def test_submit_flow():
    from citecast import CiteCast

    key = _api_key()
    row = ROWS[0]
    env = CiteCast({"task_id": row["task_id"]}, secrets={"api_key": key})

    from citecast import SubmitPredictionInput

    params = SubmitPredictionInput(predicted_citations=3, reasoning="test")
    out = asyncio.run(env.submit_prediction(params))
    assert out.finished is True
    assert out.reward == compute_reward(3, row["true_citations"])
    assert str(row["true_citations"]) not in out.blocks[0].text or row["true_citations"] == 3

    again = asyncio.run(env.submit_prediction(params))
    assert again.metadata["error"] == "already_submitted"
    assert again.reward == 0.0
