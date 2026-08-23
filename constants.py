"""
Path, cutoff, reward, and dataset constants for the CiteCast environment.

Production mounts the task file at `/orwd_data`; locally we fall back to the
package's data/ directory.
"""

import os
from pathlib import Path

# The backdated search service's default corpus fan-out does not include the
# arXiv sub-corpus, so pin it here. This module is imported by citecast.py
# before any tool call, which is what makes the setdefault effective; a
# deployment can still override it explicitly, which is why golden_tests.py
# carries a grader-marked reachability test for cc_arxiv.
os.environ.setdefault("OPENREWARD_WEB_CORPUS", "cc_arxiv,cc_news,cc_web")

if Path("/orwd_data").exists():
    ENV_PATH = Path("/orwd_data")
else:
    ENV_PATH = Path(__file__).parent / "data"

DATA_FILE = ENV_PATH / "citecast_tasks.jsonl"

# Point-in-time boundary for the agent's web tools. Every paper in the cohort
# was announced on arXiv in January 2026, so the agent sees each paper's
# earliest public reception and nothing after.
CUTOFF = "2026-01-31"

# The day the ground-truth citation counts were snapshotted from Semantic
# Scholar. Set from the actual label-fetch run, not from intention; verified
# by golden_tests.py against the shipped rows' snapshot_utc.
SNAPSHOT = "2026-08-23"

SPLITS = ["train", "test"]
# Exact split sizes (not fractions): 2,000 train + 200 test. Per-band test
# counts are allocated by largest remainder in datagen/build.py so every band
# is represented in test and the totals land exactly.
TRAIN_TOTAL = 2000
TEST_TOTAL = 200
# Salt for the within-band deterministic hash split. Changing it reshuffles
# train/test membership, so it is fixed forever.
SPLIT_SALT = "citecast-split-v1"

# Reward: r = 1/cosh(ALPHA * |ln((pred+1)/(true+1))|). At ALPHA=2.0 an exact
# prediction scores 1.0, a 2x error ~0.47, a 3x error ~0.22, a 10x error ~0.02.
# ALPHA is calibrated against the shipped label mix so that the best constant
# prediction stays at or below FARM_GATE (asserted offline in golden_tests.py).
ALPHA = 2.0
MAX_PREDICTION = 1_000_000

# Highest mean reward any constant prediction may achieve on the train split.
FARM_GATE = 0.40
# Reward an honest forecaster earns at exactly 2x error; keeps ALPHA from being
# cranked up to pass the farm gate at the cost of all resolution.
RESOLUTION_FLOOR = 0.45

# Citation bands: explicit integer bounds (house style — greppable, assertable,
# stable across rebuilds), not quantiles. hi=None means unbounded.
CITATION_BANDS = [
    ("0", 0, 0),
    ("1-2", 1, 2),
    ("3-5", 3, 5),
    ("6-10", 6, 10),
    ("11-25", 11, 25),
    ("26-50", 26, 50),
    ("51-100", 51, 100),
    ("101+", 101, None),
]
BAND_NAMES = [name for name, _, _ in CITATION_BANDS]

# Target composition of the shipped dataset, asserted by datagen/verify.py and
# golden_tests.py. Finalized from the full-pool census (2026-08-23 snapshot;
# eligible pool 22,443: 11,535 / 7,233 / 2,343 / 834 / 366 / 90 / 25 / 17 per
# band): the three >=26 bands and 11-25 take nearly their whole pools, and the
# lower bands were grid-searched to minimize the best-constant reward — this
# mix holds it at ~0.384 (constant 0), under the 0.40 gate. A middle-heavy mix
# would instead let a constant ~5-7 clear 0.44. The >=26 bands are 5.6% of the
# mix, so a policy that merely remembers which papers got famous is bounded to
# a small edge.
BAND_COUNTS = {
    "0": 650,
    "1-2": 390,
    "3-5": 285,
    "6-10": 401,
    "11-25": 350,
    "26-50": 85,
    "51-100": 23,
    "101+": 16,
}
TOTAL_TASKS = sum(BAND_COUNTS.values())  # 2200

# Distinct fixed seeds so a rerun of any stage is reproducible and no stage's
# randomness can bleed into another's.
BUILD_SEED = 20260823
VERIFY_SEED = 424243
SWEEP_SEED = 908

# Prompt construction: long author lists are truncated to this many names
# (plus "and N others"); the full count ships as author_count.
MAX_PROMPT_AUTHORS = 10
