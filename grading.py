"""
Deterministic reward arithmetic for CiteCast. Pure math, no I/O, no models —
imported by the environment, the sweep, and the golden tests alike.
"""

from __future__ import annotations

import math

from constants import ALPHA, CITATION_BANDS


def compute_reward(predicted: int, actual: int, alpha: float = ALPHA) -> float:
    """Smooth log-ratio reward in [0, 1].

    r = 1/cosh(alpha * |ln((predicted+1)/(actual+1))|)

    Symmetric in log space (predicting 2x and 0.5x the truth score the same),
    1.0 at an exact hit, and smoothly decaying — never a hard zero, so the
    gradient survives even far misses. The +1 shift makes zero-citation
    predictions well-defined.
    """
    err = abs(math.log((predicted + 1) / (actual + 1)))
    return round(1.0 / math.cosh(alpha * err), 4)


def band_of(count: int) -> str:
    """The citation band name a count falls in."""
    for name, lo, hi in CITATION_BANDS:
        if count >= lo and (hi is None or count <= hi):
            return name
    raise ValueError(f"No band for count {count}")


def candidate_constants(max_dense: int = 200, log_max: int = 2000) -> list[int]:
    """Constants worth scanning: every integer to max_dense, log-spaced beyond.

    The reward is smooth in log space, so beyond the dense range the optimum
    cannot hide between log-spaced probes.
    """
    dense = list(range(0, max_dense + 1))
    sparse: list[int] = []
    value = float(max_dense)
    while value < log_max:
        value *= 1.15
        sparse.append(int(round(value)))
    return dense + sorted(set(sparse))


def mean_reward_for_constant(constant: int, labels: list[int], alpha: float = ALPHA) -> float:
    return sum(compute_reward(constant, t, alpha) for t in labels) / len(labels)


def best_constant(labels: list[int], alpha: float = ALPHA) -> tuple[int, float]:
    """The single constant prediction with the highest mean reward.

    This is the farmability floor: a policy that never reads anything can
    always play this constant, so the shipped mix must keep it low (the gate
    lives in constants.FARM_GATE and is asserted offline in golden_tests.py).
    """
    best_c, best_r = 0, -1.0
    for c in candidate_constants():
        r = mean_reward_for_constant(c, labels, alpha)
        if r > best_r:
            best_c, best_r = c, r
    return best_c, best_r
