# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Acceptance as a function of generated-suffix length, and its use in prediction.

The selector consumed one scalar `tau` per (cell, regime), pooled from
generations of 640 tokens. `results_g98_longu.md` measured what that scalar
hides once generation runs long:

    woff/skip8   3.642 -> 4.106  across u<256 -> 3K-8K   (+12.7%)
    w512/skip4   4.553 -> 4.266                          (-6.3%)

Unwindowed acceptance rises with generation length; a windowed draft's falls,
because past the window the draft loses sight of its own recent output. A
scalar therefore under-rates deep skip and over-rates windows on any
long-output cell, and both errors grow with the generation.

The right scalar to feed a throughput prediction is not the mean of tau. Over
a generation of `G` tokens the quantity that integrates is STEPS PER TOKEN:
emitting the tokens that fall in bucket `b` costs `tokens_b / tau_b` armed
steps, so

    tau_eff(G) = G / sum_b (tokens_b / tau_b)

which is the token-weighted HARMONIC mean. Using the arithmetic mean would
overstate throughput, because it lets the high-acceptance stretches pay for
the low-acceptance ones at face value when in fact the slow stretches consume
disproportionately many steps.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


class TauCurveError(RuntimeError):
    """Raised when a curve cannot support the requested generation length."""


def bucket_bounds(u_edges: Sequence[int]) -> list[tuple[float, float]]:
    """Half-open [lo, hi) ranges implied by ascending inner edges."""
    bounds: list[tuple[float, float]] = []
    lo = 0.0
    for edge in u_edges:
        bounds.append((lo, float(edge)))
        lo = float(edge)
    bounds.append((lo, math.inf))
    return bounds


def tau_effective(
    tau_by_bucket: Mapping[str, float],
    u_edges: Sequence[int],
    gen_tokens: float,
    extrapolate: bool = True,
) -> float:
    """Token-weighted harmonic mean of tau over a generation of `gen_tokens`.

    Args:
        tau_by_bucket: Bucket index (as a string) to measured tau.
        u_edges: Ascending inner bucket boundaries.
        gen_tokens: Generation length to integrate over.
        extrapolate: If a bucket the generation reaches was never measured,
            carry the last measured bucket forward. False raises instead.
            Carrying forward is the conservative reading of the measured
            trends -- unwindowed acceptance was still rising and windowed
            still falling at the longest bucket -- so it under-states both
            effects rather than inventing their continuation.

    Raises:
        TauCurveError: If no bucket is usable, or if the generation reaches an
            unmeasured bucket and `extrapolate` is False.
    """
    if gen_tokens <= 0:
        raise TauCurveError("generation length must be positive")
    steps = 0.0
    emitted = 0.0
    last: float | None = None
    for index, (lo, hi) in enumerate(bucket_bounds(u_edges)):
        if lo >= gen_tokens:
            break
        tau = tau_by_bucket.get(str(index))
        if tau is None or tau <= 0:
            if last is None:
                continue
            if not extrapolate:
                raise TauCurveError(
                    f"bucket {index} is unmeasured and the generation reaches it"
                )
            tau = last
        last = tau
        tokens = min(gen_tokens, hi) - lo
        steps += tokens / tau
        emitted += tokens
    if steps <= 0 or emitted <= 0:
        raise TauCurveError("no usable bucket in the curve")
    return emitted / steps


def scalar_from_profile(
    profile: Mapping[str, object], depth: int = 4
) -> dict[str, float]:
    """Extract per-bucket tau at a given draft depth from a tau_profile."""
    out: dict[str, float] = {}
    buckets = profile.get("buckets") or {}
    for name, entry in buckets.items():  # type: ignore[union-attr]
        armed = int(entry.get("armed_steps", 0))  # type: ignore[union-attr]
        if not armed:
            continue
        positions = entry.get("pos_accepted") or []  # type: ignore[union-attr]
        out[str(name)] = 1.0 + sum(positions[:depth]) / armed
    return out


def pooled(tau_by_bucket: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """What the OLD scalar was: armed-step-weighted arithmetic mean.

    Kept so the two estimators can be compared on identical inputs rather
    than across differently-derived numbers.
    """
    total = sum(weights.get(b, 0.0) for b in tau_by_bucket)
    if total <= 0:
        raise TauCurveError("no weight to pool over")
    return sum(tau_by_bucket[b] * weights.get(b, 0.0) for b in tau_by_bucket) / total
