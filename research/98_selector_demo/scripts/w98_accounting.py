# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-event accounting and u-binned acceptance estimation (Phase 98 G98-0).

Consumes raw per-step, per-request draft rows and produces closure-checked,
position-resolved, generated-suffix-binned acceptance estimates with
request-level bootstrap intervals. CPU-only; no engine imports.
"""

from __future__ import annotations

import random
from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field


class W98AccountingError(ValueError):
    """A stream violates the frozen accounting contract."""


@dataclass(frozen=True)
class StepRow:
    """One request-row of one target decode event.

    Attributes:
        request_id: Stable request identity; the bootstrap resampling unit.
        suffix_len: Generated-suffix length at step entry (the u axis).
        armed: Whether the row received a draft.
        proposed: Draft tokens proposed; 0 when unarmed.
        accepted: Draft tokens accepted; 0 <= accepted <= proposed.
        clipped: Emissions truncated at a fixed-length boundary.
    """

    request_id: str
    suffix_len: int
    armed: bool
    proposed: int
    accepted: int
    clipped: int = 0


def validate_row(row: StepRow) -> None:
    """Rejects a row that cannot arise from a valid decode event."""
    if row.suffix_len < 0:
        raise W98AccountingError(f"negative suffix_len: {row!r}")
    if row.armed:
        if row.proposed <= 0:
            raise W98AccountingError(f"armed row without proposal: {row!r}")
        if not 0 <= row.accepted <= row.proposed:
            raise W98AccountingError(f"accepted outside [0, proposed]: {row!r}")
    elif row.proposed != 0 or row.accepted != 0:
        raise W98AccountingError(f"unarmed row carries a draft: {row!r}")
    if not 0 <= row.clipped <= row.accepted + 1:
        raise W98AccountingError(f"clipped outside [0, accepted + 1]: {row!r}")


def row_emitted(row: StepRow) -> int:
    """Committed emissions of one row: accepted + 1 target token - clip."""
    return row.accepted + 1 - row.clipped


@dataclass(frozen=True)
class ClosureResult:
    """Aggregate counters for one scored interval."""

    h_steps: int
    d_armed: int
    a_accepted: int
    c_clipped: int
    e_emitted: int

    @property
    def tau_eff(self) -> float:
        """Committed emissions per target step, E / H."""
        if self.h_steps == 0:
            raise W98AccountingError("tau_eff undefined for an empty interval")
        return self.e_emitted / self.h_steps


def close_interval(
    rows: Iterable[StepRow], emitted_total: int | None = None
) -> ClosureResult:
    """Builds aggregate counters and enforces E + C = A + H.

    Args:
        rows: Every step row of the scored interval.
        emitted_total: Independently counted committed emissions; when
            provided it must equal the derived E exactly.

    Returns:
        The interval's ClosureResult.

    Raises:
        W98AccountingError: On an invalid row or a closure mismatch.
    """
    h = d = a = c = e = 0
    for row in rows:
        validate_row(row)
        h += 1
        d += int(row.armed)
        a += row.accepted
        c += row.clipped
        e += row_emitted(row)
    if e + c != a + h:
        raise W98AccountingError(
            f"closure violated: E({e}) + C({c}) != A({a}) + H({h})"
        )
    if emitted_total is not None and emitted_total != e:
        raise W98AccountingError(f"frontend emissions {emitted_total} != derived E {e}")
    return ClosureResult(h, d, a, c, e)


@dataclass
class BucketCounters:
    """Position-resolved counters for one generated-suffix bucket.

    ``pos_accepted[i]`` counts scored armed steps whose accepted count
    exceeds position ``i`` (0-indexed). Clipped armed steps are excluded
    from the counters and reported separately; unarmed steps contribute
    to H only.
    """

    kmax: int
    armed_steps: int = 0
    unarmed_steps: int = 0
    clipped_excluded: int = 0
    pos_accepted: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.pos_accepted:
            self.pos_accepted = [0] * self.kmax


def bucket_index(u_edges: Sequence[int], suffix_len: int) -> int:
    """Maps a suffix length to its bucket: edges are right-open bounds."""
    return bisect_right(u_edges, suffix_len)


def bin_rows(
    rows: Iterable[StepRow], kmax: int, u_edges: Sequence[int]
) -> dict[int, BucketCounters]:
    """Bins rows into per-u-bucket position counters.

    Enforces the uniform-KMAX contract: every armed row must propose
    exactly ``kmax`` tokens, which structurally prevents mixed-K pooling.

    Args:
        rows: Step rows of one (composition, regime) stream.
        kmax: The stream's registered proposal depth.
        u_edges: Ascending inner bucket boundaries (right-open).

    Returns:
        Mapping from bucket index to its counters.
    """
    if kmax <= 0:
        raise W98AccountingError("kmax must be positive")
    if list(u_edges) != sorted(set(u_edges)):
        raise W98AccountingError(f"u_edges not strictly ascending: {u_edges}")
    buckets: dict[int, BucketCounters] = {}
    for row in rows:
        validate_row(row)
        b = buckets.setdefault(
            bucket_index(u_edges, row.suffix_len), BucketCounters(kmax)
        )
        if not row.armed:
            b.unarmed_steps += 1
            continue
        if row.proposed != kmax:
            raise W98AccountingError(
                f"mixed-K stream: proposed {row.proposed} != kmax {kmax}"
            )
        if row.clipped:
            b.clipped_excluded += 1
            continue
        b.armed_steps += 1
        for i in range(row.accepted):
            b.pos_accepted[i] += 1
    return buckets


def tau_at_depth(counters: BucketCounters, k: int) -> float:
    """Unbiased tau(k) = 1 + sum of per-position acceptance, k <= kmax."""
    if not 1 <= k <= counters.kmax:
        raise W98AccountingError(f"depth {k} outside [1, {counters.kmax}]")
    if counters.armed_steps == 0:
        raise W98AccountingError("no scored armed steps in bucket")
    total = sum(counters.pos_accepted[i] for i in range(k))
    return 1.0 + total / counters.armed_steps


def _per_request_partials(
    rows: Sequence[StepRow], kmax: int, u_edges: Sequence[int]
) -> dict[str, dict[int, BucketCounters]]:
    by_request: dict[str, list[StepRow]] = {}
    for row in rows:
        by_request.setdefault(row.request_id, []).append(row)
    return {
        rid: bin_rows(request_rows, kmax, u_edges)
        for rid, request_rows in by_request.items()
    }


def _merge(parts: Iterable[dict[int, BucketCounters]], kmax: int):
    merged: dict[int, BucketCounters] = {}
    for part in parts:
        for idx, c in part.items():
            m = merged.setdefault(idx, BucketCounters(kmax))
            m.armed_steps += c.armed_steps
            m.unarmed_steps += c.unarmed_steps
            m.clipped_excluded += c.clipped_excluded
            for i in range(kmax):
                m.pos_accepted[i] += c.pos_accepted[i]
    return merged


@dataclass(frozen=True)
class TauInterval:
    """Point estimate and percentile bootstrap interval for one bucket."""

    point: float
    lo: float
    hi: float
    undefined_draws: int


def bootstrap_tau(
    rows: Sequence[StepRow],
    kmax: int,
    u_edges: Sequence[int],
    k: int,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[int, TauInterval]:
    """Request-level percentile bootstrap of tau(k) per u bucket."""
    partials = _per_request_partials(rows, kmax, u_edges)
    ids = sorted(partials)
    if not ids:
        raise W98AccountingError("empty stream")
    full = _merge(partials.values(), kmax)
    points = {idx: tau_at_depth(c, k) for idx, c in full.items() if c.armed_steps}
    rng = random.Random(seed)
    draws: dict[int, list[float]] = {idx: [] for idx in points}
    undefined = dict.fromkeys(points, 0)
    for _ in range(n_boot):
        sample = [partials[rng.choice(ids)] for _ in ids]
        merged = _merge(sample, kmax)
        for idx in points:
            c = merged.get(idx)
            if c is None or c.armed_steps == 0:
                undefined[idx] += 1
            else:
                draws[idx].append(tau_at_depth(c, k))
    out = {}
    for idx, values in draws.items():
        if not values:
            raise W98AccountingError(f"bucket {idx} empty in every draw")
        values.sort()
        lo = values[int(0.025 * (len(values) - 1))]
        hi = values[int(0.975 * (len(values) - 1))]
        out[idx] = TauInterval(points[idx], lo, hi, undefined[idx])
    return out


def bootstrap_paired_delta(
    rows_a: Sequence[StepRow],
    rows_b: Sequence[StepRow],
    kmax: int,
    u_edges: Sequence[int],
    k: int,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[int, TauInterval]:
    """Paired-by-request bootstrap of tau_A(k) - tau_B(k) per bucket.

    Both streams must cover exactly the same request identities (the
    frozen prompt ids), so content variance cancels in the pairing.
    """
    parts_a = _per_request_partials(rows_a, kmax, u_edges)
    parts_b = _per_request_partials(rows_b, kmax, u_edges)
    if set(parts_a) != set(parts_b):
        raise W98AccountingError("paired comparison requires identical request-id sets")
    ids = sorted(parts_a)
    full_a = _merge(parts_a.values(), kmax)
    full_b = _merge(parts_b.values(), kmax)
    shared = {
        idx
        for idx in set(full_a) & set(full_b)
        if full_a[idx].armed_steps and full_b[idx].armed_steps
    }
    points = {
        idx: tau_at_depth(full_a[idx], k) - tau_at_depth(full_b[idx], k)
        for idx in shared
    }
    rng = random.Random(seed)
    draws: dict[int, list[float]] = {idx: [] for idx in points}
    undefined = dict.fromkeys(points, 0)
    for _ in range(n_boot):
        chosen = [rng.choice(ids) for _ in ids]
        merged_a = _merge([parts_a[rid] for rid in chosen], kmax)
        merged_b = _merge([parts_b[rid] for rid in chosen], kmax)
        for idx in points:
            ca, cb = merged_a.get(idx), merged_b.get(idx)
            if not ca or not cb or not ca.armed_steps or not cb.armed_steps:
                undefined[idx] += 1
            else:
                draws[idx].append(tau_at_depth(ca, k) - tau_at_depth(cb, k))
    out = {}
    for idx, values in draws.items():
        if not values:
            raise W98AccountingError(f"bucket {idx} empty in every draw")
        values.sort()
        lo = values[int(0.025 * (len(values) - 1))]
        hi = values[int(0.975 * (len(values) - 1))]
        out[idx] = TauInterval(points[idx], lo, hi, undefined[idx])
    return out
