# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""State-indexed cost surface and factored composition model (Phase 98 G98-0).

Round-1 machinery: affine latency fits with leave-one-out certification,
required-acceptance labels q = P/T, the factored single-lever composition
model, symmetric log-space intervals, and the sound elimination rule.
CPU-only; no engine imports.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


class W98CostModelError(ValueError):
    """Invalid input to the cost model."""


@dataclass(frozen=True)
class AffineFit:
    """Least-squares affine fit y = alpha + beta * Q with LOO errors."""

    alpha: float
    beta: float
    loo_rel_errors: tuple[float, ...]

    @classmethod
    def fit(cls, points: Sequence[tuple[float, float]]) -> AffineFit:
        """Fits from (Q, y) pairs; LOO errors need at least three points."""
        if len(points) < 2:
            raise W98CostModelError("affine fit needs at least two points")
        alpha, beta = _affine_ls(points)
        loo: list[float] = []
        if len(points) >= 3:
            for i, (q, y) in enumerate(points):
                rest = [p for j, p in enumerate(points) if j != i]
                a, b = _affine_ls(rest)
                if y == 0:
                    raise W98CostModelError("zero latency in fit points")
                loo.append(abs((a + b * q - y) / y))
        return cls(alpha, beta, tuple(loo))

    def predict(self, q: float) -> float:
        return self.alpha + self.beta * q

    def certified(self, tol: float = 0.05) -> bool:
        """Affine form is retained only under the LOO tolerance (W14)."""
        return bool(self.loo_rel_errors) and all(e <= tol for e in self.loo_rel_errors)


def _affine_ls(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    n = len(points)
    sx = sum(q for q, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(q * q for q, _ in points)
    sxy = sum(q * y for q, y in points)
    denom = n * sxx - sx * sx
    if denom == 0:
        raise W98CostModelError("degenerate Q values in affine fit")
    beta = (n * sxy - sx * sy) / denom
    alpha = (sy - beta * sx) / n
    return alpha, beta


def required_acceptance(p_fit: AffineFit, t_fit: AffineFit, q_state: float) -> float:
    """q_a(x) = P_a(Q)/T(Q): acceptance needed to break even at state Q."""
    t = t_fit.predict(q_state)
    if t <= 0:
        raise W98CostModelError(f"non-positive T at Q={q_state}")
    return p_fit.predict(q_state) / t


@dataclass(frozen=True)
class LeverPoint:
    """One single-lever draft-cost measurement at a fixed execution state.

    Attributes:
        weight_bytes: Draft weight bytes read per step under the quant path.
        kv_bytes: Draft KV bytes read per step under the window.
        keep_frac: 1 - k/L for skip count k; must be positive.
        d_measured: Measured draft-step cost (any consistent unit).
    """

    weight_bytes: float
    kv_bytes: float
    keep_frac: float
    d_measured: float


@dataclass(frozen=True)
class FactoredCostModel:
    """D(quant, w, k) ~= keep_frac * (Wb*kappa_w + KVb*kappa_kv + c0).

    Fit from single-lever points only; composed configurations are
    predictions carrying the symmetric log-residual envelope.
    """

    kappa_w: float
    kappa_kv: float
    c0: float
    log_envelope: float

    @classmethod
    def fit(cls, points: Sequence[LeverPoint]) -> FactoredCostModel:
        if len(points) < 3:
            raise W98CostModelError("factored fit needs at least three points")
        rows = []
        for p in points:
            if p.keep_frac <= 0 or p.d_measured <= 0:
                raise W98CostModelError(f"invalid lever point: {p!r}")
            rows.append((p.weight_bytes, p.kv_bytes, 1.0, p.d_measured / p.keep_frac))
        kappa_w, kappa_kv, c0 = _solve_ls3(rows)
        envelope = 0.0
        for p in points:
            pred = p.keep_frac * (p.weight_bytes * kappa_w + p.kv_bytes * kappa_kv + c0)
            if pred <= 0:
                raise W98CostModelError("non-positive fitted cost")
            envelope = max(envelope, abs(math.log(pred / p.d_measured)))
        return cls(kappa_w, kappa_kv, c0, envelope)

    def predict(self, weight_bytes: float, kv_bytes: float, keep_frac: float) -> float:
        if keep_frac <= 0:
            raise W98CostModelError("keep_frac must be positive")
        return keep_frac * (
            weight_bytes * self.kappa_w + kv_bytes * self.kappa_kv + self.c0
        )

    def predict_interval(
        self,
        weight_bytes: float,
        kv_bytes: float,
        keep_frac: float,
        inflation: float = 2.0,
    ) -> tuple[float, float]:
        """Symmetric log interval; composed use inflates the envelope."""
        point = self.predict(weight_bytes, kv_bytes, keep_frac)
        half = self.log_envelope * inflation
        return point * math.exp(-half), point * math.exp(half)


def _solve_ls3(
    rows: Sequence[tuple[float, float, float, float]],
) -> tuple[float, float, float]:
    ata = [[0.0] * 3 for _ in range(3)]
    atb = [0.0] * 3
    for x0, x1, x2, y in rows:
        x = (x0, x1, x2)
        for i in range(3):
            atb[i] += x[i] * y
            for j in range(3):
                ata[i][j] += x[i] * x[j]
    m = [ata[i] + [atb[i]] for i in range(3)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise W98CostModelError(
                "singular design: vary each lever axis independently"
            )
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(3):
            if r == col:
                continue
            f = m[r][col] / m[col][col]
            for c in range(col, 4):
                m[r][c] -= f * m[col][c]
    return tuple(m[i][3] / m[i][i] for i in range(3))


def s_max(k_depth: int, q_lo: float) -> float:
    """Most favorable possible speedup: full acceptance over optimistic q."""
    if q_lo <= 0:
        raise W98CostModelError("q_lo must be positive")
    return (k_depth + 1) / q_lo


def eliminate(k_depth: int, q_lo: float, eps_arm: float = 0.015) -> bool:
    """Sound Round-1 elimination: kill only what cannot pay at tau = K+1."""
    return s_max(k_depth, q_lo) < 1.0 + eps_arm


def tau_star(k_depth: int, d_over_t: float, v_over_t: float, c_over_t: float) -> float:
    """Break-even acceptance: K*D/T + V + C/T (W12 identity denominator)."""
    return k_depth * d_over_t + v_over_t + c_over_t


def surviving_counts(
    d_over_t_by_count: Mapping[int, float],
    k_depth: int,
    v_over_t: float,
    c_over_t: float,
    tau_bar: float,
) -> list[int]:
    """Skip-count ladder: keeps count k iff tau*_k <= tau_bar.

    ``tau_bar`` must be a sound upper bound on achievable acceptance
    (K+1, or the measured no-skip tau, which skipping never exceeds).
    """
    if tau_bar <= 1.0:
        raise W98CostModelError("tau_bar must exceed 1")
    return sorted(
        k
        for k, d in d_over_t_by_count.items()
        if tau_star(k_depth, d, v_over_t, c_over_t) <= tau_bar
    )
