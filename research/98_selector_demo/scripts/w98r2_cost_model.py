# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The W98-R2 corrected cost model.

Round 1 fitted ``D = keep * (W*kappa_w + KV*kappa_kv + c0)`` and under-predicted
quantized composed cells by 4-23%, one-directionally. Two specification faults
were identified and both are fixed here:

1. **`F` sat inside `keep_frac`.** `lm_head` and the embeddings are 41.3% of the
   quantized checkpoint and do NOT scale with skipped layers -- `lm_head` is one
   full-vocab GEMM per chain step whatever the body does. Round 1 multiplied
   them by `keep_frac` anyway. `F` is now a term OUTSIDE `keep_frac`, which
   makes it an irreducible floor on draft cost: measured at 3.66 ms at R1, 12%
   of the unlevered chain and 18% of the best composed configuration.
2. **`W_bytes` was collinear with `[1, q]`.** Round 1's weight column was the
   WHOLE checkpoint, so `16.4e9 - 10.3e9*q` was an affine function of the quant
   indicator and `kappa_w` was not a bytes coefficient at all -- models A and D
   gave identical predictions. The column is now BODY bytes only, with `F`
   carrying the non-body remainder.

A window term `f_win` is also added: windowing costs bookkeeping that is not
proportional to the KV bytes it saves, which Round 1 folded into `c0`.

    D(quant, w, k) = keep * ( W_layer(quant) * kappa_w
                            + KV_bytes(w)    * kappa_kv
                            + c_layer
                            + f_win * [w > 0] )
                   + F

`F` is quant-independent by construction: `lm_head` stays bf16 in both drafts.
That is the standing `ignore=["lm_head"]` decision, aligned with
EfficientRollout and independently forced by the serving stack -- vLLM builds
`lm_head` as `ParallelLMHead`, whose scheme lookup does not match
`targets=["Linear"]`, so a quantized-`lm_head` checkpoint fails to load. That
was verified by building one.

The envelope is unchanged from amendment 1 and is imported from the Round-1
module rather than restated, so the two cannot drift apart. That module is
hash-bound by the v6 authorization; importing it does not modify it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from w98_cost_model import (
    COMPOSED_INFLATION,
    Z_COVERAGE,
    W98CostModelError,
    combined_envelope,
    is_resolvable,
)

__all__ = [
    "COMPOSED_INFLATION",
    "DRAFT_LAYERS",
    "KV_BYTES_PER_TOKEN",
    "KV_SINKS",
    "N_PARAMS",
    "W_LAYER_BYTES",
    "Z_COVERAGE",
    "R2CostModel",
    "R2LeverPoint",
    "W98CostModelError",
    "combined_envelope",
    "design_strength",
    "is_resolvable",
    "keep_frac",
    "kv_bytes",
]

DRAFT_LAYERS = 36
KV_SINKS = 16
# Qwen3-8B GQA: 36 layers x 8 kv heads x 128 head dim x 2 (K and V) x 2 bytes.
KV_BYTES_PER_TOKEN = DRAFT_LAYERS * 8 * 128 * 2 * 2
# BODY bytes only -- the decoder layers. The non-body remainder (lm_head,
# embeddings, sampling) is F, and is deliberately NOT in this column.
W_LAYER_BYTES = {"target-matching": 13.892e9, "w4a16-quantized": 3.581e9}
N_PARAMS = 5


def keep_frac(skip_count: int) -> float:
    """Fraction of draft decoder layers actually executed."""
    if not 0 <= skip_count < DRAFT_LAYERS:
        raise W98CostModelError(f"skip_count out of range: {skip_count}")
    return 1.0 - skip_count / DRAFT_LAYERS


def kv_bytes(window: int | str, context_tokens: float) -> float:
    """Draft KV bytes read per step at a given window and context.

    A window bounds the keys attended to at ``window + sinks``, but only once
    the context exceeds that -- below it the window is inactive and the cost is
    the context itself. Round 1 initially used the prompt length here, which put
    every window above the context, collapsed the window axis, and made the
    design singular.

    Args:
        window: Window size in tokens, or ``"off"``.
        context_tokens: Mean KV length over the measured steps.

    Returns:
        Bytes of draft KV read per step at full layer count.
    """
    if context_tokens <= 0:
        raise W98CostModelError("context_tokens must be positive")
    if window == "off":
        return context_tokens * KV_BYTES_PER_TOKEN
    if not isinstance(window, int) or window <= 0:
        raise W98CostModelError(f"invalid window: {window!r}")
    return min(context_tokens, window + KV_SINKS) * KV_BYTES_PER_TOKEN


@dataclass(frozen=True)
class R2LeverPoint:
    """One measured configuration at one regime.

    Attributes:
        weight_layer_bytes: Body weight bytes for the quant arm.
        kv_bytes: Draft KV bytes per step at full layer count.
        keep_frac: Fraction of decoder layers executed.
        windowed: Whether a window is active, driving the ``f_win`` term.
        d_measured: Measured draft chain cost, seconds.
    """

    weight_layer_bytes: float
    kv_bytes: float
    keep_frac: float
    windowed: bool
    d_measured: float

    def design_row(self) -> tuple[float, float, float, float, float]:
        """The point's row of the design matrix, in parameter order."""
        keep = self.keep_frac
        return (
            keep * self.weight_layer_bytes,
            keep * self.kv_bytes,
            keep,
            keep * (1.0 if self.windowed else 0.0),
            1.0,
        )


def _solve_ls(
    rows: Sequence[tuple[Sequence[float], float]], n: int
) -> tuple[float, ...]:
    """Least squares via the normal equations with partial pivoting.

    Args:
        rows: ``(design_row, observation)`` pairs.
        n: Number of parameters.

    Returns:
        The fitted parameters.

    Raises:
        W98CostModelError: If the design is singular.
    """
    ata = [[0.0] * n for _ in range(n)]
    atb = [0.0] * n
    for x, y in rows:
        for i in range(n):
            atb[i] += x[i] * y
            for j in range(n):
                ata[i][j] += x[i] * x[j]
    matrix = [ata[i] + [atb[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(matrix[r][col]))
        if abs(matrix[pivot][col]) < 1e-12:
            raise W98CostModelError(
                "singular design: the fit set does not vary every axis"
            )
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        for r in range(n):
            if r == col:
                continue
            factor = matrix[r][col] / matrix[col][col]
            for c in range(col, n + 1):
                matrix[r][c] -= factor * matrix[col][c]
    return tuple(matrix[i][n] / matrix[i][i] for i in range(n))


def design_strength(points: Sequence[R2LeverPoint]) -> float:
    """Smallest singular value of the unit-column-normalised design matrix.

    A larger value means the weakest independent direction is better excited,
    which is what decides whether `F` survives estimation rather than being
    traded off against `c_layer`. Rank alone is not the criterion: Round 1's
    8-point set was already formally rank-5.

    The definition is stated here explicitly because the preregistration quotes
    conditioning figures (0.0622 / 0.0955 / 0.1686 / 0.1939) WITHOUT recording
    how they were computed, and this definition does not reproduce them. What
    does reproduce, under this and every other normalisation tried, is the
    ranking and the registered conclusion -- that adding skip16 is worth more
    than all four quant crosses combined. Treat this as a relative diagnostic
    for comparing candidate fit sets, not as a check against those numbers.

    Args:
        points: The fit set.

    Returns:
        The smallest singular value, or 0.0 if the design is rank-deficient.
    """
    import numpy as np

    if not points:
        return 0.0
    matrix = np.array([p.design_row() for p in points], dtype=float)
    norms = np.linalg.norm(matrix, axis=0)
    if not np.all(norms > 0):
        return 0.0
    return float(np.linalg.svd(matrix / norms, compute_uv=False)[-1])


@dataclass(frozen=True)
class R2CostModel:
    """The five-parameter corrected model, fitted per regime.

    Attributes:
        kappa_w: Seconds per body weight byte.
        kappa_kv: Seconds per KV byte.
        c_layer: Per-layer fixed cost.
        f_win: Per-layer window bookkeeping cost, active when windowed.
        f_fixed: ``F`` -- the cost outside ``keep_frac``, an irreducible floor.
        log_envelope: Worst absolute log residual over the fit set.
    """

    kappa_w: float
    kappa_kv: float
    c_layer: float
    f_win: float
    f_fixed: float
    log_envelope: float

    @classmethod
    def fit(cls, points: Sequence[R2LeverPoint]) -> R2CostModel:
        """Fit from the registered fit set.

        Unlike Round 1 the observation is `d_measured` itself rather than
        `d_measured / keep_frac`: with `F` outside `keep_frac` the model is no
        longer proportional to `keep`, so dividing through would misweight the
        residuals and, worse, make `F` inestimable.

        Args:
            points: At least ``N_PARAMS`` measured configurations.

        Returns:
            The fitted model.

        Raises:
            W98CostModelError: If there are too few points, a point is invalid,
                or the design is singular.
        """
        if len(points) < N_PARAMS:
            raise W98CostModelError(
                f"the corrected model needs at least {N_PARAMS} points"
            )
        rows = []
        for point in points:
            if point.keep_frac <= 0 or point.d_measured <= 0:
                raise W98CostModelError(f"invalid lever point: {point!r}")
            rows.append((point.design_row(), point.d_measured))
        kappa_w, kappa_kv, c_layer, f_win, f_fixed = _solve_ls(rows, N_PARAMS)
        envelope = 0.0
        for point in points:
            predicted = _evaluate(kappa_w, kappa_kv, c_layer, f_win, f_fixed, point)
            if predicted <= 0:
                raise W98CostModelError("non-positive fitted cost")
            envelope = max(envelope, abs(math.log(predicted / point.d_measured)))
        return cls(kappa_w, kappa_kv, c_layer, f_win, f_fixed, envelope)

    def predict(self, point: R2LeverPoint) -> float:
        """Predicted draft chain cost for a configuration, seconds."""
        if point.keep_frac <= 0:
            raise W98CostModelError("keep_frac must be positive")
        return _evaluate(
            self.kappa_w,
            self.kappa_kv,
            self.c_layer,
            self.f_win,
            self.f_fixed,
            point,
        )

    def predict_interval(
        self, point: R2LeverPoint, sigma_repro: float
    ) -> tuple[float, float]:
        """Symmetric log interval under amendment 1.

        ``sigma_repro`` is required and has no default, for the same reason as
        in Round 1: an interval that omits the measurement term is the defect
        the amendment exists to fix.

        Args:
            point: The configuration to predict.
            sigma_repro: Measured log-scale boot-to-boot stdev at this regime.

        Returns:
            ``(lo, hi)`` of the symmetric log interval.
        """
        centre = self.predict(point)
        half = combined_envelope(self.log_envelope, sigma_repro)
        return centre * math.exp(-half), centre * math.exp(half)

    def floor_fraction(self, point: R2LeverPoint) -> float:
        """Share of predicted cost that no lever can remove.

        `F` is outside `keep_frac`, so it bounds what the selector can achieve.
        Reporting it is how the registered "irreducible floor" claim is checked
        rather than asserted.
        """
        predicted = self.predict(point)
        if predicted <= 0:
            raise W98CostModelError("non-positive predicted cost")
        return self.f_fixed / predicted


def _evaluate(
    kappa_w: float,
    kappa_kv: float,
    c_layer: float,
    f_win: float,
    f_fixed: float,
    point: R2LeverPoint,
) -> float:
    row = point.design_row()
    return (
        row[0] * kappa_w
        + row[1] * kappa_kv
        + row[2] * c_layer
        + row[3] * f_win
        + row[4] * f_fixed
    )
