# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft cost integrated over a generation, not evaluated at one context.

`results_refined_lo.md` located the selector's remaining error on the LO cell
on the COST side. The Round-2 model evaluates

    D = keep * (W*kw + KV(context)*kkv + c_layer + f_win*[w>0]) + F

at ONE context per regime. That is fine when generation is short relative to
the prompt -- the R-regimes generate 640 tokens on prompts of 400 to 14000 --
and wrong when it is not: an LO request starts at ~120 tokens of context and
ends near 17000, so the draft's KV read grows by two orders of magnitude
DURING the request being predicted.

The growth is also exactly where the window lever earns its keep. An
unwindowed draft's KV term rises without bound through the generation; a
windowed one saturates at `window + sinks` and stops paying. Evaluating at a
single context cannot express that difference, which is why the flat model
does not see the 42% win the LO cell actually hands `w512/skip4`.

So cost is integrated over the generation on the same footing as acceptance:

    per-token time = (1/G) * integral_0^G [verify(p+u) + D(p+u)] / tau(u) du

with `tau(u)` from `w98_tau_u`. Both integrands move during a long request and
in opposite directions for a windowed draft -- its cost saturates while its
acceptance decays -- which is the trade the selector has to see.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from w98_tau_u import bucket_bounds

# Sinks accompany every window in this phase's lattice.
WINDOW_SINKS = 16
# Integration granularity in generated tokens. The integrands vary smoothly,
# so a coarse step is enough; 128 keeps a 32K generation to 256 segments.
SEGMENT = 128


class CostCurveError(RuntimeError):
    """Raised when an integrated cost cannot be formed from the inputs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CostCurveError(message)


def kv_positions(window: Any, context: float) -> float:
    """KV positions the DRAFT attends over at a given context length.

    Unwindowed drafts read the whole context; a windowed draft reads at most
    its window plus the sinks, which is the saturation the flat model cannot
    express.
    """
    if window in ("off", 0, None):
        return context
    return min(float(context), float(window) + WINDOW_SINKS)


def draft_cost(
    fit: Mapping[str, float],
    cfg: Mapping[str, Any],
    context: float,
    kv_bytes_per_token: float,
) -> float:
    """Round-2 draft cost evaluated at one context."""
    for key in ("kappa_w", "kappa_kv", "c_layer", "f_win", "F"):
        _require(key in fit, f"fit is missing {key}")
    keep = float(cfg["keep_frac"])
    windowed = 1.0 if cfg["window"] not in ("off", 0, None) else 0.0
    kv = kv_positions(cfg["window"], context) * kv_bytes_per_token
    layer = (
        float(cfg["weight_bytes"]) * fit["kappa_w"]
        + kv * fit["kappa_kv"]
        + fit["c_layer"]
        + fit["f_win"] * windowed
    )
    return keep * layer + fit["F"]


def integrated_per_token(
    fit: Mapping[str, float],
    cfg: Mapping[str, Any],
    tau_by_bucket: Mapping[str, float],
    u_edges: Sequence[int],
    prompt_tokens: float,
    gen_tokens: float,
    verify_at: Any,
    kv_bytes_per_token: float,
    segment: int = SEGMENT,
) -> dict[str, float]:
    """Mean per-token time over a generation, integrating cost and acceptance.

    Args:
        fit: Round-2 parameters for the regime.
        cfg: `keep_frac`, `window`, `weight_bytes` for the arm.
        tau_by_bucket: Measured acceptance curve.
        u_edges: Bucket boundaries for that curve.
        prompt_tokens: Context the request starts from.
        gen_tokens: Tokens the request generates.
        verify_at: Callable mapping context to target verify cost per step.
        kv_bytes_per_token: Bytes of KV per position, per layer stack.
        segment: Integration granularity in generated tokens.

    Returns:
        The mean per-token time, the implied rate, and the mean draft cost --
        the last so a flat-context prediction can be compared against the
        same model rather than against a differently-derived number.
    """
    _require(gen_tokens > 0, "generation length must be positive")
    bounds = bucket_bounds(u_edges)

    def tau_at(u: float) -> float:
        last = None
        for index, (lo, hi) in enumerate(bounds):
            value = tau_by_bucket.get(str(index))
            if value:
                last = value
            if lo <= u < hi:
                return value or last or 0.0
        return last or 0.0

    total_time = 0.0
    total_cost = 0.0
    total_steps = 0.0
    emitted = 0.0
    u = 0.0
    while u < gen_tokens:
        width = min(float(segment), gen_tokens - u)
        centre = u + width / 2.0
        context = prompt_tokens + centre
        tau = tau_at(centre)
        _require(tau > 0, f"no acceptance at u={centre}")
        cost = draft_cost(fit, cfg, context, kv_bytes_per_token)
        steps = width / tau
        total_time += steps * (float(verify_at(context)) + cost)
        total_cost += steps * cost
        total_steps += steps
        emitted += width
        u += width
    return {
        "per_token_s": total_time / emitted,
        "tokens_per_s": emitted / total_time,
        "mean_draft_cost_s": total_cost / total_steps,
        "armed_steps": total_steps,
    }


def flat_per_token(
    fit: Mapping[str, float],
    cfg: Mapping[str, Any],
    tau: float,
    context: float,
    verify_at: Any,
    kv_bytes_per_token: float,
) -> dict[str, float]:
    """The prediction as it was: one context, one scalar acceptance."""
    _require(tau > 0, "acceptance must be positive")
    cost = draft_cost(fit, cfg, context, kv_bytes_per_token)
    per_token = (float(verify_at(context)) + cost) / tau
    return {
        "per_token_s": per_token,
        "tokens_per_s": 1.0 / per_token,
        "mean_draft_cost_s": cost,
    }


def linear_verify(v_fixed: float, v_per_token: float):
    """Target verify cost as a function of context.

    The target attends over the whole context whatever the draft's window is,
    so this term grows for every arm alike and cancels from a ranking only if
    it is held constant -- which is precisely what the flat model did.
    """

    def verify_at(context: float) -> float:
        return v_fixed + v_per_token * float(context)

    return verify_at


# --- the batch drain -------------------------------------------------------
#
# Under natural EOS requests finish at different lengths, so the concurrent
# batch decays through a run and per-token cost rises: a step's batch-SHARED
# work is amortised over fewer requests. `results_refined_lo.md` measured the
# consequence -- a constant-batch prediction over-states throughput by 1.5-2x
# at LO -- and named it as the next term. This is that term.
#
# The decomposition is physical rather than fitted. Per decode step:
#
#   * weights are read ONCE regardless of how many requests are in flight,
#     as are the per-layer launch constant, the window overhead and the
#     floor -> batch-SHARED;
#   * KV is read per sequence -> batch-PROPORTIONAL.
#
# That split is also why speculation pays most at low batch on dense models
# (C1's structural finding): the shared term is divided by fewer requests.


def kv_cost_factor(positions: float, gamma: float, p_ref: float) -> float:
    """Superlinearity multiplier on the KV term. 1.0 when gamma == 1."""
    _require(positions >= 0, "positions must be non-negative")
    _require(p_ref > 0, "reference position count must be positive")
    if positions <= 0:
        return 0.0
    return (positions / p_ref) ** (gamma - 1.0)


def survival(gen_lengths: Sequence[float], u: float) -> int:
    """Requests still generating at position `u`.

    All requests in these cells are admitted together, so they advance in
    lockstep and share one position; the active count is then just the
    survival function of the generation-length distribution.
    """
    return sum(1 for length in gen_lengths if length > u)


def split_cost(
    fit: Mapping[str, float],
    cfg: Mapping[str, Any],
    context: float,
    kv_bytes_per_token: float,
    fit_batch: float,
    gamma: float = 1.0,
    p_ref: float = 14_000.0,
) -> tuple[float, float]:
    """Draft cost per step, split into (batch-shared, per-request) parts.

    The Round-2 fit was taken at one batch per regime, so its KV coefficient
    absorbed that batch: the design column carried per-SEQUENCE bytes while
    the measured cost covered the whole step. Dividing by the fitting batch
    recovers a per-request coefficient, which is what a varying batch needs.
    """
    _require(fit_batch > 0, "fitting batch must be positive")
    keep = float(cfg["keep_frac"])
    windowed = 1.0 if cfg["window"] not in ("off", 0, None) else 0.0
    shared = (
        keep
        * (
            float(cfg["weight_bytes"]) * fit["kappa_w"]
            + fit["c_layer"]
            + fit["f_win"] * windowed
        )
        + fit["F"]
    )
    positions = kv_positions(cfg["window"], context)
    kv = positions * kv_bytes_per_token * kv_cost_factor(positions, gamma, p_ref)
    per_request = keep * kv * fit["kappa_kv"] / fit_batch
    return shared, per_request


def integrated_with_drain(
    fit: Mapping[str, float],
    cfg: Mapping[str, Any],
    tau_by_bucket: Mapping[str, float],
    u_edges: Sequence[int],
    prompt_tokens: float,
    gen_lengths: Sequence[float],
    verify_shared: float,
    verify_per_request: float,
    kv_bytes_per_token: float,
    fit_batch: float,
    segment: int = SEGMENT,
    gamma: float = 1.0,
    p_ref: float = 14_000.0,
    curvature: float = 0.0,
) -> dict[str, float]:
    """Per-token time over a run whose batch decays as requests finish.

    Emitting `du` tokens per active request costs `du / tau(u)` steps and
    yields `B(u) * du` tokens, so a shrinking `B` raises per-token cost even
    though the per-step cost falls.

    `curvature` adds a `c * B^2` term to the step. Everything else here is
    either shared (falling as 1/B per token) or per-request (flat), so the
    predicted speedup is monotone in batch at any coefficients -- and the LO
    sweep measures an arm that peaks at batch 8. Measured directly from the
    traces, the armed step is CONVEX in batch (+0.083 ms/req^2, a quadratic
    fitting to 0.15%) while the parked step is CONCAVE (-0.064), which is the
    ratio that produces an interior maximum. The mechanism is that an armed
    verify carries `B * (K+1)` query positions against the parked path's
    `B * 1`, so it meets the compute-bound regime at a fraction of the batch.
    Zero reproduces the affine model exactly.
    """
    _require(bool(gen_lengths), "no generation lengths")
    bounds = bucket_bounds(u_edges)

    def tau_at(u: float) -> float:
        last = None
        for index, (lo, hi) in enumerate(bounds):
            value = tau_by_bucket.get(str(index))
            if value:
                last = value
            if lo <= u < hi:
                return value or last or 0.0
        return last or 0.0

    horizon = float(max(gen_lengths))
    total_time = 0.0
    total_tokens = 0.0
    u = 0.0
    while u < horizon:
        width = min(float(segment), horizon - u)
        centre = u + width / 2.0
        active = survival(gen_lengths, centre)
        if active:
            context = prompt_tokens + centre
            tau = tau_at(centre)
            _require(tau > 0, f"no acceptance at u={centre}")
            shared, per_request = split_cost(
                fit, cfg, context, kv_bytes_per_token, fit_batch, gamma, p_ref
            )
            step = (
                (verify_shared + shared)
                + active * (verify_per_request + per_request)
                + curvature * active * active
            )
            steps = width / tau
            total_time += steps * step
            total_tokens += active * width
        u += width
    _require(total_tokens > 0, "no tokens emitted")
    return {
        "per_token_s": total_time / total_tokens,
        "tokens_per_s": total_tokens / total_time,
        "mean_active_batch": total_tokens / horizon,
    }


# --- window-aware attention -----------------------------------------------
#
# `results_refined_lo.md` left one residual: `w512/skip4` measured 1.408x
# against a predicted 1.250x, the window saving ~11% more than the model
# credits. The obvious repair -- add an attention term scaling with KV
# positions -- does NOT work, and the reason is worth stating: attention
# bytes and attention flops are both linear in positions, so such a term is
# exactly collinear with the KV-bytes term already present and nothing in a
# linear fit can separate them.
#
# A window can only save MORE than a linear model credits if cost grows
# SUPERLINEARLY with positions. That is physically ordinary -- a 512-position
# working set is resident in cache while a 17000-position one streams from
# HBM, so the per-byte cost is not the same constant in the two regimes.
#
# One parameter expresses it, with the current model as the gamma = 1 case:
#
#     kv_cost(p) = kappa_kv * bytes(p) * (p / p_ref) ** (gamma - 1)
#
# gamma > 1 makes long contexts disproportionately expensive, which is the
# same statement as a short window being disproportionately cheap. It is NOT
# fitted to the single residual above: one point cannot identify a curve, and
# fitting it to that point would be the overfit this phase keeps refusing.
# `probe_w98_window_sweep` measures the shape instead.


def draft_cost_windowed(
    fit: Mapping[str, float],
    cfg: Mapping[str, Any],
    context: float,
    kv_bytes_per_token: float,
    gamma: float = 1.0,
    p_ref: float = 14_000.0,
) -> float:
    """Draft cost with a superlinear KV term.

    `p_ref` anchors the parameterisation at the context the fit was taken at,
    so gamma only redistributes cost between short and long working sets
    instead of rescaling everything.
    """
    keep = float(cfg["keep_frac"])
    windowed = 1.0 if cfg["window"] not in ("off", 0, None) else 0.0
    positions = kv_positions(cfg["window"], context)
    kv = positions * kv_bytes_per_token * kv_cost_factor(positions, gamma, p_ref)
    layer = (
        float(cfg["weight_bytes"]) * fit["kappa_w"]
        + kv * fit["kappa_kv"]
        + fit["c_layer"]
        + fit["f_win"] * windowed
    )
    return keep * layer + fit["F"]
