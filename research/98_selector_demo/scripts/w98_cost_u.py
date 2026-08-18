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
