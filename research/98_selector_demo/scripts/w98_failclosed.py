# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fail-closed arming rule: do not speculate on a margin you cannot see.

D3 scored the two-round selector at 95.1% of omniscient, and essentially all
of the gap is one cell. At R4 (batch 8, 8k context, summarization) every lever
family loses to no speculation at all, and the selector armed anyway: 522.3
against OFF's 684.9 decode tok/s, a 24% regression on that regime. Its other
five picks cost under 1% combined.

So the defect is not a ranking error. The selector ranked the armed candidates
correctly at R4; it simply had no rule for *declining to arm*. And its own
prediction was already saying so: R4's predicted margin over OFF was 1.026,
the thinnest in the map, in the regime whose cost fit was the loosest.

The rule
--------

Arm only when the predicted advantage exceeds what the cost model can
actually resolve::

    margin(regime)    = predicted[best armed cell] / predicted[off]
    threshold(regime) = max(EPSILON_ARM, envelope(regime))
    arm  iff  margin - 1 > threshold(regime)

Both inputs are quantities this phase already registered, so the rule adds no
free parameter and nothing tuned on the outcome it is meant to fix:

* ``EPSILON_ARM = 0.015`` is the arming rent from the Round-1 preregistration
  -- the margin below which arming is not worth its own machinery.
* ``envelope(regime)`` is the D1' combined envelope from `d1p_fits.json`,
  ``sqrt(fit_term^2 + (Z*sigma_repro)^2) * inflation``: the cost model's own
  scored uncertainty for that regime, from a campaign that closed before this
  rule existed.

Applied to the committed D3 prediction map, the rule fires at R4 alone::

    regime  margin   threshold   armed?
    R1      1.335    0.0254      yes  (13x its threshold)
    R4      1.026    0.0724      NO   -- inside the error bar
    R5      1.632    0.1362      yes
    R5cot   1.805    0.0956      yes
    R6      1.228    0.0323      yes
    R8      1.180    0.0174      yes

Why the comparison is conservative, not merely convenient
---------------------------------------------------------

``envelope`` bounds the relative error of the DRAFT CHAIN cost `D`, while
`margin` is a throughput ratio. An armed step costs `D + verify`, so a
relative error of `e` in `D` moves throughput by `e * D/(D+verify) < e`.
Using the full envelope therefore overstates the uncertainty and fails closed
slightly more often than a tight propagation would. For a rule whose entire
purpose is to decline when unsure, erring toward declining is the right
direction, and it keeps the threshold traceable to one registered number
instead of a derivation with its own assumptions.

Honesty about provenance
------------------------

This rule was written after seeing R4 fail in D3, so its evaluation on the D3
grid is IN-SAMPLE and is reported as such. What is not in-sample: the two
numbers that set the threshold both predate it, and the firing set is
committed from the prediction map -- never from the measured grid -- before
any scoring runs (`w98_prereg_failclosed.md`). The out-of-sample check is a
fresh interleaved measurement (G98-F).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# The Round-1 registered arming rent: below this a win does not pay for the
# machinery that produced it. Kept as a floor so the rule never admits a
# margin the phase already called negligible, even in a regime whose cost fit
# happens to be unusually tight.
EPSILON_ARM = 0.015
OFF_KEY = "off"
REGIMES = ("R1", "R4", "R5", "R5cot", "R6", "R8")


class FailClosedError(RuntimeError):
    """Raised when the rule cannot be evaluated from the registered inputs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FailClosedError(message)


def envelopes_from_fits(fits: Mapping[str, Any]) -> dict[str, float]:
    """Per-regime cost-model envelope from the D1' fit record.

    Args:
        fits: The ``fits`` mapping of `d1p_fits.json`.

    Returns:
        Regime to combined log-scale envelope, for regimes that were fitted.

    Raises:
        FailClosedError: If no regime carries a usable envelope.
    """
    out = {
        regime: float(value["envelope"])
        for regime, value in fits.items()
        if value.get("fitted") and value.get("envelope") is not None
    }
    _require(bool(out), "no fitted regime carries an envelope")
    return out


def threshold(regime: str, envelopes: Mapping[str, float]) -> float:
    """The margin a configuration must clear at this regime to be armed.

    Raises:
        FailClosedError: If the regime has no registered envelope. A missing
            envelope means the cost model was never shown to resolve this
            regime, which is a reason to decline rather than to guess.
    """
    _require(regime in envelopes, f"regime {regime!r} has no fitted envelope")
    return max(EPSILON_ARM, envelopes[regime])


def decide(
    predicted: Mapping[str, Mapping[str, float]],
    regime: str,
    envelopes: Mapping[str, float],
) -> dict[str, Any]:
    """Pick a configuration for one regime, or decline to arm.

    Args:
        predicted: Cell key to regime to predicted decode rate. Must contain
            the ``off`` cell, which is the fallback and the baseline.
        regime: Regime identifier.
        envelopes: Per-regime cost-model envelopes.

    Returns:
        A record with the chosen cell, the best armed candidate, the margin,
        the threshold it had to clear, and whether the rule fired.

    Raises:
        FailClosedError: If the map lacks ``off`` or any armed candidate.
    """
    _require(OFF_KEY in predicted, "prediction map has no 'off' cell")
    off_rate = float(predicted[OFF_KEY][regime])
    _require(off_rate > 0.0, f"{regime}: 'off' predicts a non-positive rate")
    armed = {
        cell: float(rates[regime])
        for cell, rates in predicted.items()
        if cell != OFF_KEY and regime in rates
    }
    _require(bool(armed), f"{regime}: no armed candidate in the prediction map")
    best = max(armed, key=armed.__getitem__)
    margin = armed[best] / off_rate
    limit = threshold(regime, envelopes)
    arm = (margin - 1.0) > limit
    return {
        "regime": regime,
        "choice": best if arm else OFF_KEY,
        "best_armed": best,
        "margin": margin,
        "threshold": limit,
        "threshold_source": (
            "envelope" if envelopes[regime] >= EPSILON_ARM else "epsilon_arm"
        ),
        "armed": arm,
        "rule_fired": not arm,
    }


def apply_rule(
    predicted: Mapping[str, Mapping[str, float]],
    envelopes: Mapping[str, float],
    regimes: tuple[str, ...] = REGIMES,
) -> dict[str, dict[str, Any]]:
    """Evaluate the rule for every regime."""
    return {regime: decide(predicted, regime, envelopes) for regime in regimes}


def firing_set(decisions: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Regimes where the rule declines to arm, sorted."""
    return sorted(r for r, d in decisions.items() if d["rule_fired"])


def commitment_digest(decisions: Mapping[str, Mapping[str, Any]]) -> str:
    """Digest over the decisions, for the pre-measurement commitment.

    Rounded before hashing so the digest is reproducible across platforms
    without being sensitive to float formatting.
    """
    payload = {
        regime: {
            "choice": d["choice"],
            "armed": bool(d["armed"]),
            "margin": round(float(d["margin"]), 6),
            "threshold": round(float(d["threshold"]), 6),
        }
        for regime, d in sorted(decisions.items())
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_inputs(
    predictions_path: Path, fits_path: Path
) -> tuple[dict[str, Any], dict[str, float]]:
    """Read the committed prediction map and the D1' envelopes."""
    with Path(predictions_path).open(encoding="utf-8") as handle:
        predictions = json.load(handle)
    with Path(fits_path).open(encoding="utf-8") as handle:
        fits = json.load(handle)["fits"]
    _require(
        "predicted_decode_tokens_per_s" in predictions,
        "prediction map lacks predicted_decode_tokens_per_s",
    )
    return predictions["predicted_decode_tokens_per_s"], envelopes_from_fits(fits)
