#!/usr/bin/env python3
"""Registered W100 scoring rules, as code (registration items 3-5).

Campaign runners import this module and call, in order:

* ``boot_gate(cpu_affinity)`` before every boot -- foreign-lane check via
  the phase-98 host-load machinery, fail closed.
* ``measurement_gate(observations)`` after every measurement --
  `w98_host_load.measurement_verdict`, INLINE (the G98-F lesson: applied
  post-hoc it can only annotate, applied inline it rejects the boot).
* ``cap_rule(cell, finish_reasons)`` per (cell, arm, batch).
* ``identity_gate(cell, reference, measured)`` at T=0, against the same
  campaign's stock-AR arm.

A runner that skips any of these is not running the registered protocol.
This module is digest-pinned by the barrier; changing it after
registration is a protocol amendment and needs a superseding barrier.
"""
from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

_W98_SCRIPTS = (Path(__file__).resolve().parent.parent.parent
                / "98_selector_demo" / "scripts")
sys.path.insert(0, str(_W98_SCRIPTS))

import w98_host_load as hostload  # noqa: E402

# ---- registered grid constants (mirror w100_prompt_manifest.json) ----

CELL_BATCHES = {"LI": (1, 8, 16), "LO": (1, 8, 16, 32),
                "LIO": (1, 8, 16), "SS": (1, 8, 32, 64)}
CELL_MAX_N = {"LI": 32, "LO": 60, "LIO": 32, "SS": 128}
CAPS = {"LI": 2048, "LO": 32_768, "LIO": 4096, "SS": 1024}
CAP_HIT_LIMIT = 0.10
# LO is exempt from the 10% raise-and-rerun rule: its cap is KnapSpec's
# own AIME budget (parity is the cell's purpose), sits near Qwen3-8B's
# native 40960, and at T=0 cap-hits are arm-invariant (losslessness =>
# identical greedy streams => identical caps => exactly equal work).
CAP_EXEMPT_CELLS = frozenset({"LO"})


def n_for(cell: str, batch: int) -> int:
    return min(CELL_MAX_N[cell], max(16, 2 * batch))


class ProtocolViolation(RuntimeError):
    """A registered rule failed; the measurement must be discarded."""


def boot_gate(cpu_affinity: str) -> dict:
    """Refuse to boot on a contaminated lane. Returns the host snapshot
    for the record."""
    foreign = hostload.foreign_lane_processes(cpu_affinity)
    if foreign:
        raise ProtocolViolation(
            f"foreign processes on lane cpus: {foreign[:5]}")
    return hostload.host_snapshot(cpu_affinity)


def measurement_gate(step_ms_by_batch: Mapping[int, float]) -> dict:
    """Inline measurement-shape verdict, fail closed.

    Phase-98's `hostload.measurement_verdict` is hard-bound to its
    R1/R8/R6 cheap-regime triple, so W100 re-applies the SAME two limbs
    to its own analog: mean decode step time across the SS cell's batch
    sweep. Physics unchanged -- step time must RISE with batch
    (a partial clamp lifts cheap points onto a floor above the dear
    ones) and must not collapse onto a common value (hard clamp). The
    spread threshold is inherited from phase-98's calibration and is
    conservative here (SS b1->b64 spans far more than the calibrated
    0.046 floor); monotonicity is the load-bearing limb. Recalibrate the
    spread limb once the first clean W100 boots provide references.
    """
    if len(step_ms_by_batch) < 3:
        raise ProtocolViolation(
            "measurement gate needs >=3 SS batch points to judge shape")
    batches = sorted(step_ms_by_batch)
    values = [step_ms_by_batch[b] for b in batches]
    if any(not v for v in values):
        raise ProtocolViolation("missing SS step values; not a pass")
    monotonic = all(values[i] < values[i + 1]
                    for i in range(len(values) - 1))
    spread = (max(values) - min(values)) / values[0]
    rec = {"batches": batches, "step_ms": values,
           "monotonic": monotonic, "relative_spread": round(spread, 4)}
    if not monotonic:
        raise ProtocolViolation(
            f"SS step time not increasing in batch ({values}): clamp "
            "signature")
    if spread < hostload.MIN_RELATIVE_SPREAD:
        raise ProtocolViolation(
            f"SS batch spread {spread:.4f} < "
            f"{hostload.MIN_RELATIVE_SPREAD}: regimes collapsed onto a "
            "common floor")
    rec["verdict"] = "clean"
    return rec


def cap_rule(cell: str, finish_reasons: Sequence[str]) -> dict:
    """Cap-hit fraction, with the registered LO exemption."""
    hits = sum(f == "length" for f in finish_reasons)
    frac = hits / len(finish_reasons)
    rec = {"cell": cell, "cap_hits": hits, "n": len(finish_reasons),
           "fraction": round(frac, 4),
           "exempt": cell in CAP_EXEMPT_CELLS}
    if frac > CAP_HIT_LIMIT and cell not in CAP_EXEMPT_CELLS:
        raise ProtocolViolation(
            f"{cell}: cap-hit fraction {frac:.0%} exceeds "
            f"{CAP_HIT_LIMIT:.0%}; raise the cap and re-run")
    return rec


def identity_gate(cell: str, reference: Mapping[int, str],
                  measured: Mapping[int, str]) -> None:
    """T=0 output identity, hard failure.

    AMENDED (w100_prereg_amendment1.md, rule 3c): valid only where
    bit-equality is guaranteed — same engine config, same batching,
    i.e. re-boots of the SAME arm. Applying it across engine configs
    (any spec arm vs stock) was the v1 design error: vLLM is not
    batch-invariant across schedulers, greedy ties flip at ~0.5%/token,
    and the gate fires on numerics rather than bugs. Cross-config
    checks use length_sanity_gate + divergence_diagnostic instead.
    """
    diverged = sorted(i for i in measured
                      if reference.get(i) != measured[i])
    if diverged:
        raise ProtocolViolation(
            f"{cell}: T=0 output divergence at prompt_index "
            f"{diverged[:8]}{'...' if len(diverged) > 8 else ''} "
            f"({len(diverged)}/{len(measured)})")


# Amendment 1, rule 3a: mean-length band and sign-test thresholds.
LENGTH_MEAN_BAND = 0.10
SIGN_TEST_LIMIT = 0.80
LENGTH_MEAN_EXEMPT = frozenset({"LO"})   # chaotic amplification


def length_sanity_gate(cell: str, reference_out: Mapping[int, int],
                       measured_out: Mapping[int, int]) -> dict:
    """Cross-engine-config bug catcher (amendment 1, rule 3a).

    ``reference_out``/``measured_out`` map prompt_index -> output token
    count at T=0 (stock vs arm). Symmetric tie-flip jitter passes; a
    lever bleeding into the target path fails on the mean band or the
    one-sided sign test.
    """
    n = len(measured_out)
    ref_mean = sum(reference_out[i] for i in measured_out) / n
    got_mean = sum(measured_out.values()) / n
    rel = got_mean / ref_mean - 1.0
    deltas = [measured_out[i] - reference_out[i] for i in measured_out]
    nonzero = [d for d in deltas if d]
    pos_frac = (sum(d > 0 for d in nonzero) / len(nonzero)
                if nonzero else 0.0)
    rec = {"cell": cell, "n": n, "mean_rel_delta": round(rel, 4),
           "pos_fraction": round(pos_frac, 4),
           "identical": sum(d == 0 for d in deltas)}
    if cell not in LENGTH_MEAN_EXEMPT and abs(rel) > LENGTH_MEAN_BAND:
        raise ProtocolViolation(
            f"{cell}: mean output length shifted {rel:+.1%} vs stock "
            f"(band ±{LENGTH_MEAN_BAND:.0%}): systematic, not jitter")
    if n >= 16 and nonzero and not (
            1 - SIGN_TEST_LIMIT <= pos_frac <= SIGN_TEST_LIMIT):
        raise ProtocolViolation(
            f"{cell}: {pos_frac:.0%} of length deltas share a sign: "
            "one-sided shift, not tie-flip jitter")
    return rec


def divergence_diagnostic(reference_sha: Mapping[int, str],
                          measured_sha: Mapping[int, str],
                          mean_out_toks: float) -> dict:
    """Amendment 1, rule 3b: reported, never gated."""
    n = len(measured_sha)
    same = sum(reference_sha.get(i) == measured_sha[i]
               for i in measured_sha)
    frac = same / n
    # (1-p)^L = frac  =>  p = 1 - frac^(1/L)
    p = (1 - frac ** (1 / mean_out_toks)) if 0 < frac < 1 else None
    return {"identical_fraction": round(frac, 4),
            "implied_flip_rate_per_token": (
                round(p, 6) if p is not None else None)}
