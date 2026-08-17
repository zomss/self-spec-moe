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
    """T=0 cross-arm output identity, hard failure.

    ``reference`` maps prompt_index -> sha256 of the stock-AR arm's
    output text in the SAME campaign; ``measured`` likewise for the arm
    under test. Any divergence is a lossless-verify correctness alarm,
    not noise.
    """
    diverged = sorted(i for i in measured
                      if reference.get(i) != measured[i])
    if diverged:
        raise ProtocolViolation(
            f"{cell}: T=0 output divergence vs stock at prompt_index "
            f"{diverged[:8]}{'...' if len(diverged) > 8 else ''} "
            f"({len(diverged)}/{len(measured)})")
