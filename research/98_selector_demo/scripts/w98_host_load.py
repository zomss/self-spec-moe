# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-boot host-load gate for the W98-R2 campaign.

Why this exists
---------------

The draft chain under the pinned Option-A runtime (piecewise + Marlin) is
substantially HOST-bound. When the box is loaded, the chain does not slow down
smoothly -- it CLAMPS to a host floor that hides the GPU work entirely, and the
measurement stops being a function of the levers at all.

X17 caught this directly. Three skip16 boots, same code, same GPU, minutes
apart:

    cell                     draft chain R1 -> R5      what it means
    tm/w256/skip16           19.41 -> 19.84 ms         real: flat because the
                                                       256-token window really
                                                       does bound the work
    tm/woff/skip16           29.56 -> 32.66 ms         clamped: window is OFF,
                                                       so attention MUST scale
                                                       with a 33x context range
                                                       and it did not
    w4a16/w256/skip16        31.12 -> 31.22 ms         clamped ABOVE the
                                                       unquantized boot, which
                                                       is backwards

A clamped boot is not noisy, it is wrong, and it is wrong in a direction that
looks plausible: it inflates cheap configurations toward a common floor, which
is exactly the "composition is more expensive than predicted" signature Round 1
spent this phase chasing. Nothing in the record would have flagged it.

The instrument
--------------

Every boot already logs two pure host-side quantities before any measurement
starts: torch.compile wall time and CUDA graph capture wall time. Neither
depends on the measurement, both are free, and they separate cleanly:

    boot set                       compile_s      capture_s
    v6 campaign, all 27 scored     9.72 - 11.29   8 - 9
    X17/X18 quiet boots            9.54 - 9.86    7 - 9
    X17 loud boots                 15.63, 16.15   13, 14

A 38% gap with nothing in between. `compile_s` is the primary gate because it
is nearly independent of the levers -- across v6 it moves only ~1.5 s over every
skip count and both quant arms -- while capture RATE tracks skip count strongly
(6.5 it/s at skip0 rising to 9.75 at skip16) and so is kept as a diagnostic
rather than a threshold.

Thresholds sit between the two populations with margin on both sides: quiet
tops out at 11.29 s compile / 9 s capture, loud starts at 15.63 s / 13 s.

This module does not touch any hash-bound Round-1 artifact. The Round-1 runner
is bound as `round1_runner` in the v6 authorization and its campaign is scored
and closed; the gate applies to Round 2 onward.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import regex as re

# Between the quiet population (max 11.29 s) and the loud one (min 15.63 s).
QUIET_MAX_COMPILE_S = 12.5
# Between quiet (max 9 s) and loud (min 13 s).
QUIET_MAX_CAPTURE_S = 11.0
# How many times a contaminated boot may be retried before the campaign gives
# up on the cell. A loud box usually stays loud for a while, so retrying
# forever would just burn the session.
DEFAULT_MAX_ATTEMPTS = 3
# Cheap regimes in increasing batch order: R1 is batch 1, R8 is 16, R6 is 32.
# Draft chain rises along this list on every clean boot on record.
CHEAP_REGIMES_BY_BATCH = ("R1", "R8", "R6")
# Below this, the cheap regimes have collapsed toward a common floor. Clean
# boots span 0.046-0.176; the known clamped boots sit at 0.005, 0.012, 0.022.
MIN_RELATIVE_SPREAD = 0.03

_COMPILE_RE = re.compile(
    r"Compiling a graph for compile range \([^)]*\) takes ([\d.]+) s"
)
_CAPTURE_SECS_RE = re.compile(r"Graph capturing finished in (\d+) secs")
# Emitted instead of the compile line when the torch.compile cache is warm.
_CACHE_LOAD_RE = re.compile(
    r"Directly load the compiled graph\(s\) for compile range \([^)]*\) "
    r"from the cache, took ([\d.]+) s"
)
_PIECEWISE_RE = re.compile(
    r"PIECEWISE\): 100%\|[^|]*\| (\d+)/\1 \[[^\]]*?([\d.]+)it/s\]"
)
_FULL_RE = re.compile(r"decode, FULL\): 100%\|[^|]*\| (\d+)/\1 \[[^\]]*?([\d.]+)it/s\]")
# Emitted per draft chain call under the pinned piecewise runtime.
_PIECEWISE_RUNTIME_MARK = "Draft chain PIECEWISE:"
# Whole-chain announces its augmented capture sizes and never logs the above.
_WHOLECHAIN_RUNTIME_MARK = "capture sizes augmented"


class HostLoadError(RuntimeError):
    """Raised when a boot cannot be shown to have run on a quiet box."""


@dataclass(frozen=True)
class BootLoadSignature:
    """Host-side cost of bringing one boot up, before it measures anything.

    Attributes:
        compile_s: Total torch.compile wall time across all compile ranges.
        capture_s: CUDA graph capture wall time.
        piecewise_it_s: Piecewise capture rate, diagnostic only.
        full_it_s: Full-decode capture rate, diagnostic only.
        runtime: ``piecewise``, ``wholechain``, or ``unknown``.
    """

    compile_s: float | None
    capture_s: float | None
    piecewise_it_s: float | None
    full_it_s: float | None
    runtime: str = "unknown"
    compile_cached: bool = False
    cache_load_s: float | None = None

    @property
    def verdict(self) -> str:
        """``quiet``, ``loud``, ``not_applicable``, or ``unknown``.

        ``not_applicable`` is returned for any runtime other than the pinned
        piecewise one. Whole-chain captures a much larger graph budget, so its
        capture time is dominated by the cut-point count rather than by host
        contention -- a legitimate whole-chain boot in this phase took 86 s to
        capture against a perfectly normal 10.35 s compile. Judging it by the
        piecewise band would call correct work contaminated.

        A WARM COMPILE CACHE has no compile time to measure. The boot logs
        "Directly load the compiled graph(s) ... from the cache" instead, and
        that is the normal state for any repeated configuration -- the campaign
        boots each anchor three times per stage, so all but the first are warm
        by construction. Treating the absent compile time as ``unknown`` made
        the gate fail closed and reject those boots, which would have aborted
        the campaign at its anchors on a perfectly quiet box. Such boots are
        judged on capture time alone.
        """
        if self.capture_s is None:
            return "unknown"
        if self.compile_s is None and not self.compile_cached:
            # No compile time AND no evidence of a cache hit: the log is not
            # one this gate understands, so it does not get a pass.
            return "unknown"
        if self.runtime != "piecewise":
            return "not_applicable"
        if self.compile_s is not None and self.compile_s > QUIET_MAX_COMPILE_S:
            return "loud"
        if self.capture_s > QUIET_MAX_CAPTURE_S:
            return "loud"
        return "quiet"

    @property
    def reason(self) -> str | None:
        """Which threshold failed, for the campaign record."""
        verdict = self.verdict
        if verdict == "unknown":
            return "boot log has no compile/capture markers"
        if verdict == "not_applicable":
            return f"gate is calibrated for piecewise, boot ran {self.runtime}"
        if verdict == "quiet":
            return None
        parts = []
        if self.compile_s is not None and self.compile_s > QUIET_MAX_COMPILE_S:
            parts.append(f"compile {self.compile_s:.2f}s > {QUIET_MAX_COMPILE_S}s")
        if self.capture_s is not None and self.capture_s > QUIET_MAX_CAPTURE_S:
            parts.append(f"capture {self.capture_s:.0f}s > {QUIET_MAX_CAPTURE_S}s")
        return "; ".join(parts)

    def as_record(self) -> dict[str, Any]:
        """Serialisable form, with the verdict resolved."""
        return {**asdict(self), "verdict": self.verdict, "reason": self.reason}


def runtime_from_log(text: str) -> str:
    """Identify which draft runtime a boot actually ran.

    A piecewise boot logs ``Draft chain PIECEWISE:`` on every draft chain call.
    A whole-chain boot never does, and instead reports its augmented capture
    sizes. Detecting this from the log rather than trusting the environment
    means a boot that silently fell back is classified by what it DID -- the
    lesson from the inert whole-chain fix earlier in this phase, where the
    environment claimed one runtime and eight boots ran another.

    Args:
        text: The full boot log.

    Returns:
        ``piecewise``, ``wholechain``, or ``unknown``.
    """
    if _PIECEWISE_RUNTIME_MARK in text:
        return "piecewise"
    if _WHOLECHAIN_RUNTIME_MARK in text:
        return "wholechain"
    return "unknown"


def signature_from_log(text: str) -> BootLoadSignature:
    """Extract the host-load signature from one boot's captured stdout.

    Args:
        text: The full boot log.

    Returns:
        The signature; fields are None when the corresponding marker is absent,
        which drives an ``unknown`` verdict rather than a silent pass.
    """
    compiles = [float(v) for v in _COMPILE_RE.findall(text)]
    capture = _CAPTURE_SECS_RE.search(text)
    piecewise = _PIECEWISE_RE.findall(text)
    full = _FULL_RE.findall(text)
    cache_loads = [float(v) for v in _CACHE_LOAD_RE.findall(text)]
    return BootLoadSignature(
        compile_s=sum(compiles) if compiles else None,
        capture_s=float(capture.group(1)) if capture else None,
        # tqdm rewrites the line, so the LAST match is the completed rate.
        piecewise_it_s=float(piecewise[-1][1]) if piecewise else None,
        full_it_s=float(full[-1][1]) if full else None,
        runtime=runtime_from_log(text),
        compile_cached=bool(cache_loads) and not compiles,
        # Recorded, not gated: there is no calibration for how long a warm
        # cache load should take on a quiet box.
        cache_load_s=sum(cache_loads) if cache_loads else None,
    )


def signature_from_log_path(path: Path) -> BootLoadSignature:
    """``signature_from_log`` for a log on disk; missing file yields unknown."""
    if not path.is_file():
        return BootLoadSignature(None, None, None, None)
    return signature_from_log(path.read_text(encoding="utf-8", errors="replace"))


def _lane_cpu_ids(cpu_affinity: str) -> set[int]:
    """Expand a taskset-style CPU list such as ``0-15,32`` into ids."""
    ids: set[int] = set()
    for part in cpu_affinity.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.update(range(int(lo), int(hi) + 1))
        else:
            ids.add(int(part))
    return ids


def foreign_lane_processes(cpu_affinity: str, min_pcpu: float = 50.0) -> list[str]:
    """Find busy processes outside this session that may land on the lane.

    A process contends only if it is both busy and ALLOWED to run on the lane's
    CPUs. X17's co-tenants were unpinned (``Cpus_allowed_list`` 0-191) while the
    lane pins to 0-15, so they were free to preempt the launching thread that
    the host-bound draft chain depends on.

    Args:
        cpu_affinity: The lane's taskset CPU list, e.g. ``"0-15"``.
        min_pcpu: Ignore processes below this %CPU.

    Returns:
        Human-readable descriptions of the contending processes.
    """
    lane = _lane_cpu_ids(cpu_affinity)
    session = {os.getpid(), os.getppid()}
    try:
        listing = subprocess.run(
            ["ps", "-eo", "pid,pcpu,comm", "--no-headers"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    out: list[str] = []
    for line in listing.splitlines():
        fields = line.split(None, 2)
        if len(fields) < 3:
            continue
        try:
            pid, pcpu = int(fields[0]), float(fields[1])
        except ValueError:
            continue
        if pid in session or pcpu < min_pcpu:
            continue
        try:
            status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        except OSError:
            continue
        match = re.search(r"^Cpus_allowed_list:\s*(\S+)", status, re.MULTILINE)
        if not match:
            continue
        if _lane_cpu_ids(match.group(1)) & lane:
            out.append(f"pid {pid} {fields[2].strip()} {pcpu:.0f}%cpu")
    return out


def host_snapshot(cpu_affinity: str | None = None) -> dict[str, Any]:
    """Advisory pre-boot picture of the box.

    This does not gate anything on its own -- the authoritative signal is the
    post-boot signature, which reflects conditions DURING the boot rather than
    before it. It is recorded so a contaminated boot can be explained.

    Args:
        cpu_affinity: The lane's CPU list, if contention should be listed.

    Returns:
        Load averages and, when a lane is given, the contending processes.
    """
    load1, load5, load15 = os.getloadavg()
    snapshot: dict[str, Any] = {"loadavg": [load1, load5, load15]}
    if cpu_affinity:
        snapshot["lane_cpu_affinity"] = cpu_affinity
        snapshot["foreign_lane_processes"] = foreign_lane_processes(cpu_affinity)
    return snapshot


@dataclass(frozen=True)
class MeasurementVerdict:
    """Within-boot check on the measurement itself, after it has run.

    The startup signature reads compile and capture time, which happen BEFORE
    any measurement. A boot whose lane goes loud afterwards passes it and still
    produces a clamped measurement -- that is exactly what aborted the first
    G98-C run, at +25.8% on a cell with 0.027% historical CV.

    This looks at the shape of the result instead, and needs no cross-session
    reference. Two independent signatures, because contamination has two shapes:

    * **ordering** -- draft chain must rise with batch across the cheap
      regimes (R1 batch 1, R8 batch 16, R6 batch 32). A partial clamp lifts the
      cheapest regimes onto a floor ABOVE the most expensive one and inverts
      this. All 25 clean boots on record are monotonic; the aborted campaign
      boot is the only non-monotonic boot ever recorded.
    * **relative spread** -- a hard clamp collapses the cheap regimes onto a
      common value. Clean boots span 0.046 to 0.176; the three known clamped
      boots sit at 0.005, 0.012 and 0.022.

    Calibrated on 29 boots (25 clean, 4 contaminated), where the combined rule
    gives zero false positives and catches all four. The threshold sits between
    the populations but the margin is only ~2x, and the spread limb is a
    heuristic where the ordering limb is physically motivated. Treat a
    borderline rejection as worth reading, not as proof.
    """

    values_ms: tuple[float, ...]
    monotonic: bool
    relative_spread: float

    @property
    def verdict(self) -> str:
        """``clean`` or ``clamped``."""
        if not self.monotonic:
            return "clamped"
        if self.relative_spread < MIN_RELATIVE_SPREAD:
            return "clamped"
        return "clean"

    @property
    def reason(self) -> str | None:
        if self.monotonic and self.relative_spread >= MIN_RELATIVE_SPREAD:
            return None
        if not self.monotonic:
            return (
                "cheap regimes are not increasing in batch "
                f"({', '.join(f'{v:.2f}' for v in self.values_ms)} ms): a clamp "
                "lifts the cheapest regimes onto a floor above the dearest"
            )
        return (
            f"cheap-regime relative spread {self.relative_spread:.4f} < "
            f"{MIN_RELATIVE_SPREAD}: the regimes collapsed toward a common floor"
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "values_ms": list(self.values_ms),
            "monotonic": self.monotonic,
            "relative_spread": self.relative_spread,
            "verdict": self.verdict,
            "reason": self.reason,
        }


def measurement_verdict(observations: Mapping[str, Any]) -> MeasurementVerdict | None:
    """Judge a completed boot by the shape of its own measurement.

    Args:
        observations: Per-regime records carrying ``draft_chain_s``.

    Returns:
        The verdict, or None if the cheap regimes are not all present, in which
        case the caller should not treat silence as a pass.
    """
    values = []
    for regime in CHEAP_REGIMES_BY_BATCH:
        entry = observations.get(regime) or {}
        value = entry.get("draft_chain_s")
        if not value:
            return None
        values.append(value * 1000.0)
    monotonic = all(values[i] < values[i + 1] for i in range(len(values) - 1))
    return MeasurementVerdict(
        values_ms=tuple(values),
        monotonic=monotonic,
        relative_spread=(max(values) - min(values)) / values[0],
    )


def guarded_boot(
    label: str,
    boot: Callable[[int], str],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    on_reject: Callable[[str, int, BootLoadSignature], None] | None = None,
    inspect: Callable[[int], str | None] | None = None,
) -> tuple[BootLoadSignature, list[dict[str, Any]]]:
    """Run one campaign boot, retrying while the box is loud.

    This is the whole integration surface for a campaign runner: hand it a
    callable that performs one boot and returns that boot's log text, and it
    returns only once a boot has been shown to run quiet.

    Every attempt is returned, rejected ones included, so the campaign record
    shows what was discarded rather than silently reporting the survivor. A
    campaign that quietly retries until it gets a number it likes is doing
    selection, not measurement -- the discarded attempts are what distinguish
    the two.

    Args:
        label: Cell name, used in the raised message and the attempt log.
        boot: Callable taking the 1-based attempt number and returning the
            boot's log text. It is responsible for writing its own outputs.
        max_attempts: How many times to retry a loud box before giving up.
        on_reject: Optional hook called with (label, attempt, signature) after
            each rejected attempt, e.g. to discard that attempt's output files.
        inspect: Optional post-measurement check, called with the attempt
            number once the startup gate has passed. Returning a string
            rejects the attempt with that reason; returning None accepts it.
            This is where the measurement-shape gate runs, because the startup
            signature cannot see contamination that begins after capture.

    Returns:
        The accepted boot's signature, and the per-attempt record.

    Raises:
        HostLoadError: If no attempt produced a clean measurement on a quiet
            box.
    """
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        signature = signature_from_log(boot(attempt))
        record = {"attempt": attempt, **signature.as_record()}
        rejection = None if signature.verdict == "quiet" else signature.reason
        if rejection is None and inspect is not None:
            rejection = inspect(attempt)
            if rejection is not None:
                # The startup gate passed and the measurement did not; say so,
                # rather than filing it under "loud box".
                record["verdict"] = "clamped"
                record["reason"] = rejection
        attempts.append(record)
        if rejection is None:
            return signature, attempts
        if on_reject is not None:
            on_reject(label, attempt, signature)
    raise HostLoadError(
        f"{label}: no clean boot in {max_attempts} attempts -- "
        + "; ".join(f"#{a['attempt']} {a['verdict']} ({a['reason']})" for a in attempts)
    )


def require_quiet(signature: BootLoadSignature, label: str) -> None:
    """Raise unless the boot demonstrably ran on a quiet box.

    Fails closed: an ``unknown`` verdict raises too, so a log whose markers
    moved cannot pass by accident.

    Args:
        signature: The boot's host-load signature.
        label: Cell name, for the message.

    Raises:
        HostLoadError: If the boot was loud, or cannot be shown to be quiet.
    """
    if signature.verdict != "quiet":
        raise HostLoadError(
            f"{label}: host-load gate {signature.verdict} -- {signature.reason}"
        )


def scan(data_root: Path, patterns: list[str]) -> dict[str, Any]:
    """Extract signatures for every boot log under ``data_root``.

    Boot logs are gitignored (`*.log`) and traces are too, so the evidence the
    thresholds rest on cannot live in the logs themselves. This distils each log
    to the four numbers that matter and yields a small artifact that CAN be
    checked in, reviewed, and regression-tested against.

    Args:
        data_root: Phase data directory.
        patterns: Globs relative to ``data_root``.

    Returns:
        A record keyed by boot name, with the thresholds that produced it.
    """
    boots: dict[str, Any] = {}
    for pattern in patterns:
        for path in sorted(data_root.glob(pattern)):
            signature = signature_from_log_path(path)
            if signature.compile_s is None and signature.capture_s is None:
                continue
            # Key on the path relative to the data root. Keying on the parent
            # directory alone collides across campaigns -- every g98_b_v* has a
            # `singles/`, so later scans would silently overwrite earlier ones.
            key = str(path.relative_to(data_root).with_suffix(""))
            boots[key] = signature.as_record()
    return {
        "schema_version": 1,
        "record_type": "w98_host_load_signatures",
        "thresholds": {
            "quiet_max_compile_s": QUIET_MAX_COMPILE_S,
            "quiet_max_capture_s": QUIET_MAX_CAPTURE_S,
        },
        "boots": boots,
    }


def main() -> int:
    """Record host-load signatures for every boot log under a data root."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pattern", action="append", default=None)
    args = parser.parse_args()
    record = scan(args.data_root, args.pattern or ["**/*.log"])
    args.out.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    counts: dict[str, int] = {}
    for boot in record["boots"].values():
        counts[boot["verdict"]] = counts.get(boot["verdict"], 0) + 1
    print(json.dumps({"boots": len(record["boots"]), "verdicts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
