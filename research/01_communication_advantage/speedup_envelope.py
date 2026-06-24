#!/usr/bin/env python3
"""Compute the communication-only speedup envelope for local MoE drafting.

The calculator intentionally ignores model-quality effects except for a
synthetic per-token acceptance probability beta. It answers the first go/no-go
question: even with perfect or high acceptance, does communication-disabled
draft add enough speedup on top of the measured reference stack?
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path


REQUIRED_COLUMNS = (
    "reference_stack",
    "batch_size",
    "draft_length",
    "t_base_ms",
    "t_draft_local_ms",
    "t_verify_ms",
)


@dataclass(frozen=True)
class TimingPoint:
    reference_stack: str
    batch_size: int
    draft_length: int
    t_base_ms: float
    t_draft_local_ms: float
    t_verify_ms: float

    @property
    def cycle_ms(self) -> float:
        return self.draft_length * self.t_draft_local_ms + self.t_verify_ms


@dataclass(frozen=True)
class EnvelopePoint:
    timing: TimingPoint
    target_speedup: float
    s_max: float
    beta_min: float | None


@dataclass(frozen=True)
class SweepPoint:
    timing: TimingPoint
    beta: float
    speedup: float


def expected_tokens(beta: float, draft_length: int) -> float:
    """Return E[emitted tokens] for independent per-token acceptance beta."""
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    return sum(beta**i for i in range(draft_length + 1))


def speedup(timing: TimingPoint, beta: float) -> float:
    """Return speedup relative to the measured reference stack latency."""
    return expected_tokens(beta, timing.draft_length) * timing.t_base_ms / (
        timing.cycle_ms
    )


def beta_min_for_target(
    timing: TimingPoint,
    target_speedup: float,
    *,
    tolerance: float = 1e-6,
) -> float | None:
    """Return the minimum beta needed to reach target_speedup.

    Returns None when even beta=1 cannot reach the target.
    """
    if speedup(timing, 1.0) < target_speedup:
        return None
    if speedup(timing, 0.0) >= target_speedup:
        return 0.0

    lo = 0.0
    hi = 1.0
    while hi - lo > tolerance:
        mid = (lo + hi) / 2.0
        if speedup(timing, mid) >= target_speedup:
            hi = mid
        else:
            lo = mid
    return hi


def parse_positive_float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{key} must be positive, got {row[key]!r}")
    return value


def parse_positive_int(row: dict[str, str], key: str) -> int:
    value = int(row[key])
    if value <= 0:
        raise ValueError(f"{key} must be positive, got {row[key]!r}")
    return value


def read_timings(path: Path) -> list[TimingPoint]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("input CSV has no header")
        fieldnames = set(reader.fieldnames)
        has_reference_stack = "reference_stack" in fieldnames
        has_legacy_baseline = "baseline" in fieldnames
        expected_columns = set(REQUIRED_COLUMNS) - {"reference_stack"}
        missing = expected_columns - fieldnames
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(f"input CSV missing columns: {missing_columns}")
        if not has_reference_stack and not has_legacy_baseline:
            raise ValueError("input CSV missing column: reference_stack")

        timings = []
        for row_number, row in enumerate(reader, start=2):
            try:
                timings.append(
                    TimingPoint(
                        reference_stack=(
                            row["reference_stack"]
                            if has_reference_stack
                            else row["baseline"]
                        ),
                        batch_size=parse_positive_int(row, "batch_size"),
                        draft_length=parse_positive_int(row, "draft_length"),
                        t_base_ms=parse_positive_float(row, "t_base_ms"),
                        t_draft_local_ms=parse_positive_float(
                            row, "t_draft_local_ms"
                        ),
                        t_verify_ms=parse_positive_float(row, "t_verify_ms"),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid row {row_number}: {exc}") from exc

    if not timings:
        raise ValueError("input CSV contains no timing rows")
    return timings


def compute_envelope(
    timings: list[TimingPoint],
    target_speedup: float,
) -> list[EnvelopePoint]:
    if not math.isfinite(target_speedup) or target_speedup <= 0.0:
        raise ValueError(f"target speedup must be positive, got {target_speedup}")
    return [
        EnvelopePoint(
            timing=timing,
            target_speedup=target_speedup,
            s_max=speedup(timing, 1.0),
            beta_min=beta_min_for_target(timing, target_speedup),
        )
        for timing in timings
    ]


def compute_sweep(
    timings: list[TimingPoint],
    betas: list[float],
) -> list[SweepPoint]:
    return [
        SweepPoint(timing=timing, beta=beta, speedup=speedup(timing, beta))
        for timing in timings
        for beta in betas
    ]


def format_float(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def print_csv(points: list[EnvelopePoint]) -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "reference_stack",
            "batch_size",
            "draft_length",
            "t_base_ms",
            "t_draft_local_ms",
            "t_verify_ms",
            "cycle_ms",
            "target_speedup",
            "s_max",
            "beta_min",
            "verdict",
        ]
    )
    for point in points:
        timing = point.timing
        writer.writerow(
            [
                timing.reference_stack,
                timing.batch_size,
                timing.draft_length,
                format_float(timing.t_base_ms),
                format_float(timing.t_draft_local_ms),
                format_float(timing.t_verify_ms),
                format_float(timing.cycle_ms),
                format_float(point.target_speedup),
                format_float(point.s_max),
                format_float(point.beta_min),
                "go" if point.beta_min is not None else "no-go",
            ]
        )


def print_markdown(points: list[EnvelopePoint]) -> None:
    print(
        "| reference_stack | B | k | T_base_ms | T_draft_local_ms | "
        "T_verify_ms | cycle_ms | target | S_max | beta_min | verdict |"
    )
    print(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | --- |"
    )
    for point in points:
        timing = point.timing
        verdict = "go" if point.beta_min is not None else "no-go"
        print(
            f"| {timing.reference_stack} | {timing.batch_size} | "
            f"{timing.draft_length} | {format_float(timing.t_base_ms)} | "
            f"{format_float(timing.t_draft_local_ms)} | "
            f"{format_float(timing.t_verify_ms)} | "
            f"{format_float(timing.cycle_ms)} | "
            f"{format_float(point.target_speedup)} | "
            f"{format_float(point.s_max)} | {format_float(point.beta_min)} | "
            f"{verdict} |"
        )


def print_sweep_csv(points: list[SweepPoint]) -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "reference_stack",
            "batch_size",
            "draft_length",
            "t_base_ms",
            "t_draft_local_ms",
            "t_verify_ms",
            "cycle_ms",
            "beta",
            "speedup",
        ]
    )
    for point in points:
        timing = point.timing
        writer.writerow(
            [
                timing.reference_stack,
                timing.batch_size,
                timing.draft_length,
                format_float(timing.t_base_ms),
                format_float(timing.t_draft_local_ms),
                format_float(timing.t_verify_ms),
                format_float(timing.cycle_ms),
                format_float(point.beta),
                format_float(point.speedup),
            ]
        )


def print_sweep_markdown(points: list[SweepPoint]) -> None:
    print(
        "| reference_stack | B | k | cycle_ms | beta | speedup |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for point in points:
        timing = point.timing
        print(
            f"| {timing.reference_stack} | {timing.batch_size} | "
            f"{timing.draft_length} | {format_float(timing.cycle_ms)} | "
            f"{format_float(point.beta)} | {format_float(point.speedup)} |"
        )


def parse_betas(raw_betas: str) -> list[float]:
    betas = []
    for raw_beta in raw_betas.split(","):
        beta = float(raw_beta)
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {raw_beta!r}")
        betas.append(beta)
    if not betas:
        raise ValueError("at least one beta is required")
    return betas


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute S_max and beta_min for communication-disabled local MoE "
            "drafting from measured reference-stack/draft/verify timings."
        )
    )
    parser.add_argument(
        "timings_csv",
        type=Path,
        help=(
            "CSV with columns: "
            "reference_stack,batch_size,draft_length,t_base_ms,"
            "t_draft_local_ms,t_verify_ms. Legacy baseline column is also "
            "accepted."
        ),
    )
    parser.add_argument(
        "--target-speedup",
        type=float,
        default=1.0,
        help="Speedup threshold used to compute beta_min. Defaults to break-even.",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "markdown"),
        default="markdown",
        help="Output format.",
    )
    parser.add_argument(
        "--acceptance-rates",
        help=(
            "Comma-separated beta values for an acceptance sweep. When set, "
            "the script outputs speedup(beta) instead of beta_min."
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        timings = read_timings(args.timings_csv)
        if args.acceptance_rates:
            betas = parse_betas(args.acceptance_rates)
            sweep_points = compute_sweep(timings, betas)
        else:
            points = compute_envelope(timings, args.target_speedup)
    except ValueError as exc:
        parser.error(str(exc))

    if args.acceptance_rates:
        if args.format == "csv":
            print_sweep_csv(sweep_points)
        else:
            print_sweep_markdown(sweep_points)
    else:
        if args.format == "csv":
            print_csv(points)
        else:
            print_markdown(points)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
