#!/usr/bin/env python3
"""Emulate multi-node timing envelopes from measured shape-scaling data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REQUIRED_COLUMNS = (
    "reference_stack",
    "batch_size",
    "draft_length",
    "t_base_ms",
    "t_verify_ms",
)


def parse_float_list(raw_value: str) -> list[float]:
    values = [float(value) for value in raw_value.split(",") if value]
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    if any(value < 0.0 or value >= 1.0 for value in values):
        raise argparse.ArgumentTypeError("values must be in [0, 1)")
    return values


def stack_suffix(value: float) -> str:
    return str(int(round(value * 100))).zfill(2)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("input CSV has no header")
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames)
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(f"input CSV missing columns: {missing_columns}")
        return list(reader)


def emulated_rows(
    measured_rows: list[dict[str, str]],
    exposed_fractions: list[float],
    dbo_hide_fraction: float,
) -> list[dict[str, str]]:
    rows = []
    for row in measured_rows:
        batch_size = row["batch_size"]
        draft_length = row["draft_length"]
        t_base = float(row["t_base_ms"])
        t_verify = float(row["t_verify_ms"])
        verify_ratio = t_verify / t_base

        for f_exposed in exposed_fractions:
            local_draft = 1.0 - f_exposed
            dbo_base = 1.0 - dbo_hide_fraction * f_exposed
            f_name = stack_suffix(f_exposed)
            h_name = stack_suffix(dbo_hide_fraction)

            rows.append(
                {
                    "reference_stack": f"emu_deepep_f{f_name}",
                    "batch_size": batch_size,
                    "draft_length": draft_length,
                    "t_base_ms": "1.000000",
                    "t_draft_local_ms": f"{local_draft:.6f}",
                    "t_verify_ms": f"{verify_ratio:.6f}",
                }
            )
            rows.append(
                {
                    "reference_stack": (
                        f"emu_deepep_dbo_ref_only_f{f_name}_h{h_name}"
                    ),
                    "batch_size": batch_size,
                    "draft_length": draft_length,
                    "t_base_ms": f"{dbo_base:.6f}",
                    "t_draft_local_ms": f"{local_draft:.6f}",
                    "t_verify_ms": f"{verify_ratio:.6f}",
                }
            )
            rows.append(
                {
                    "reference_stack": (
                        f"emu_deepep_dbo_verify_overlap_f{f_name}_h{h_name}"
                    ),
                    "batch_size": batch_size,
                    "draft_length": draft_length,
                    "t_base_ms": f"{dbo_base:.6f}",
                    "t_draft_local_ms": f"{local_draft:.6f}",
                    "t_verify_ms": f"{verify_ratio * dbo_base:.6f}",
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("measured_timing_csv", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--exposed-fractions",
        type=parse_float_list,
        default=[0.2, 0.4, 0.6],
    )
    parser.add_argument("--dbo-hide-fraction", type=float, default=0.5)
    args = parser.parse_args()

    if not 0.0 <= args.dbo_hide_fraction < 1.0:
        parser.error("--dbo-hide-fraction must be in [0, 1)")

    rows = emulated_rows(
        read_rows(args.measured_timing_csv),
        args.exposed_fractions,
        args.dbo_hide_fraction,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "reference_stack",
                "batch_size",
                "draft_length",
                "t_base_ms",
                "t_draft_local_ms",
                "t_verify_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
