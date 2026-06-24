#!/usr/bin/env python3
"""Emulate multi-node communication by injecting exposed delay analytically."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_float_list(raw_value: str) -> list[float]:
    values = [float(value) for value in raw_value.split(",") if value]
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    if any(value < 0.0 for value in values):
        raise argparse.ArgumentTypeError("all values must be non-negative")
    return values


def parse_int_list(raw_value: str) -> list[int]:
    values = [int(value) for value in raw_value.split(",") if value]
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute-local-ms", type=float, required=True)
    parser.add_argument("--verify-compute-ratio", type=float, default=1.0)
    parser.add_argument("--batch-sizes", type=parse_int_list, default=[1, 2, 4, 8])
    parser.add_argument("--draft-lengths", type=parse_int_list, default=[1, 2, 4, 8])
    parser.add_argument(
        "--comm-delays-ms",
        type=parse_float_list,
        default=[0.5, 1.0, 2.0, 4.0, 8.0],
    )
    parser.add_argument(
        "--dbo-hide-fraction",
        type=float,
        default=0.5,
        help="Fraction of exposed communication hidden in DBO reference stack.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    if args.compute_local_ms <= 0.0:
        parser.error("--compute-local-ms must be positive")
    if args.verify_compute_ratio <= 0.0:
        parser.error("--verify-compute-ratio must be positive")
    if not 0.0 <= args.dbo_hide_fraction < 1.0:
        parser.error("--dbo-hide-fraction must be in [0, 1)")

    rows = []
    for batch_size in args.batch_sizes:
        for draft_length in args.draft_lengths:
            verify_compute = args.compute_local_ms * args.verify_compute_ratio
            for delay_ms in args.comm_delays_ms:
                suffix = str(delay_ms).replace(".", "p")
                t_base = args.compute_local_ms + delay_ms
                t_draft = args.compute_local_ms
                t_verify = verify_compute + delay_ms

                rows.append(
                    {
                        "reference_stack": f"emu_delay_{suffix}ms",
                        "batch_size": batch_size,
                        "draft_length": draft_length,
                        "t_base_ms": f"{t_base:.6f}",
                        "t_draft_local_ms": f"{t_draft:.6f}",
                        "t_verify_ms": f"{t_verify:.6f}",
                    }
                )

                hidden_delay = delay_ms * (1.0 - args.dbo_hide_fraction)
                rows.append(
                    {
                        "reference_stack": (
                            f"emu_delay_{suffix}ms_dbo_h"
                            f"{int(args.dbo_hide_fraction * 100)}"
                        ),
                        "batch_size": batch_size,
                        "draft_length": draft_length,
                        "t_base_ms": f"{args.compute_local_ms + hidden_delay:.6f}",
                        "t_draft_local_ms": f"{t_draft:.6f}",
                        "t_verify_ms": f"{verify_compute + hidden_delay:.6f}",
                    }
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
