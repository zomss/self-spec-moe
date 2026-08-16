#!/usr/bin/env python3
"""Why are window and skip weak? Implementation, operating point, or physics?

Window-only scores 1.01x and skip-only 1.02x aggregate against quantization's
1.20x, which invites the reading that our implementations of sparse attention
and layer skipping are simply bad -- the literature reports both working well
as self-speculative levers.

The prediction map carries the two quantities that separate the hypotheses,
per cell and per regime: the measured draft-chain cost `d_hat` and the
measured acceptance `tau_hat`. A lever is weak because it fails to cut cost,
or because it costs acceptance, and those have different fixes.

Three readings this produces:

1. **Cost efficiency.** Skipping `n` of 36 layers should cut draft cost to
   `1 - n/36`. Measured against that ideal, the skip implementation is
   either efficient or carrying overhead.
2. **The windowing floor.** Windowing can only save what there is to trim.
   At short context the saving is zero by construction, and any measured
   ratio ABOVE 1.0 is implementation overhead, quantified.
3. **The operating point.** D2(b) measured that the frozen nested skip-8 set
   the cost campaigns actually booted reaches tau 5.327 where a
   knapsack-chosen set of the same COUNT reaches 6.423 (+20.6%). The lever
   was therefore evaluated below its achievable acceptance, at no extra
   cost. `--knapsack-uplift` re-scores skip-8 under that factor.

Everything here is a single-lever reading against the unmodified draft
(`target-matching/woff/skip0`), so it isolates the lever from composition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PHASE = Path(__file__).resolve().parent.parent
BASE = "target-matching/woff/skip0"
LAYERS = 36
REGIMES = ("R1", "R4", "R5", "R5cot", "R6", "R8")

SINGLES = {
    "quant w4a16": ("w4a16-quantized/woff/skip0", None),
    "window 1024": ("target-matching/w1024/skip0", None),
    "window 512": ("target-matching/w512/skip0", None),
    "window 256": ("target-matching/w256/skip0", None),
    "window 128": ("target-matching/w128/skip0", None),
    "skip 4": ("target-matching/woff/skip4", 4),
    "skip 8": ("target-matching/woff/skip8", 8),
}

# D2(b) reference arm: frozen nested skip-8 tau 5.327, knapsack-chosen set of
# the same count 6.423. Same layer COUNT, so the cost term is unchanged.
KNAPSACK_UPLIFT_SKIP8 = 6.423 / 5.327


def speedup_vs_off(entry: dict[str, float], tau_scale: float = 1.0) -> float:
    """tau * V / (d + V) -- the armed rate over the unspeculated rate."""
    verify = entry["verify_s"]
    return (
        entry["tau_hat_at_k4"] * tau_scale * verify / (entry["d_hat_s"] + verify)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", type=Path, default=PHASE / "data/g98_e/d3_predictions.json"
    )
    parser.add_argument(
        "--out", type=Path, default=PHASE / "data/g98_e/lever_mechanics.json"
    )
    args = parser.parse_args()

    detail = json.loads(args.predictions.read_text())["detail"]
    base = {r: detail[BASE][r] for r in REGIMES}

    record: dict[str, Any] = {
        "record_type": "w98_lever_mechanics",
        "schema_version": 1,
        "base_cell": BASE,
        "layers": LAYERS,
        "knapsack_uplift_skip8": round(KNAPSACK_UPLIFT_SKIP8, 6),
        "levers": {},
        "base": {
            r: {
                "d_hat_ms": round(base[r]["d_hat_s"] * 1000, 4),
                "verify_ms": round(base[r]["verify_s"] * 1000, 4),
                "tau_hat_at_k4": round(base[r]["tau_hat_at_k4"], 4),
                "speedup_vs_off": round(speedup_vs_off(base[r]), 6),
            }
            for r in REGIMES
        },
    }

    header = f"{'lever':16s}" + "".join(f"{r:>9s}" for r in REGIMES)
    print("DRAFT COST d_hat as a fraction of the unmodified draft "
          "(lower is better)")
    print(header)
    for name, (cell, count) in SINGLES.items():
        ratios = {
            r: detail[cell][r]["d_hat_s"] / base[r]["d_hat_s"] for r in REGIMES
        }
        ideal = None if count is None else 1.0 - count / LAYERS
        line = f"{name:16s}" + "".join(f"{ratios[r]:9.3f}" for r in REGIMES)
        if ideal is not None:
            line += f"   ideal {ideal:.3f}"
        print(line)
        record["levers"].setdefault(name, {})["cost_fraction"] = {
            r: round(v, 6) for r, v in ratios.items()
        }
        if ideal is not None:
            record["levers"][name]["ideal_cost_fraction"] = round(ideal, 6)
            record["levers"][name]["cost_overhead_vs_ideal"] = {
                r: round(ratios[r] / ideal - 1.0, 6) for r in REGIMES
            }

    print("\nACCEPTANCE tau_hat as a fraction of the unmodified draft "
          "(higher is better)")
    print(header)
    for name, (cell, _count) in SINGLES.items():
        ratios = {
            r: detail[cell][r]["tau_hat_at_k4"] / base[r]["tau_hat_at_k4"]
            for r in REGIMES
        }
        print(f"{name:16s}" + "".join(f"{ratios[r]:9.3f}" for r in REGIMES))
        record["levers"][name]["acceptance_fraction"] = {
            r: round(v, 6) for r, v in ratios.items()
        }

    print("\nSPEEDUP vs OFF = tau * V / (d + V). Below 1.000 means the lever "
          "LOSES to no speculation.")
    print(header)
    print(
        f"{'unmodified draft':16s}"
        + "".join(f"{speedup_vs_off(base[r]):9.3f}" for r in REGIMES)
    )
    for name, (cell, _count) in SINGLES.items():
        values = {r: speedup_vs_off(detail[cell][r]) for r in REGIMES}
        wins = sum(1 for v in values.values() if v > 1.0)
        print(
            f"{name:16s}" + "".join(f"{values[r]:9.3f}" for r in REGIMES)
            + f"   wins {wins}/6"
        )
        record["levers"][name]["speedup_vs_off"] = {
            r: round(v, 6) for r, v in values.items()
        }
        record["levers"][name]["regimes_won"] = wins

    cell = SINGLES["skip 8"][0]
    values = {
        r: speedup_vs_off(detail[cell][r], KNAPSACK_UPLIFT_SKIP8) for r in REGIMES
    }
    wins = sum(1 for v in values.values() if v > 1.0)
    print(
        f"\n{'skip 8 (knapsack set)':16s}"
        + "".join(f"{values[r]:9.3f}" for r in REGIMES)
        + f"   wins {wins}/6"
    )
    print(
        "  ESTIMATE: D2(b)'s +20.6% acceptance uplift at k=8 applied uniformly. "
        "It was\n  measured on one regime's retention probe, so the per-regime "
        "spread is not known."
    )
    record["skip8_knapsack_estimate"] = {
        "speedup_vs_off": {r: round(v, 6) for r, v in values.items()},
        "regimes_won": wins,
        "caveat": (
            "uniform application of a single-regime uplift; direction is "
            "measured, per-regime magnitude is not"
        ),
    }

    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
