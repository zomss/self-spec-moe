#!/usr/bin/env python3
"""Turn X31's Nsight traces into an expected-vs-overhead account.

`probe_w98_nsys_levers.py` records, per lever combination, the NVTX range tree
of a captured window of engine steps plus a kernel-class split of the GPU time
inside it. This reads those and asks the only question that separates a lever
that is badly SUITED from one that is badly IMPLEMENTED:

    given what this lever changes, how much SHOULD the draft chain have cost,
    and where did the rest go?

## The two decompositions

**Vertical** -- the draft chain against its own NVTX children:

    draft_chain = draft_forward_first + K * draft_forward   <- model work
                + step0_* + step_* + chain_setup            <- per-step orchestration
                + kv_window_rewrite                         <- the window's bookkeeping
                + unattributed                              <- launch gaps, host stalls

Only the first line is work the lever is supposed to change. Everything below
it is fixed cost that a lever can only dilute or inflate, never justify.

**Horizontal** -- each lever against the `base` (unmodified draft) run:

* `skip N`      expects model time x (36-N)/36, and NOTHING else to move.
* `window W`    expects only the ATTENTION kernel class to shrink, by roughly
                min(L, W + sinks)/L; at short context, where the window already
                covers the sequence, it expects no change at all.
* `w4a16`       expects only the GEMM kernel class to shrink.

`overhead = measured - expected` per range and per kernel class.

## Reading it

nsys inflates absolute time through per-API overhead, so ratios within a trace
and ratios between traces taken the same way are the currency; absolute
milliseconds are not comparable to the wall-clock campaign numbers.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PHASE = Path(__file__).resolve().parent.parent
LAYERS = 36
WINDOW_SINKS = 16
DEPLOY_K = 4

# NVTX range name -> which part of the account it belongs to.
MODEL_RANGES = (":draft_forward", ":draft_forward_first")
ORCHESTRATION_PREFIXES = (":step0_", ":step_", ":chain_setup", ":cpu_")
WINDOW_RANGE = ":kv_window_rewrite"

# Approximate decode context per regime, for the window's expected saving.
REGIME_CONTEXT = {
    "R1": 512, "R4": 8192, "R5": 14336, "R5cot": 14336, "R6": 512, "R8": 512,
}


def _ms(value: str | float) -> float:
    return float(str(value).replace(",", "")) / 1e6


def load(output_dir: Path, regime: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(output_dir.glob(f"*__{regime}.json")):
        record = json.loads(path.read_text())
        if record.get("record_type") == "w98_nsys_lever":
            out[record["combination"]] = record
    return out


def ranges(record: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    rows = ((record.get("nsys") or {}).get("nvtx") or {}).get("rows") or []
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        name = row.get("Range")
        if not name:
            continue
        out[name] = {
            "instances": float(row.get("Range Instances", 0) or 0),
            "avg_ms": _ms(row.get("Proj Avg (ns)", 0) or 0),
            "total_ms": _ms(row.get("Total Proj Time (ns)", 0) or 0),
        }
    return out


def account(record: Mapping[str, Any]) -> dict[str, Any]:
    """Vertical decomposition of one combination's draft chain."""
    rows = ranges(record)
    chain = rows.get(":draft_chain")
    if not chain or not chain["instances"]:
        return {"armed": record.get("armed", False), "chain_ms": None}
    chains = chain["instances"]

    def per_chain(name: str) -> float:
        row = rows.get(name)
        return row["total_ms"] / chains if row else 0.0

    model = sum(per_chain(n) for n in MODEL_RANGES)
    window = per_chain(WINDOW_RANGE)
    orchestration = sum(
        v["total_ms"] / chains
        for k, v in rows.items()
        if k.startswith(ORCHESTRATION_PREFIXES)
    )
    total = chain["avg_ms"]
    return {
        "armed": record.get("armed", False),
        "chain_ms": round(total, 4),
        "model_ms": round(model, 4),
        "orchestration_ms": round(orchestration, 4),
        "window_rewrite_ms": round(window, 4),
        "unattributed_ms": round(total - model - orchestration - window, 4),
        "model_share": round(model / total, 4) if total else None,
        "kernel_classes_ms": {
            k: round(v / 1e6, 4)
            for k, v in ((record.get("nsys") or {}).get("kernel_classes") or {}).items()
        },
    }


def expectations(combo: str, cfg: Mapping[str, Any], regime: str) -> dict[str, Any]:
    """What the lever should change, relative to `base`, and what it should not."""
    skip = int(cfg.get("skip_count") or 0)
    window = cfg.get("window")
    quant = cfg.get("quant")
    exp: dict[str, Any] = {"model_scale": 1.0, "attention_scale": 1.0,
                           "gemm_scale": 1.0, "basis": []}
    if skip:
        exp["model_scale"] *= (LAYERS - skip) / LAYERS
        exp["attention_scale"] *= (LAYERS - skip) / LAYERS
        exp["gemm_scale"] *= (LAYERS - skip) / LAYERS
        scale = (LAYERS - skip) / LAYERS
        exp["basis"].append(f"skip {skip}/{LAYERS} layers -> x{scale:.4f}")
    if window not in (None, "off", 0):
        ctx = REGIME_CONTEXT.get(regime, 512)
        kept = min(ctx, int(window) + WINDOW_SINKS)
        scale = kept / ctx
        exp["attention_scale"] *= scale
        exp["basis"].append(
            f"window {window}+{WINDOW_SINKS} of ctx~{ctx} -> attention x{scale:.4f}"
            + (" (nothing to trim)" if scale >= 1.0 else "")
        )
    if quant and quant != "target-matching":
        # w4a16: 4-bit weights against bf16. In a memory-bound decode the
        # linear layers' weight traffic falls ~4x; this is the optimistic
        # bound, and the gap to it is the kernel's own inefficiency.
        exp["gemm_scale"] *= 0.25
        exp["basis"].append("w4a16 weights -> gemm x0.25 (optimistic bound)")
    return exp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", default="R1")
    parser.add_argument(
        "--dir", type=Path, default=PHASE / "data/probe_nsys_levers"
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    records = load(args.dir, args.regime)
    if not records:
        raise SystemExit(f"no X31 records for regime {args.regime} in {args.dir}")
    accounts = {k: account(v) for k, v in records.items()}
    base = accounts.get("base")

    result: dict[str, Any] = {
        "record_type": "w98_nsys_accounting",
        "schema_version": 1,
        "regime": args.regime,
        "note": (
            "nsys inflates absolute time; ratios are the currency, not "
            "milliseconds"
        ),
        "accounts": accounts,
        "levers": {},
    }

    print(f"=== {args.regime}: draft-chain vertical account (ms per chain) ===")
    print(f"{'combo':10s} {'chain':>8s} {'model':>8s} {'orch':>7s} "
          f"{'window':>7s} {'unattr':>7s}  model share")
    for combo, acc in accounts.items():
        if acc["chain_ms"] is None:
            print(f"{combo:10s} {'--  (no draft chain: OFF arm)':>40s}")
            continue
        print(
            f"{combo:10s} {acc['chain_ms']:8.3f} {acc['model_ms']:8.3f} "
            f"{acc['orchestration_ms']:7.3f} {acc['window_rewrite_ms']:7.3f} "
            f"{acc['unattributed_ms']:7.3f}  {acc['model_share']:.3f}"
        )

    if base and base["chain_ms"]:
        print("\n=== horizontal: measured vs expected, against `base` ===")
        print(f"{'combo':10s} {'model x':>9s} {'expected':>9s} "
              f"{'overhead':>9s}   basis")
        for combo, acc in accounts.items():
            if combo in ("base", "off") or acc["chain_ms"] is None:
                continue
            cfg = records[combo]["config"]
            exp = expectations(combo, cfg, args.regime)
            measured = acc["model_ms"] / base["model_ms"]
            expected = exp["model_scale"]
            # The window acts on attention only, so its model-level expectation
            # needs the base's own attention share to be meaningful.
            base_classes = base["kernel_classes_ms"]
            attn = base_classes.get("attention", 0.0)
            gemm = base_classes.get("gemm", 0.0)
            gpu = attn + gemm
            if gpu:
                expected = (
                    attn * exp["attention_scale"] + gemm * exp["gemm_scale"]
                ) / gpu
                if exp["model_scale"] != 1.0:
                    expected *= exp["model_scale"] / 1.0
            overhead = measured - expected
            result["levers"][combo] = {
                "measured_model_scale": round(measured, 4),
                "expected_model_scale": round(expected, 4),
                "overhead": round(overhead, 4),
                "expectation_basis": exp["basis"],
                "kernel_classes_ms": acc["kernel_classes_ms"],
            }
            print(
                f"{combo:10s} {measured:9.4f} {expected:9.4f} {overhead:+9.4f}   "
                f"{'; '.join(exp['basis'])}"
            )

    out = args.out or (args.dir / f"accounting_{args.regime}.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
