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

from probe_w98_nsys_levers import classify_kernel

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
            k: round(v / 1e6, 4) for k, v in kernel_classes(record).items()
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
        # Removing layers scales EVERY class, so it is expressed per class and
        # NOT also in `model_scale` -- doing both squares the ratio, which is
        # what made skip4's bound read 0.796 instead of 0.889.
        scale = (LAYERS - skip) / LAYERS
        exp["attention_scale"] *= scale
        exp["gemm_scale"] *= scale
        exp["other_scale"] = exp.get("other_scale", 1.0) * scale
        exp["elementwise_scale"] = exp.get("elementwise_scale", 1.0) * scale
        exp["basis"].append(f"skip {skip}/{LAYERS} layers -> all classes x{scale:.4f}")
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


def kernel_classes(record: Mapping[str, Any]) -> dict[str, float]:
    """Recompute kernel classes from the raw rows, in nanoseconds.

    Deliberately not the `kernel_classes` stored at extraction time: the
    classifier has been corrected since some records were written (split-K
    GEMM epilogues were landing in `elementwise`), and recomputing here makes
    the fix retroactive instead of requiring a re-run.
    """
    rows = ((record.get("nsys") or {}).get("kernels") or {}).get("rows") or []
    out: dict[str, float] = {}
    for row in rows:
        name = row.get("Name") or ""
        raw = row.get("Total Time (ns)")
        if raw is None:
            continue
        try:
            value = float(str(raw).replace(",", ""))
        except (TypeError, ValueError):
            continue
        bucket = classify_kernel(name)
        out[bucket] = out.get(bucket, 0.0) + value
    return out


def draft_kernel_ms(
    record: Mapping[str, Any], off: Mapping[str, Any], steps: int = 40
) -> dict[str, float]:
    """Per-step kernel time attributable to the DRAFT, by class.

    An armed step runs K+1 draft forwards plus one verify; the OFF arm runs
    that same verify and nothing else. Subtracting OFF therefore isolates the
    draft, which is the only thing a lever changes -- comparing whole-capture
    totals instead would dilute every ratio by the unchanged verify.
    """
    armed = kernel_classes(record)
    base_off = kernel_classes(off)
    out: dict[str, float] = {}
    for key in set(armed) | set(base_off):
        value = (armed.get(key, 0.0) - base_off.get(key, 0.0)) / 1e6 / steps
        out[key] = round(value, 4)
    out["total"] = round(sum(out.values()), 4)
    return out


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

    off_rec = records.get("off")
    if off_rec and base:
        print("\n=== draft-only GPU kernel time per step (armed minus OFF), ms ===")
        classes = ("attention", "gemm", "elementwise", "memory", "other", "total")
        print(f"{'combo':10s}" + "".join(f"{c:>13s}" for c in classes))
        draft_kernels: dict[str, dict[str, float]] = {}
        for combo, record in records.items():
            if combo == "off" or not accounts[combo]["chain_ms"]:
                continue
            split = draft_kernel_ms(record, off_rec)
            draft_kernels[combo] = split
            print(
                f"{combo:10s}"
                + "".join(f"{split.get(c, 0.0):13.3f}" for c in classes)
            )
        result["draft_kernel_ms"] = draft_kernels
        if "base" in draft_kernels:
            print(f"\n{'combo':10s}" + "".join(f"{c + ' x':>13s}" for c in classes))
            for combo, split in draft_kernels.items():
                if combo == "base":
                    continue
                ref = draft_kernels["base"]
                print(
                    f"{combo:10s}"
                    + "".join(
                        f"{(split.get(c, 0.0) / ref[c]):13.4f}"
                        if ref.get(c)
                        else f"{'-':>13s}"
                        for c in classes
                    )
                )

    if base and base["chain_ms"] and off_rec and "base" in result.get(
        "draft_kernel_ms", {}
    ):
        ref = result["draft_kernel_ms"]["base"]
        print("\n=== where the lever's promise goes ===")
        print("  bound   : what the lever COULD give if its kernels hit the")
        print("            physical limit (layers removed / KV not read /")
        print("            weight bytes not loaded)")
        print("  kernels : what the kernels actually delivered")
        print("  model   : what the draft FORWARD actually cost")
        print("  so: kernel gap = kernels - bound (kernel efficiency), and")
        print("      fixed gap  = model - kernels (per-forward cost that does")
        print("                   not scale with the work removed)")
        print()
        print(f"{'combo':10s} {'bound':>8s} {'kernels':>8s} {'model':>8s} "
              f"{'kernel gap':>11s} {'fixed gap':>10s}   basis")
        for combo, acc in accounts.items():
            if combo in ("base", "off") or acc["chain_ms"] is None:
                continue
            split = result["draft_kernel_ms"].get(combo)
            if not split:
                continue
            cfg = records[combo]["config"]
            exp = expectations(combo, cfg, args.regime)
            scales = {
                "attention": exp["attention_scale"],
                "gemm": exp["gemm_scale"],
                "elementwise": exp.get("elementwise_scale", 1.0),
                "other": exp.get("other_scale", 1.0),
            }
            total = ref.get("total", 0.0)
            bound = (
                sum(
                    ref.get(k, 0.0) * scales.get(k, 1.0)
                    for k in ("attention", "gemm", "elementwise", "other")
                )
                / total
                * exp["model_scale"]
                if total
                else None
            )
            kernels = split["total"] / total if total else None
            model = acc["model_ms"] / base["model_ms"]
            result["levers"][combo] = {
                "bound_scale": round(bound, 4) if bound else None,
                "kernel_scale": round(kernels, 4) if kernels else None,
                "model_scale": round(model, 4),
                "kernel_gap": round(kernels - bound, 4) if bound and kernels else None,
                "fixed_gap": round(model - kernels, 4) if kernels else None,
                "expectation_basis": exp["basis"],
                "draft_kernel_ms": split,
            }
            print(
                f"{combo:10s} {bound:8.4f} {kernels:8.4f} {model:8.4f} "
                f"{kernels - bound:+11.4f} {model - kernels:+10.4f}   "
                f"{'; '.join(exp['basis'])}"
            )

    out = args.out or (args.dir / f"accounting_{args.regime}.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
