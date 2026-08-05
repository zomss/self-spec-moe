#!/usr/bin/env python3
"""Score E3 — the MLA/MoE gate arms. Tests P7.

Three arms per arch on the same batch sweep and datasets:

  off     AR anchor (no spec configured at all)
  uncond  spec ALWAYS on at the map's best static config -- what a deployment
          without a gate ships
  gated   same config + the compiled C2 policy (K/OFF per step)

P7 as registered: "on b1/b8 real-data regimes the gate keeps aggregate
S >= 0.98; at b32/b64 it arms and beats AR."

**P7's baseline is restated by E0's measurement.** E0 found the PARKED spec
engine is not free: it ran 2.5% below AR at llama b16 and ~0% at b8. So a
gated arm cannot reach S = 1.0 wherever parking costs anything, and the
honest ceiling is the parked engine, not AR. Two quantities are therefore
reported and kept distinct:

  gate_value  = gated / uncond   -- what the gate is WORTH (the claim)
  gate_ceiling_gap = 1 - gated   -- how far the gated arm still sits below AR,
                                    which at low batch is mostly parked-engine
                                    overhead, NOT a decision error

Reporting only the second would blame the policy for an engine cost.
"""
import json
import statistics as st
import sys
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
DATA = PHASE / "data"
ARMS = ["off", "uncond", "gated"]
LOW_BATCH = (1, 8)          # where the map says MLA/MoE lose
HIGH_BATCH = (32, 64)       # where C1 Stage B found their only wins


def load(arch, seed):
    out = {}
    for a in ARMS:
        f = DATA / f"e3_{arch}_{a}_s{seed}.json"
        if f.exists():
            d = json.loads(f.read_text())
            out[a] = {(c["rid"], c["batch"]): c
                      for c in d["cells"] if "toks" in c}
    return out


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    results = []
    for arch in ("mla", "moe"):
        arms = load(arch, seed)
        if "off" not in arms:
            print(f"[{arch}] no AR anchor yet -- skipped")
            continue
        keys = sorted(set(arms["off"]) &
                      set.intersection(*[set(arms[a]) for a in arms]))
        if not keys:
            print(f"[{arch}] no common cells yet")
            continue
        print(f"\n===== {arch} seed={seed} "
              f"(arms: {', '.join(sorted(arms))}) =====")
        print(f"  {'cell':14s} {'AR tok/s':>9s} {'uncond':>8s} {'gated':>8s}"
              f" {'gate val':>9s}  {'acc(unc)':>8s} {'acc(gat)':>8s}")
        rows = []
        for k in keys:
            ar = arms["off"][k]["toks"]
            u = arms["uncond"][k]["toks"] / ar if "uncond" in arms else None
            g = arms["gated"][k]["toks"] / ar if "gated" in arms else None
            gv = (g / u - 1) * 100 if (u and g) else None
            rows.append({"rid": k[0], "batch": k[1], "ar_toks": ar,
                         "S_uncond": round(u, 4) if u else None,
                         "S_gated": round(g, 4) if g else None,
                         "gate_value_pct": round(gv, 2) if gv is not None else None,
                         "accept_uncond": arms.get("uncond", {}).get(k, {}).get("accept"),
                         "accept_gated": arms.get("gated", {}).get(k, {}).get("accept")})
            print(f"  {k[0]+'/b'+str(k[1]):14s} {ar:9.1f} "
                  f"{u if u else float('nan'):8.4f} {g if g else float('nan'):8.4f} "
                  f"{gv if gv is not None else float('nan'):+8.2f}%  "
                  f"{str(rows[-1]['accept_uncond']):>8s} "
                  f"{str(rows[-1]['accept_gated']):>8s}")

        def agg(sel):
            u = [r["S_uncond"] for r in rows
                 if r["batch"] in sel and r["S_uncond"]]
            g = [r["S_gated"] for r in rows
                 if r["batch"] in sel and r["S_gated"]]
            return (st.mean(u) if u else None, st.mean(g) if g else None)

        summary = {"arch": arch, "seed": seed, "cells": rows}
        for label, sel in (("low_batch_b1_b8", LOW_BATCH),
                           ("high_batch_b32_b64", HIGH_BATCH)):
            u, g = agg(sel)
            if u is None or g is None:
                continue
            summary[label] = {
                "S_uncond": round(u, 4), "S_gated": round(g, 4),
                "gate_value_pct": round((g / u - 1) * 100, 2),
                "gap_below_AR_pct": round((1 - g) * 100, 2),
            }
            s = summary[label]
            print(f"  [{label:18s}] uncond {u:.4f} -> gated {g:.4f}   "
                  f"gate worth {s['gate_value_pct']:+.2f}%   "
                  f"gated sits {s['gap_below_AR_pct']:+.2f}% below AR")

        lo = summary.get("low_batch_b1_b8")
        hi = summary.get("high_batch_b32_b64")
        if lo and hi:
            summary["P7_low_batch_ge_098"] = bool(lo["S_gated"] >= 0.98)
            summary["P7_high_batch_beats_AR"] = bool(hi["S_gated"] > 1.0)
            print(f"  P7 low-batch gated >= 0.98: "
                  f"{'PASS' if summary['P7_low_batch_ge_098'] else 'FAIL'}"
                  f" ({lo['S_gated']:.4f})   "
                  f"P7 high-batch beats AR: "
                  f"{'PASS' if summary['P7_high_batch_beats_AR'] else 'FAIL'}"
                  f" ({hi['S_gated']:.4f})")
            print("  NOTE: any low-batch shortfall is parked-engine overhead "
                  "(E0: 2.5% at b16, ~0% at b8) plus probe churn, not "
                  "necessarily a decision error -- see gate_value.")
        results.append(summary)
    if results:
        (DATA / "e3_gate.json").write_text(json.dumps(
            {"seed": seed, "results": results}, indent=1))
        print(f"\nwrote {DATA}/e3_gate.json")


if __name__ == "__main__":
    main()
