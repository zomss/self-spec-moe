#!/usr/bin/env python3
"""W14 item C — analytic leverage router.

Computes byte shares of the draft step at an execution state and routes
MEASUREMENT BUDGET accordingly. It never eliminates a configuration.

    quant potential  -> weight-byte share
    window / kv-quant-> KV-byte share
    skip count k     -> ~k/L of layer work (both terms)

Threshold 15% (pre-declared, not tuned):
  >= 15% : measure a full cost curve
  <  15% : retain, use a deliberately non-pruning wide interval or defer
           cost to Round 2; at most an anchor spot-check.

CAVEAT (repeat wherever this is cited): the W13 D/T spread used as a
validation target comes from the SYNC-BRACKETED profiler, i.e. the
biased instrument. That is acceptable here because C routes budget and
cannot eliminate -- but C's agreement is NOT evidence the cost model is
accurate. Leverage predicts the DRAFT-COMPONENT spread; q_true is a
whole-step quantity. Different tests.
"""
import argparse
import json
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
THRESHOLD = 0.15          # pre-declared profiling-priority threshold

# arch -> layers, kv heads, head_dim, non-expert(dense) params, bytes/param
ARCH = {
    "qwen3-8b": dict(L=36, kv_heads=8, head_dim=128, params=8.2e9,
                     draft_bpp=0.5625, target_bpp=2.0),
}


def shares(arch, n_active, total_kv_tokens, draft_bpp=None):
    """Weight- and KV-byte shares of ONE draft decode step."""
    a = ARCH[arch]
    bpp = a["draft_bpp"] if draft_bpp is None else draft_bpp
    w_bytes = a["params"] * bpp
    kv_bytes = total_kv_tokens * a["L"] * a["kv_heads"] * a["head_dim"] * 2 * 2
    tot = w_bytes + kv_bytes
    return w_bytes / tot, kv_bytes / tot, w_bytes, kv_bytes


def route(arch, n_active, total_kv_tokens, skip_count=0, draft_bpp=None):
    ws, ks, wb, kb = shares(arch, n_active, total_kv_tokens, draft_bpp)
    L = ARCH[arch]["L"]
    lev = {"quant": ws, "window": ks, "kv_quant": ks,
           "skip": (skip_count / L) if skip_count else 0.0}
    return {
        "n_active": n_active, "total_kv": total_kv_tokens,
        "weight_share": round(ws, 4), "kv_share": round(ks, 4),
        "weight_GB": round(wb / 1e9, 3), "kv_GB": round(kb / 1e9, 3),
        "leverage": {k: round(v, 4) for k, v in lev.items()},
        "measure_full_curve": sorted(k for k, v in lev.items()
                                     if v >= THRESHOLD),
        "defer_or_spotcheck": sorted(k for k, v in lev.items()
                                     if 0 < v < THRESHOLD),
    }


def validate_against_w13():
    """Directional check vs banked W13 draft D/T spread (biased source --
    see module caveat). Leverage must ORDER the cells the same way."""
    p = PHASE / "data" / "w13" / "w13_scored.json"
    if not p.exists():
        return None
    rows = json.load(open(p))
    ctx = {"R1": 1107, "R5": 14550, "R5cot": 17100}
    by = {}
    for r in rows:
        by.setdefault((r["rid"], r["batch"]), []).append(r["D_over_T"])
    out = []
    for (rid, b), v in by.items():
        if len(v) < 2:
            continue
        spread = (max(v) - min(v)) / min(v)
        kv_share = route("qwen3-8b", b, b * ctx[rid])["kv_share"]
        out.append({"cell": f"{rid} b{b}", "kv_share": kv_share,
                    "measured_DT_spread": round(spread, 4),
                    "routed": "measure" if kv_share >= THRESHOLD else "defer"})
    out.sort(key=lambda r: r["kv_share"])
    # rank correlation between predicted leverage and measured spread
    n = len(out)
    rk_l = {r["cell"]: i for i, r in enumerate(
        sorted(out, key=lambda r: r["kv_share"]))}
    rk_s = {r["cell"]: i for i, r in enumerate(
        sorted(out, key=lambda r: r["measured_DT_spread"]))}
    d2 = sum((rk_l[c] - rk_s[c]) ** 2 for c in rk_l)
    rho = 1 - 6 * d2 / (n * (n * n - 1)) if n > 1 else float("nan")
    return out, rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="qwen3-8b")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    print(f"W14/C leverage router — threshold {THRESHOLD:.0%} "
          f"(routes measurement only; never eliminates)\n")
    cells = [(1, 1107), (8, 1107), (1, 14550), (8, 14550),
             (1, 17100), (8, 17100), (32, 2000), (32, 14550)]
    print(f"{'n_act':>6}{'total_KV':>10}{'weight%':>9}{'KV%':>7}"
          f"{'  measure-full-curve':<26}{'defer/spot-check'}")
    rows = []
    for b, ctx in cells:
        r = route(args.arch, b, b * ctx)
        rows.append(r)
        print(f"{b:>6}{b*ctx:>10}{100*r['weight_share']:>8.1f}%"
              f"{100*r['kv_share']:>6.1f}%  "
              f"{','.join(r['measure_full_curve']):<24}"
              f"{','.join(r['defer_or_spotcheck'])}")
    print("\nskip leverage is k/L, independent of state: "
          + ", ".join(f"k={k} -> {100*k/ARCH[args.arch]['L']:.0f}%"
                      for k in (2, 4, 8)))
    print(f"  -> skip clears {THRESHOLD:.0%} only at k >= "
          f"{int(THRESHOLD*ARCH[args.arch]['L'])+1} layers "
          f"(k=2 is 5.6%: defer; k=8 is 22%: measure)")

    out = {"threshold": THRESHOLD, "arch": args.arch, "cells": rows}
    if args.validate:
        v = validate_against_w13()
        if v:
            tbl, rho = v
            print("\nDirectional validation vs banked W13 draft D/T spread")
            print("  (BIASED SOURCE -- routes budget only, not evidence the")
            print("   cost model is accurate; see module docstring)")
            print(f"\n{'cell':<11}{'KV share':>10}{'D/T spread':>12}{'routed':>9}")
            for r in tbl:
                print(f"{r['cell']:<11}{100*r['kv_share']:>9.1f}%"
                      f"{100*r['measured_DT_spread']:>11.1f}%{r['routed']:>9}")
            print(f"\n  Spearman rho(leverage, measured spread) = {rho:+.2f}")
            # The SAFETY property, which is what actually matters: C must
            # never DEFER a cell that has real spread (under-routing loses
            # signal). Over-routing only wastes a spot-check.
            under = [r for r in tbl if r["routed"] == "defer"
                     and r["measured_DT_spread"] > 0.15]
            over = [r for r in tbl if r["routed"] == "measure"
                    and r["measured_DT_spread"] < 0.15]
            print(f"  UNDER-routed (deferred despite real spread): "
                  f"{len(under)} {[r['cell'] for r in under]}  <- dangerous")
            print(f"  over-routed (measured despite small spread): "
                  f"{len(over)} {[r['cell'] for r in over]}  <- merely wasteful")
            print("\n  NOTE: byte share is an UPPER BOUND on achievable")
            print("  leverage -- a byte reduction converts to time only if the")
            print("  step is bandwidth-bound. At b1 the draft is dispatch-bound")
            print("  (2-15x off roofline, W8), so R5/R5cot b1 read 28x less KV")
            print("  under w512 yet show ~2% D/T spread. The bound is loose")
            print("  exactly there, which is why rho is only moderate. Since C")
            print("  routes budget and never eliminates, erring toward")
            print("  over-routing is the safe direction.")
            out["validation"] = {"rows": tbl, "spearman_rho": rho,
                                 "under_routed": [r["cell"] for r in under],
                                 "over_routed": [r["cell"] for r in over]}
    d = PHASE / "data" / "w14"
    d.mkdir(parents=True, exist_ok=True)
    (d / "w14c_leverage.json").write_text(json.dumps(out, indent=1))
    print("\nsaved ->", d / "w14c_leverage.json")


if __name__ == "__main__":
    main()
