#!/usr/bin/env python3
"""Map the accept-length-per-GB frontier of the comm-free draft (single-node baseline).

Combines measured acceptance with the per-device resident-expert memory it costs, to
find the Pareto frontier we must beat. Inputs are READ-ONLY from prior phases:
  - Phase 25 Exp2 (qwen3_fp4_local_curve.json): NVFP4 local-routing beta vs cache C.
  - Phase 22 full-coverage anchors (FP8 ~0.95) hard-coded with citation.
Accept length is E[accepted] at k=4 (geometric from one-step beta; validated against the
Phase 24 B1 real k-sweep, which measured 2.74 at 0.5E vs 2.64 geometric here).
"""

from __future__ import annotations

import json
from pathlib import Path

# Qwen3-30B-A3B expert geometry
HIDDEN, MOE_INTER, N_LAYERS = 2048, 768, 48
PARAMS_PER_EXPERT = 3 * HIDDEN * MOE_INTER  # gate+up+down = 4.72M
N_EXPERTS = 128
BITS = {"nvfp4": 4.5, "fp8": 8.0, "bf16": 16.0}  # incl. NVFP4 block-scale overhead

PH25 = Path(__file__).resolve().parents[1] / "25_local_routing_strategy/data/qwen3_fp4_local_curve.json"


def mem_gb(c, bits):
    return c * N_LAYERS * PARAMS_PER_EXPERT * bits / 8 / 1e9


def eacc(b, k=4):
    return b * (1 - b ** k) / (1 - b)


def main():
    curve = json.loads(PH25.read_text())["results"]["nvfp4"]
    betaC = {r["C"]: r["sampled"] for r in curve}

    pts = []
    for c in sorted(betaC):
        b = betaC[c]
        pts.append({"config": f"FP4 C={c}", "C": c, "frac": c / N_EXPERTS,
                    "mem_gb": round(mem_gb(c, BITS["nvfp4"]), 1),
                    "beta": round(b, 3), "acc_len": round(eacc(b), 2)})
    # full-coverage anchors at higher precision (Phase 22)
    pts.append({"config": "FP8 full", "C": N_EXPERTS, "frac": 1.0,
                "mem_gb": round(mem_gb(N_EXPERTS, BITS["fp8"]), 1),
                "beta": 0.95, "acc_len": round(eacc(0.95), 2)})
    pts.append({"config": "bf16 full", "C": N_EXPERTS, "frac": 1.0,
                "mem_gb": round(mem_gb(N_EXPERTS, BITS["bf16"]), 1),
                "beta": 1.0, "acc_len": 4.0})

    for p in pts:
        p["acc_per_gb"] = round(p["acc_len"] / p["mem_gb"], 3)

    # marginal returns along the FP4 sweep
    fp4 = [p for p in pts if p["config"].startswith("FP4")]
    prev = {"mem_gb": 0.0, "acc_len": 0.0}
    marg = []
    for p in fp4:
        dm, da = p["mem_gb"] - prev["mem_gb"], p["acc_len"] - prev["acc_len"]
        marg.append({"to": p["config"], "mem_gb": p["mem_gb"],
                     "d_acc": round(da, 2), "d_mem": round(dm, 1),
                     "marg_acc_per_gb": round(da / dm, 3) if dm else None})
        prev = p

    print(f"{'config':>12} {'frac':>5} {'memGB':>6} {'beta':>5} {'acc_len':>7} {'acc/GB':>7}")
    for p in pts:
        print(f"{p['config']:>12} {p['frac']:>5.2f} {p['mem_gb']:>6.1f} "
              f"{p['beta']:>5.2f} {p['acc_len']:>7.2f} {p['acc_per_gb']:>7.3f}")
    print("\nmarginal accept-length per extra GB (FP4 sweep):")
    for m in marg:
        print(f"  ->{m['to']:>9} (+{m['d_mem']:>4.1f}GB): {m['marg_acc_per_gb']} acc/GB")

    out = Path(__file__).resolve().parent / "data/frontier.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"points": pts, "marginal_fp4": marg}, indent=2))
    print(f"[saved] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
