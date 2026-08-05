#!/usr/bin/env python3
"""W6 item 1: the offline-only error floor (w6_design.md section 4).

Floor A: offline (C4, rho=1) argmax over {OFF, deployed w512, deployed
w2048}; scored by live S_dec. Floor B: unrestricted offline argmax over
all oracle configs; scored when measured, else UNSCORED.

Live S_dec provenance (episode-rejected medians):
  results_w2.md table (W2 sweep, notune, piecewise realization);
  llama R5 w2048 = W4c piecewise serving dec 61.2/45.2 = 1.354 (the W2
  cell was episode-contaminated, superseded -- results_w2.md F4).
  OFF = 1.0 (parked cost <=0.6%, folded into the noise band).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from round1_shortlist import DEPLOYED_Q, REGIME_CELL, cell_rf, rf_maps

LIVE = {  # (arch, regime) -> {action: live S_dec}
    ("dense", "R4"): {"w512": 0.976, "w2048": 0.971},
    ("dense", "R5"): {"w512": 1.308, "w2048": 1.272},
    ("dense", "R5cot"): {"w512": 1.682, "w2048": 1.649},
    ("dense", "R8"): {"w512": 1.051, "w2048": 1.003},
    ("dense", "R1"): {"w512": 1.108, "w2048": 1.104},
    ("dense", "R6"): {"w512": 1.025, "w2048": 1.032},
    ("llama", "R4"): {"w512": 1.065, "w2048": 1.027},
    ("llama", "R5"): {"w512": 1.009, "w2048": 1.354},
    ("llama", "R5cot"): {"w512": 1.637, "w2048": 1.597},
    ("llama", "R8"): {"w512": 0.999, "w2048": 0.996},
    ("llama", "R1"): {"w512": 0.745, "w2048": 1.116},
    ("llama", "R6"): {"w512": 0.989, "w2048": 0.809},
}
REGIMES = ["R4", "R5", "R5cot", "R8", "R1", "R6"]


def s_pred(m, k, rid):
    got = cell_rf(m, k, rid)
    if not got:
        return None
    r, f, _ = got
    return (1 + f * k) / (k * r + 1)


out = {}
for arch in ("dense", "llama"):
    maps = rf_maps(arch)
    dq = DEPLOYED_Q[arch]
    rows, lossesA = [], []
    for rid in REGIMES:
        live = dict(LIVE[(arch, rid)], OFF=1.0)
        best_live_a, best_live_v = max(live.items(), key=lambda x: x[1])
        # Floor A: restricted actions, offline K per action
        predA = {"OFF": 1.0}
        for w in ("512", "2048"):
            cfg = f"q-{dq}_w-{w}_s-b2"
            if cfg in maps:
                cands = [s_pred(maps[cfg], k, rid) for k in (2, 4)]
                cands = [c for c in cands if c]
                if cands:
                    predA[f"w{w}"] = max(cands)
        pickA = max(predA, key=predA.get)
        lossA = best_live_v - live[pickA]
        lossesA.append(lossA)
        # Floor B: unrestricted offline argmax
        bestB, bestB_v = "OFF", 1.0
        for cfg, m in maps.items():
            for k in (2, 4):
                v = s_pred(m, k, rid)
                if v and v > bestB_v:
                    bestB, bestB_v = f"{cfg}/K{k}", v
        scoredB = None
        for a in live:
            if a != "OFF" and f"q-{dq}_w-{a[1:]}_s-b2" in bestB:
                scoredB = best_live_v - live[a]
        if bestB == "OFF":
            scoredB = best_live_v - 1.0
        rows.append({
            "regime": rid, "live_best": f"{best_live_a}={best_live_v}",
            "offline_pick_A": f"{pickA} (pred {predA[pickA]:.3f})",
            "live_of_pick_A": live[pickA], "loss_A": round(lossA, 4),
            "offline_pick_B": f"{bestB} (pred {bestB_v:.3f})",
            "loss_B": (round(scoredB, 4) if scoredB is not None
                       else "UNSCORED"),
        })
        print(f"[{arch}] {rid:6s} live-best {best_live_a}="
              f"{best_live_v:.3f} | A-pick {pickA:6s} "
              f"(pred {predA[pickA]:.3f}, live {live[pickA]:.3f}) "
              f"loss {lossA:+.3f} | B-pick {bestB} "
              f"loss {rows[-1]['loss_B']}")
    out[arch] = {"cells": rows,
                 "floorA_mean": round(sum(lossesA) / len(lossesA), 4),
                 "floorA_worst": round(max(lossesA), 4)}
    print(f"[{arch}] Floor A mean {out[arch]['floorA_mean']:+.4f}  "
          f"worst {out[arch]['floorA_worst']:+.4f}\n")

p = Path(__file__).parents[1] / "data" / "w6"
p.mkdir(parents=True, exist_ok=True)
(p / "error_floor.json").write_text(json.dumps(out, indent=1))
print("saved ->", p / "error_floor.json")
