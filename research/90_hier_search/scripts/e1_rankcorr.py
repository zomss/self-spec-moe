#!/usr/bin/env python3
"""Phase 90 E1: validate proxy scores against the COMMITTED beta record.

P1a (skip proxy): predicted set-damage = sum of per-layer importance
     over the set; Spearman vs measured beta across the greedy-scan
     arms (83 e1_dense log: 8B; 86 beta32_greedy log: 32B).
     Also: does bottom-k importance reproduce the greedy sets?
P1b (window proxy): beyond-W locality mass vs the measured window
     record -- R4 step function (win512 2.70 / win2048 2.67 /
     win8192 4.06 accept) via mass-in-band on CNN/DM refs; c4-vs-cnndm
     mass ratio vs the ondist/task beta split.
P2  (concentration): fraction of (layer,head) carrying 70% of
     beyond-512 mass on CNN/DM.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr

D = Path(__file__).resolve().parents[1] / "data"
REPO = Path(__file__).resolve().parents[3]


def load_importance(prefix):
    imp = {}
    for line in open(D / f"{prefix}_importance.csv").read().splitlines()[1:]:
        m, r, layer, v = line.rsplit(",", 3)
        imp[int(layer)] = float(v)
    return imp


def load_locality(prefix):
    loc = defaultdict(dict)  # W -> (layer, head) -> mass
    for line in open(D / f"{prefix}_locality.csv").read().splitlines()[1:]:
        m, r, layer, head, W, v = line.rsplit(",", 5)
        loc[int(W)][(int(layer), int(head))] = float(v)
    return loc


def arms_from_log(path, pat):
    per = defaultdict(list)
    for m in re.finditer(pat, open(path, errors="ignore").read()):
        layers = tuple(int(x) for x in m.group(1).split("-"))
        per[layers].append(float(m.group(2)))
    return {k: sum(v) / len(v) for k, v in per.items()}


def skip_validation(tag, imp, arms, greedy_sets):
    preds, betas = [], []
    for layers, beta in arms.items():
        preds.append(sum(imp[la] for la in layers))
        betas.append(beta)
    rho, p = spearmanr(preds, [-b for b in betas])
    order = sorted(imp, key=imp.get)
    res = {"n_arms": len(arms), "spearman_damage_vs_negbeta": round(rho, 3),
           "p": float(f"{p:.2e}")}
    hits = {}
    for k, gset in greedy_sets.items():
        bottom = set(order[:k])
        hits[f"greedy{k}"] = {
            "greedy": sorted(gset), "proxy_bottomk": sorted(bottom),
            "overlap": len(bottom & set(gset))}
    res["greedy_order"] = hits
    print(f"[P1a {tag}] arms={len(arms)} spearman={rho:.3f} (p={p:.1e})")
    for k, h in hits.items():
        print(f"          {k}: greedy={h['greedy']} "
              f"proxy={h['proxy_bottomk']} overlap={h['overlap']}")
    return res


def main():
    out = {}

    # ---- P1a skip: 32B ----
    imp32 = load_importance("q332b_c4")
    arms32 = arms_from_log(
        REPO / "research/86_scale_headtohead/logs/beta32_greedy.log",
        r"igs_([0-9-]+) p\d+: beta=(0\.\d+)")
    g32 = {1: [4], 2: [4, 7], 3: [4, 7, 16], 4: [2, 4, 7, 16]}
    out["p1a_32b"] = skip_validation("32B", imp32, arms32, g32)

    # ---- P1a skip: 8B ----
    imp8 = load_importance("q38b_c4")
    arms8 = arms_from_log(
        REPO / "research/83_profiled_levers/logs/e1_dense.log",
        r"ig_([0-9-]+) p\d+: beta=(0\.\d+)")
    g8 = {1: [5], 3: [4, 5, 6], 5: [3, 4, 5, 6, 12],
          7: [3, 4, 5, 6, 7, 11, 12]}
    out["p1a_8b"] = skip_validation("8B", imp8, arms8, g8)

    # ---- P1b window: R4 step function ----
    loc8k = load_locality("q38b_cnndm8k")
    tot = {W: sum(loc8k[W].values()) / max(len(loc8k[W]), 1)
           for W in sorted(loc8k)}
    # mass the win-W draft cannot see, mean over (layer, head):
    print(f"[P1b] mean beyond-W mass (cnndm 8k): "
          + ", ".join(f"W{W}={v:.4f}" for W, v in tot.items()))
    # step check: does mass barely drop 512->2048 (win2048 useless)
    # while dropping a lot 2048->beyond-article (win8192 works)?
    if 512 in tot and 2048 in tot:
        drop_512_2048 = (tot[512] - tot[2048]) / max(tot[512], 1e-9)
        out["p1b_step"] = {
            "beyond_mass": {str(k): round(v, 4) for k, v in tot.items()},
            "frac_of_beyond512_captured_by_2048": round(drop_512_2048, 3),
        }
        print(f"[P1b] widening 512->2048 captures only "
              f"{drop_512_2048*100:.1f}% of the missing mass "
              f"(measured accept: 2.70 -> 2.67, i.e. ~0)")

    # c4 vs cnndm beyond-512 (task dependence)
    locc4 = load_locality("q38b_c4")
    loccn = load_locality("q38b_cnndm")
    m_c4 = sum(locc4[512].values()) / max(len(locc4[512]), 1)
    m_cn = sum(loccn[512].values()) / max(len(loccn[512]), 1)
    out["p1b_task"] = {"beyond512_c4": round(m_c4, 4),
                      "beyond512_cnndm": round(m_cn, 4)}
    print(f"[P1b] beyond-512 mass: c4={m_c4:.4f} cnndm={m_cn:.4f} "
          f"(measured: win512 beta .975 ondist vs accept-starved on cnndm)")

    # ---- P2 concentration ----
    pairs = sorted(loc8k[512].items(), key=lambda kv: -kv[1])
    total = sum(v for _, v in pairs)
    run, k70 = 0.0, 0
    for i, (_, v) in enumerate(pairs):
        run += v
        if run >= 0.7 * total:
            k70 = i + 1
            break
    frac = k70 / len(pairs)
    out["p2_concentration"] = {
        "pairs_total": len(pairs), "pairs_for_70pct": k70,
        "fraction": round(frac, 3),
        "verdict": "PASS" if frac <= 0.25 else "FAIL"}
    print(f"[P2] {k70}/{len(pairs)} (layer,head) pairs = "
          f"{frac*100:.1f}% carry 70% of beyond-512 mass -> "
          f"{'PASS' if frac <= 0.25 else 'FAIL'} (gate <=25%)")

    (D / "rankcorr.json").write_text(json.dumps(out, indent=1))
    print("[e1] saved ->", D / "rankcorr.json")


if __name__ == "__main__":
    main()
