#!/usr/bin/env python3
"""Strategy map v2: 76's composition with MEASURED beta (77) on both axes.

Differences vs v1 (76/scripts/e2_strategy_map.py):
  - beta comes from data/beta.csv, per (model, arm, CTX) -- ctx-specific, not
    a borrowed constant. Greedy beta (E1/E3 ground truths were greedy).
  - layer-skip arms become REAL selection arms (beta measured; v1 could only
    do break-even analysis).
  - local-route uses lr25 (the EP4 quarter-shard actually measured by E1's
    m_localroute cost arm) instead of P24's half-cache 0.85 -- honest pairing.
  - kvq stays a REFERENCE column (no draft-only pool exists in the harness),
    but now with its measured draft-only beta.

Arm pairing (76 cost arm -> 77 beta arm):
  d_w4marlin/d_w4machete -> q_int4     d_fp8w8a8 -> q_fp8
  m_fp8marlin/m_fp8block -> q_fp8      *_win -> win512
  *_skip50 -> skip50, d_skip25 -> skip25
  m_localroute -> lr25                 *_kvq -> kvq_fp8 (reference)

Emits the v2 map + a cell-by-cell diff vs v1. Writes data/strategy_map_v2.md.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

PHASE = Path(__file__).resolve().parent.parent
REPO = PHASE.parent.parent
P76 = REPO / "research/76_lever_latency_sweep"
sys.path.insert(0, str(P76 / "scripts"))
from e2_strategy_map import BETA as BETA_V1, best_speedup  # noqa: E402

PAIR = {
    "d_w4marlin": "q_int4", "d_w4machete": "q_int4", "d_fp8w8a8": "q_fp8",
    "d_win": "win512", "d_skip50": "skip50", "d_skip25": "skip25",
    "d_kvq": "kvq_fp8",
    "m_fp8marlin": "q_fp8", "m_fp8block": "q_fp8", "m_win": "win512",
    "m_localroute": "lr25", "m_skip50": "skip50", "m_kvq": "kvq_fp8",
}
REFERENCE = {"d_kvq", "m_kvq"}          # not draft-only implementable today
MODEL_OF = {"dense": "dense", "moe": "moe"}
CTX_OF = {2: 2048, 16: 16384, 32: 32768}


def load_beta():
    b = {}
    for r in csv.DictReader((PHASE / "data/beta.csv").open()):
        b[(r["model"], r["arm"], int(r["ctx"]))] = float(r["beta_greedy"])
    return b


def load_R():
    R = defaultdict(dict)
    for row in csv.DictReader((P76 / "data/e1/summary.csv").open()):
        if int(row.get("overpool", 0)):
            continue
        R[row["group"]][(row["arm"], (int(row["batch"]), int(row["ctx"])))] = float(row["ratio"])
    return R


def main() -> int:
    beta = load_beta()
    R = load_R()
    out = ["# Strategy map v2 — measured R × measured β (77)\n"]

    for group in ("dense", "moe"):
        cells = sorted({c for (_, c) in R[group]})
        arms = sorted({a for (a, _) in R[group] if a in PAIR})
        sel = [a for a in arms if a not in REFERENCE]
        out.append(f"\n## {group}\n")
        hdr = ["cell"] + sel + ["WINNER v2", "v1 winner", "Δ"]
        out.append("| " + " | ".join(hdr) + " |")
        out.append("|" + "|".join("---" for _ in hdr) + "|")
        for cell in cells:
            b_ctx, c_k = cell
            row, best2, best1 = [], (1.0, None, 0), (1.0, None, 0)
            for a in sel:
                r = R[group].get((a, cell))
                bmeas = beta.get((MODEL_OF[group], PAIR[a], CTX_OF[c_k]))
                if r is None or bmeas is None:
                    row.append("-")
                    continue
                s2, g2 = best_speedup(min(bmeas, 0.995), r)
                row.append(f"{s2:.2f}")
                if s2 > best2[0]:
                    best2 = (s2, a, g2)
                if a in BETA_V1:                       # v1 comparison
                    s1, g1 = best_speedup(BETA_V1[a][0], r)
                    if s1 > best1[0]:
                        best1 = (s1, a, g1)
            def w(b):
                return f"{b[1].split('_', 1)[1]} {b[0]:.2f}× (γ{b[2]})" if b[1] else "OFF"
            delta = "" if (best2[1] == best1[1]) else "**WINNER CHANGED**"
            if best2[1] == best1[1] and abs(best2[0] - best1[0]) > 0.08:
                delta = f"Δ{best2[0] - best1[0]:+.2f}×"
            out.append(f"| b{b_ctx}/{c_k}k | " + " | ".join(row) +
                       f" | **{w(best2)}** | {w(best1)} | {delta} |")

        # reference row: measured draft-only kvq
        refarm = "d_kvq" if group == "dense" else "m_kvq"
        vals = []
        for cell in cells:
            r = R[group].get((refarm, cell))
            bm = beta.get((MODEL_OF[group], "kvq_fp8", CTX_OF[cell[1]]))
            vals.append(f"{best_speedup(min(bm, 0.995), r)[0]:.2f}" if r and bm else "-")
        out.append(f"\n_reference (not implementable draft-only today): kvq_fp8 "
                   f"composed = {', '.join(vals)} across cells_")

    out.append("\n\n**Legend**: composed speedup* = max_γ τ_β(γ)/(γ·R+1); β "
               "measured (77, ctx-specific, greedy, β capped 0.995 for the "
               "geometric model); R measured (76 E1). v1 = borrowed-β map "
               "(76 E2). MLA awaits its cost tier.")
    text = "\n".join(out)
    (PHASE / "data/strategy_map_v2.md").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
