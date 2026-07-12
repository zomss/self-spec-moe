#!/usr/bin/env python3
"""V4: the selector -- strategy map v3 across THREE architectures.

Composes measured R (76 dataset, now including the MLA confirmation tier)
with measured beta (77, ctx-specific, per-architecture) via the P75-validated
formula. This is the closed loop the project set out to build: given
(architecture, batch, ctx), pick the best lever + depth, including "OFF".

MLA notes: local-route has measured beta (0.88-0.99, the shared-expert
anchor) but NO cost arm at NVLink (its regime is comm-bound fabrics --
deferred); kvq stays a reference column (no draft-only pool implemented).
Writes data/strategy_map_v3.md.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

PHASE = Path(__file__).resolve().parent.parent
P76 = PHASE.parent / "76_lever_latency_sweep"
P77 = PHASE.parent / "77_acceptance_map"
sys.path.insert(0, str(P76 / "scripts"))
from e2_strategy_map import best_speedup  # noqa: E402

PAIR = {  # cost arm -> beta arm
    "d_w4marlin": "q_int4", "d_w4machete": "q_int4", "d_fp8w8a8": "q_fp8",
    "d_win": "win512", "d_win128": "win128", "d_skip50": "skip50",
    "d_skip25": "skip25", "d_kvq": "kvq_fp8",
    "m_fp8marlin": "q_fp8", "m_fp8block": "q_fp8", "m_win": "win512",
    "m_win128": "win128", "m_localroute": "lr25", "m_skip50": "skip50",
    "m_kvq": "kvq_fp8",
    "ds_fp8marlin": "q_fp8", "ds_fp8block": "q_fp8", "ds_win": "win512",
    "ds_skip50": "skip50", "ds_kvq": "kvq_fp8",
}
REFERENCE = {"d_kvq", "m_kvq", "ds_kvq"}
BETA_MODEL = {"dense": "dense", "moe": "moe", "mla": "mla"}
CTX_OF = {2: 2048, 8: 8192, 16: 16384, 32: 32768}
E2E_TRUTH = [  # (group, cell, winner-arm-prefix, measured x) -- consistency row
    ("dense", (1, 2), "w4", 1.21), ("moe", (8, 16), "win", 1.16),
    ("moe", (8, 32), "win", 1.34), ("dense", (32, 16), "win", 1.54),
    ("moe", (32, 32), "win", 1.64),
]


def main() -> int:
    beta = {}
    for r in csv.DictReader((P77 / "data/beta.csv").open()):
        beta[(r["model"], r["arm"], int(r["ctx"]))] = float(r["beta_greedy"])
    R = defaultdict(dict)
    for row in csv.DictReader((P76 / "data/e1/summary.csv").open()):
        if int(row.get("overpool", 0)):
            continue
        R[row["group"]][(row["arm"], (int(row["batch"]), int(row["ctx"])))] = \
            float(row["ratio"])

    out = ["# Strategy map v3 — three architectures, measured R × measured β\n"]
    winners = {}
    for group in ("dense", "moe", "mla"):
        cells = sorted({c for (a, c) in R[group] if a in PAIR and c[1] != 8})
        arms = sorted({a for (a, _) in R[group] if a in PAIR})
        sel = [a for a in arms if a not in REFERENCE]
        out.append(f"\n## {group}\n")
        hdr = ["cell"] + [a.split("_", 1)[1] for a in sel] + ["WINNER"]
        out.append("| " + " | ".join(hdr) + " |")
        out.append("|" + "|".join("---" for _ in hdr) + "|")
        for cell in cells:
            b, ck = cell
            row, best = [], (1.0, None, 0)
            for a in sel:
                r = R[group].get((a, cell))
                bm = beta.get((BETA_MODEL[group], PAIR[a], CTX_OF[ck]))
                if r is None or bm is None:
                    row.append("-")
                    continue
                s, g = best_speedup(min(bm, 0.995), r)
                row.append(f"{s:.2f}")
                if s > best[0]:
                    best = (s, a, g)
            win = (f"**{best[1].split('_', 1)[1]} {best[0]:.2f}× (γ{best[2]})**"
                   if best[1] else "**OFF**")
            winners[(group, cell)] = best
            out.append(f"| b{b}/{ck}k | " + " | ".join(row) + f" | {win} |")

    out.append("\n## Consistency with the five e2e ground-truth cells\n")
    out.append("| cell | v3 winner (predicted) | e2e measured |")
    out.append("|---|---|---|")
    for (g, cell, wpref, meas) in E2E_TRUTH:
        w = winners.get((g, cell))
        tag = w[1].split("_", 1)[1] if w and w[1] else "OFF"
        okay = "✓" if wpref in tag else "✗"
        out.append(f"| {g} b{cell[0]}/{cell[1]}k | {tag} {w[0]:.2f}× | "
                   f"{wpref}* {meas}× {okay} |")
    out.append(
        "\n**Notes**: β ctx-specific per architecture (77), capped 0.995; R "
        "measured (76 + MLA confirmation tier). MLA local-route: β measured "
        "(0.88-0.99, shared-expert anchor) but its cost regime is comm-bound "
        "fabric — no NVLink cost arm; kvq columns are reference-only. Deep-γ "
        "cells are rooflines (delivery 90%@γ3 → 86%@γ6, P75/76-E3).\n\n"
        "**Honest readings**: (1) the MLA column is effectively an OFF region "
        "— every winner ≤1.13× roofline, i.e. within the noise band and "
        "known delivery factor of parity; this CONFIRMS the registered V3 "
        "claim (no strong self-spec lever on NVLink MLA — its measured-β "
        "strengths, shared-expert local-route 0.95-0.99 and shallow-skip "
        "0.97, await the comm-bound fabric and a skip cost arm). (2) MoE "
        "b4 long-ctx winners are NOT robust (S4 placement noise; the "
        "win128 1.58× at b4/32k rides a noisy R) — quote b8/b32 rows. "
        "(3) win128 vs win512 winners at b8+/16k trade within ±0.02 — same-"
        "family ties, not crossovers.")
    text = "\n".join(out)
    (PHASE / "data/strategy_map_v3.md").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
