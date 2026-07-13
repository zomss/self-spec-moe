#!/usr/bin/env python3
"""Winner hardening: the OFF-hardening search over ALL cells (ON regions too).

Symmetric to off_hardening.py: the map's POSITIVE winners must also be the
argmax over the full (extended, discretized) combination space, not just
over the v4 component sets. Reports, per cell, the exhaustive argmax vs
the registered v3/v4 winner and flags any cell where a config outside the
v4 search set prices above the winner.

Motivation (2026-07-13, KnapSpec review): the map's OFF verdicts must mean
"the argmax over the whole combination space loses", not "the levers we
tried lose". Two fixes over 80/search_v4.py:
  1. EXTENDED component sets — lr50/lr25 (MLA beta .90-.98, previously
     absent from MLA's search set), skip25, win128 everywhere applicable;
     subsets up to size 4 (<=1 each of win/quant/skip/lr).
  2. OPTIMISTIC pricing — product-law beta with NO destructive penalty
     (measured exceptions are destructive, so product is an upper bound),
     measured combo beta/R where available, term-edit R otherwise; reported
     at delivery = 1 (the Phase-81 floor-free chain). If even this upper
     bound loses, OFF is search-backed. Cells where a challenger clears the
     bar are CANDIDATES FOR MEASUREMENT, not map updates.

kvq stays excluded (not draft-only implementable under SHARED_KV; its pool
variant is a reference column, per map v3/v4).

Output: data/off_hardening.md — per OFF cell, top-8 priced configs + verdict.
"""

import itertools
import sys
from pathlib import Path

PHASE = Path(__file__).resolve().parent.parent
R80 = PHASE.parent / "80_lever_composition"
P76 = PHASE.parent / "76_lever_latency_sweep"
P78 = PHASE.parent / "78_cost_model"
sys.path.insert(0, str(R80 / "scripts"))
sys.path.insert(0, str(P76 / "scripts"))
sys.path.insert(0, str(P78 / "scripts"))

import search_v4 as SV  # noqa: E402
from e2_strategy_map import best_speedup  # noqa: E402

CTX_OF = SV.CTX_OF

# extended component sets: (token, beta-arm, cost-spec)
COMPONENTS = {
    "dense": [
        ("win512", "win512", dict(win=528)),
        ("win128", "win128", dict(win=144)),
        ("q_int4", "q_int4", dict(w=0.28, kappa="marlin")),
        ("q_fp8", "q_fp8", dict(w=0.5, kappa="fp8d")),
        ("skip125", "skip125", dict(ls=0.875)),
        ("skip25", "skip25", dict(ls=0.75)),
    ],
    "moe": [
        ("win512", "win512", dict(win=528)),
        ("win128", "win128", dict(win=144)),
        ("q_fp8", "q_fp8", dict(w=0.5, kappa="fp8m")),
        ("skip125", "skip125", dict(ls=0.875)),
        ("skip25", "skip25", dict(ls=0.75)),
        ("lr50", "lr50", dict(comm_off=True, local=True)),
        ("lr25", "lr25", dict(comm_off=True, local=True)),
    ],
    "mla": [
        ("win512", "win512", dict(win=528)),
        ("win128", "win128", dict(win=144)),
        ("q_fp8", "q_fp8", dict(w=0.5, kappa="fp8m")),
        ("skip125", "skip125", dict(ls=0.875)),
        ("skip25", "skip25", dict(ls=0.75)),
        ("lr50", "lr50", dict(comm_off=True, local=True)),
        ("lr25", "lr25", dict(comm_off=True, local=True)),
    ],
}

# ALL cells, all groups (over-capacity bf16 denominators excluded)
OFF_CELLS = {
    "dense": [(b, c) for b in (1, 8, 32) for c in (2, 16, 32)
              if (b, c) != (32, 32)],
    "moe": [(b, c) for b in (4, 8, 32) for c in (2, 16, 32)],
    "mla": [(b, c) for b in (4, 8, 32) for c in (2, 16, 32)],
}

CLASS = {"win": lambda t: t.startswith("win"),
         "q": lambda t: t.startswith("q_"),
         "skip": lambda t: t.startswith("skip"),
         "lr": lambda t: t.startswith("lr")}


def cands(comps):
    out = []
    for size in (1, 2, 3, 4):
        for combo in itertools.combinations(comps, size):
            toks = [c[0] for c in combo]
            if any(sum(f(t) for t in toks) > 1 for f in CLASS.values()):
                continue
            out.append(combo)
    return out


def main() -> int:
    beta_s, beta_c, Rs, blob, v3b = SV.load_all()
    lines = ["# Winner hardening — exhaustive combo search over ALL cells",
             "", "Pricing: measured combo beta/R first; else product-law beta",
             "(UPPER bound; measured exceptions are destructive) x term-edit R.",
             "delivery = 1 (Phase-81 floor-free chain). kvq excluded (not",
             "draft-only implementable). `!` = combo in the measured-destructive",
             "family (mla, ctx>=32k, skip x context-lever) — its price is",
             "optimistic by ~0.10-0.12 beta.", ""]
    to_measure = []
    for group, cells in OFF_CELLS.items():
        n_cfg = len(cands(COMPONENTS[group]))
        lines.append(f"\n## {group} ({n_cfg} configs x gamma<=8 per cell)\n")
        for b, ck in cells:
            ctx = CTX_OF[ck]
            scored = []
            for combo in cands(COMPONENTS[group]):
                toks = sorted(c[0] for c in combo)
                name = "+".join(toks)
                destr = (group == "mla" and ck >= 32
                         and any(t.startswith("skip") for t in toks)
                         and any(t in SV.CTX_LEVERS for t in toks))
                bm = beta_c.get((group, name, ctx))
                src_b = "measB"
                if bm is None:
                    parts = [beta_s.get((SV.BETA_MODEL[group], c[1], ctx))
                             for c in combo]
                    if any(x is None for x in parts):
                        continue
                    bm = 1.0
                    for x in parts:
                        bm *= x
                    src_b = "prodB"
                r = SV.combo_R_measured(group, name, b, ck, Rs)
                src_r = "measR"
                if r is None and len(combo) == 1:
                    arm = {("dense", "win512"): "d_win", ("dense", "win128"): "d_win128",
                           ("dense", "q_int4"): "d_w4marlin", ("dense", "q_fp8"): "d_fp8w8a8",
                           ("moe", "win512"): "m_win", ("moe", "win128"): "m_win128",
                           ("moe", "q_fp8"): "m_fp8marlin", ("moe", "lr50"): "m_localroute",
                           ("mla", "win512"): "ds_win", ("mla", "q_fp8"): "ds_fp8marlin",
                           }.get((group, toks[0]))
                    r = Rs[group].get((arm, (b, ck))) if arm else None
                if r is None:
                    spec = {}
                    for c in combo:
                        spec.update(c[2])
                    try:
                        r = SV.term_R(group, spec, b, ck, blob, v3b)
                        src_r = "modelR"
                    except Exception:
                        continue
                s, g = best_speedup(min(bm, 0.995), r)
                scored.append((s, name + ("!" if destr else ""), g,
                               f"{src_b}/{src_r}", bm, r))
            scored.sort(reverse=True)
            best = scored[0]
            verdict = ("OFF HOLDS" if best[0] <= 1.05 else
                       "MARGINAL — measure" if best[0] <= 1.15 else
                       "CHALLENGER — measure")
            lines.append(f"### b{b}/{ck}k — max {best[0]:.2f}x ({best[1]}) "
                         f"-> **{verdict}**\n")
            lines.append("| config | speedup* | gamma | beta | R | source |")
            lines.append("|---|---|---|---|---|---|")
            for s, name, g, src, bm, r in scored[:8]:
                lines.append(f"| {name} | {s:.2f}x | {g} | {bm:.3f} | {r:.3f} | {src} |")
            lines.append("")
            if best[0] > 1.05:
                to_measure.append((group, (b, ck), best))
    lines.append("\n## Measurement queue (challengers above 1.05x optimistic)\n")
    for g, cell, best in sorted(to_measure, key=lambda x: -x[2][0]):
        lines.append(f"- {g} b{cell[0]}/{cell[1]}k: {best[1]} priced {best[0]:.2f}x "
                     f"(gamma{best[2]}, {best[3]})")
    text = "\n".join(lines)
    (PHASE / "data/winner_hardening.md").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
