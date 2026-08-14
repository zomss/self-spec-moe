#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profiling-budget backtest: how few GPU-minutes recover the strategy map?

The selection step is exhaustive and free (CPU pricing over 53 configs);
the COST of the search is profiling. This backtest replays the minimal
protocol against the completed record, treating MLA as the new
architecture (which is how the project actually unfolded: dense+MoE
profiled in 76/77, MLA arrived unseen in 78-V3). At each budget level we
reveal only what the protocol would have measured, recompute every cell's
argmax, and score against the full-information map.

Budget levels
  L0 (0 min):   mechanism priors only. beta from config-derived classes
                (portable q_fp8 = other-arch mean; kvq via QK-norm rule --
                config shows kv_a_layernorm; lr via shared-expert rule;
                window/skip = other-arch class values, i.e. WRONG where the
                arch breaks them). R from the config-only transfer (V3
                failure mode: fully layer-proportional floor).
  L1 (+5 min):  one bf16 anchor cell -> V3b floor split for R.
  L2 (+~16 min): the mechanism checklist flags window/skip/lr as
                arch-ambiguous -> reveal their measured beta at ONE ctx
                (16k), extrapolated flat (the ctx-stability assumption --
                deliberately wrong for MLA-window, to see if it matters).
  L3 (+~8 min): decision-aware refinement: any lever whose measured-16k
                beta deviates >0.05 from its prior gets its other-ctx betas
                revealed (here: window), plus measured combo beta for the
                flagged destructive family.
  FULL:         everything measured (ground truth = off_hardening pricing).

Scores: winner match rate, median/max regret (true speedup of chosen config
vs true best), and OFF/ON decision accuracy. Output: data/search_backtest.md
"""

import itertools
import statistics
import sys
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
R80 = PHASE.parent / "80_lever_composition"
P76 = PHASE.parent / "76_lever_latency_sweep"
P78 = PHASE.parent / "78_cost_model"
sys.path.insert(0, str(R80 / "scripts"))
sys.path.insert(0, str(P76 / "scripts"))
sys.path.insert(0, str(P78 / "scripts"))

import search_v4 as SV  # noqa: E402
from e2_strategy_map import best_speedup  # noqa: E402

CTX_OF = SV.CTX_OF
CELLS = [(b, c) for b in (4, 8, 32) for c in (2, 16, 32)]
COMPONENTS = [
    ("win512", "win512", dict(win=528)),
    ("win128", "win128", dict(win=144)),
    ("q_fp8", "q_fp8", dict(w=0.5, kappa="fp8m")),
    ("skip125", "skip125", dict(ls=0.875)),
    ("skip25", "skip25", dict(ls=0.75)),
    ("lr50", "lr50", dict(comm_off=True, local=True)),
    ("lr25", "lr25", dict(comm_off=True, local=True)),
]
CLASS = {
    "win": lambda t: t.startswith("win"),
    "q": lambda t: t.startswith("q_"),
    "skip": lambda t: t.startswith("skip"),
    "lr": lambda t: t.startswith("lr"),
}
AMBIGUOUS = ("win512", "win128", "skip125", "skip25", "lr50", "lr25")

# GPU-minute accounting (measured wall times from the record):
#   refs for one (arch, ctx): ~7 min (today's math sweep, MLA);
#   one beta arm after refs: ~1.5 min; anchor cell: ~5 min (V3b);
#   full MLA profile actually spent: ~4.5 h serve sweep + ~35 min beta.
# L2 includes a second 5-min anchor: one measured q_fp8 R cell -- the
# checklist flags per-kernel kappa as arch-specific (the V3 lesson).
# L3: any ambiguous lever whose measured-16k beta deviates >0.05 from its
# prior gets its other-ctx betas revealed (2 more ctx ref sets + arms).
# L4: decision-aware R -- cells whose top-2 gap < the model-R error (0.09)
# get measured R for their measured-R-able singles (~2.5 min each).
COST = {
    "L0": 0,
    "L1": 5,
    "L2": 10 + 7 + 6 * 1.5,
    "L3": 10 + 7 + 6 * 1.5 + 20,
    "L4": None,
    "FULL": 270 + 35,
}
KAPPA_ANCHOR_CELL = (8, 16)  # one mid-grid fp8 R measurement


def kappa_offset(Rs, blob, v3b):
    """Calibrate the fp8 R at one measured cell; return additive offset."""
    b, ck = KAPPA_ANCHOR_CELL
    meas = Rs["mla"].get(("ds_fp8marlin", (b, ck)))
    if meas is None:
        return 0.0
    model = SV.term_R("mla", dict(w=0.5, kappa="fp8m"), b, ck, blob, v3b)
    return meas - model


def cands():
    out = []
    for size in (1, 2, 3, 4):
        for combo in itertools.combinations(COMPONENTS, size):
            toks = [c[0] for c in combo]
            if any(sum(f(t) for t in toks) > 1 for f in CLASS.values()):
                continue
            out.append(combo)
    return out


def prior_beta(beta_s, arm, ctx):
    """L0 config-derived class priors for MLA, from dense+moe measurements
    plus the mechanism rules (no MLA measurement)."""
    if arm.startswith("q_"):  # portability verdict: transfer the mean
        vals = [beta_s.get((m, arm, ctx)) for m in ("dense", "moe")]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None
    if arm == "kvq_fp8":  # QK-norm rule: config has kv_a_layernorm
        return 0.99
    if arm.startswith("lr"):  # shared-expert rule: config has 2 shared
        return {"lr50": 0.90, "lr25": 0.80}[arm]
    # window/skip: other-arch class value (deliberately wrong where MLA breaks)
    vals = [beta_s.get((m, arm, ctx)) for m in ("dense", "moe")]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def term_R_noanchor(spec, b, ck, blob, v3b):
    """V3 failure mode: whole floor layer-proportional (no anchor split)."""
    import fit_cost_model as M

    p = blob["moe/additive"]["params"]
    pv = [
        0.0,
        0.0,
        p["BW_GBs"],
        p["h_ms_per_Mtok"],
        p["c0"],
        p["c1"],
        p["kappa_fp8m"],
        p["kappa_fib"],
        p["kappa_a2atile"],
    ]
    arm_spec = dict(spec)
    if not arm_spec.pop("comm_off", None):
        arm_spec["comm"] = True
    M.ARMS[("mla", "_tmp")] = arm_spec
    M.ARMS[("mla", "_tmpbf")] = dict(comm=True)
    ls = arm_spec.get("ls", 1.0)
    f_total = v3b["f_step"] + 1.78
    t = M.predict("mla", pv, "_tmp", b, ck, "additive") + f_total * ls
    d = M.predict("mla", pv, "_tmpbf", b, ck, "additive") + f_total
    return t / d


def price_cell(level, b, ck, beta_s, beta_c, Rs, blob, v3b):
    ctx = CTX_OF[ck]
    best = (1.0, "OFF", 0)
    scored = []
    for combo in cands():
        toks = sorted(c[0] for c in combo)
        name = "+".join(toks)
        # ---- beta by level
        bm = None
        if level == "FULL":
            bm = beta_c.get(("mla", name, ctx))
        if (
            level in ("L3", "L4")
            and any(t.startswith("skip") for t in toks)
            and any(t in SV.CTX_LEVERS for t in toks)
            and ck >= 32
        ):
            bm = beta_c.get(("mla", name, ctx))  # flagged family: measured
        if bm is None:
            parts = []
            for c in combo:
                arm = c[1]
                v = None
                if level == "FULL":
                    v = beta_s.get(("mla", arm, ctx))
                elif level in ("L2", "L3", "L4") and arm in AMBIGUOUS:
                    prior = prior_beta(beta_s, arm, CTX_OF[16])
                    meas16 = beta_s.get(("mla", arm, CTX_OF[16]))
                    if (
                        level in ("L3", "L4")
                        and meas16 is not None
                        and prior is not None
                        and abs(meas16 - prior) > 0.05
                    ):
                        v = beta_s.get(("mla", arm, ctx))  # ctx revealed
                    else:
                        v = meas16  # flat 16k
                elif level in ("L2", "L3", "L4") and arm.startswith("q_"):
                    v = prior_beta(beta_s, arm, ctx)
                if v is None and level in ("L0", "L1", "L2", "L3"):
                    v = prior_beta(beta_s, arm, ctx)
                if v is None:
                    break
                parts.append(v)
            else:
                bm = 1.0
                for x in parts:
                    bm *= x
        if bm is None:
            continue
        # ---- R by level
        r = None
        src_meas = False
        if level == "L4" and len(combo) == 1 and (b, ck) in CONTESTED:
            arm = {"win512": "ds_win", "q_fp8": "ds_fp8marlin"}.get(toks[0])
            r = Rs["mla"].get((arm, (b, ck))) if arm else None
            src_meas = r is not None
        if level == "FULL":
            r = SV.combo_R_measured("mla", name, b, ck, Rs)
            if r is None and len(combo) == 1:
                arm = {"win512": "ds_win", "q_fp8": "ds_fp8marlin"}.get(toks[0])
                r = Rs["mla"].get((arm, (b, ck))) if arm else None
        spec = {}
        for c in combo:
            spec.update(c[2])
        if r is None:
            try:
                if level == "L0":
                    r = term_R_noanchor(spec, b, ck, blob, v3b)
                else:
                    r = SV.term_R("mla", spec, b, ck, blob, v3b)
                    if level in ("L2", "L3", "L4") and "kappa" in spec:
                        r += KOFF[0]  # quant-R anchor calibration
            except Exception:
                continue
        s, g = best_speedup(min(bm, 0.995), r)
        scored.append((s, name, g, src_meas))
        if s > best[0]:
            best = (s, name, g)
    if level == "L4" and scored:
        # conservatism rule: within model-R error of the top, prefer a
        # measured-R config over a model-priced one.
        scored.sort(reverse=True)
        top = scored[0]
        if not top[3]:
            for s, name, g, meas in scored:
                if top[0] - s > 0.09:
                    break
                if meas:
                    return (s, name, g)
    return best


KOFF = [0.0]
CONTESTED = set()


def main() -> int:
    beta_s, beta_c, Rs, blob, v3b = SV.load_all()
    KOFF[0] = kappa_offset(Rs, blob, v3b)
    # L4 contested cells: decided FROM L3 pricing (no truth peeking)
    n_meas = 0
    for cell in CELLS:
        ctx = CTX_OF[cell[1]]
        scores = []
        for combo in cands():
            toks = sorted(c[0] for c in combo)
            parts = []
            ok = True
            for c in combo:
                prior = prior_beta(beta_s, c[1], CTX_OF[16])
                m16 = beta_s.get(("mla", c[1], CTX_OF[16]))
                if c[1].startswith("q_"):
                    v = prior
                elif m16 is not None and prior is not None and abs(m16 - prior) > 0.05:
                    v = beta_s.get(("mla", c[1], ctx))
                else:
                    v = m16 if m16 is not None else prior
                if v is None:
                    ok = False
                    break
                parts.append(v)
            if not ok:
                continue
            bm = 1.0
            for x in parts:
                bm *= x
            spec = {}
            for c in combo:
                spec.update(c[2])
            try:
                r = SV.term_R("mla", spec, cell[0], cell[1], blob, v3b)
                if "kappa" in spec:
                    r += KOFF[0]
            except Exception:
                continue
            scores.append(best_speedup(min(bm, 0.995), r)[0])
        scores.sort(reverse=True)
        if len(scores) >= 2 and scores[0] - scores[1] < 0.09:
            CONTESTED.add(cell)
            n_meas += 2  # ds_win + ds_fp8marlin at cell
    COST["L4"] = COST["L3"] + 2.5 * n_meas
    truth = {
        cell: price_cell("FULL", *cell, beta_s, beta_c, Rs, blob, v3b) for cell in CELLS
    }
    lines = [
        "# Profiling-budget backtest — MLA as the new architecture",
        "",
        "Ground truth = full-information pricing (measured beta/R).",
        "Regret = true speedup of the chosen config vs the true best",
        "(chosen config re-priced with full information).",
        "",
    ]
    lines.append(
        "| level | GPU-min | winners match | near-opt (reg<=0.02) | "
        "OFF/ON match | median regret | max regret |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    detail = ["\n## Per-cell detail\n"]
    for level in ("L0", "L1", "L2", "L3", "L4", "FULL"):
        match = onoff = 0
        regrets = []
        detail.append(f"\n### {level}\n")
        detail.append("| cell | predicted | true | regret |")
        detail.append("|---|---|---|---|")
        for cell in CELLS:
            pred = price_cell(level, *cell, beta_s, beta_c, Rs, blob, v3b)
            t = truth[cell]
            # re-price the predicted config under full information
            if pred[1] == t[1]:
                true_of_pred = t[0]
            elif pred[1] == "OFF":
                true_of_pred = 1.0
            else:
                combo = [c for c in COMPONENTS if c[0] in pred[1].split("+")]
                full = price_cell("FULL", *cell, beta_s, beta_c, Rs, blob, v3b)
                # price just this config fully
                toks = sorted(c[0] for c in combo)
                name = "+".join(toks)
                ctx = CTX_OF[cell[1]]
                bm = beta_c.get(("mla", name, ctx))
                if bm is None:
                    parts = [beta_s.get(("mla", c[1], ctx)) for c in combo]
                    bm = 1.0
                    for x in parts:
                        bm *= x if x else 1.0
                spec = {}
                for c in combo:
                    spec.update(c[2])
                r = SV.combo_R_measured("mla", name, cell[0], cell[1], Rs)
                if r is None and len(combo) == 1:
                    arm = {"win512": "ds_win", "q_fp8": "ds_fp8marlin"}.get(toks[0])
                    r = Rs["mla"].get((arm, cell)) if arm else None
                if r is None:
                    r = SV.term_R("mla", spec, cell[0], cell[1], blob, v3b)
                true_of_pred, _ = best_speedup(min(bm, 0.995), r)
                _ = full
            reg = t[0] - true_of_pred
            regrets.append(reg)
            match += pred[1] == t[1]
            onoff += (pred[0] > 1.05) == (t[0] > 1.05)
            detail.append(
                f"| b{cell[0]}/{cell[1]}k | {pred[1]} {pred[0]:.2f}x "
                f"| {t[1]} {t[0]:.2f}x | {reg:+.3f} |"
            )
        nearopt = sum(r <= 0.02 for r in regrets)
        lines.append(
            f"| {level} | {COST[level]:.0f} | {match}/9 | {nearopt}/9 "
            f"| {onoff}/9 | {statistics.median(regrets):+.3f} "
            f"| {max(regrets):+.3f} |"
        )
    text = "\n".join(lines + detail)
    (PHASE / "data/search_backtest.md").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
