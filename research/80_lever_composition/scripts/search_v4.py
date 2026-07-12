#!/usr/bin/env python3
"""E2: the offline-profiling search -> strategy map v4 (composed configs).

Per (architecture, batch, ctx), enumerate draft CONFIGURATIONS = lever subset
(size <=3: <=1 window, <=1 weight-quant, optional kvq/skip125/lr50) x gamma<=8
and pick the best composed speedup, including OFF.

beta sources (priority): measured combo (E0 beta_combo.csv) > product law
over measured singles (77) -- with the E0 exceptions coded:
  - mla + skip + any context lever at ctx>=32k: EXCLUDED unless measured
    (E0: destructive, -0.10..-0.12)
  - product law elsewhere (E0: median dev ~0.01)
R sources (priority): measured combo arm (76 JSONs) > measured single
(summary.csv) > term-edit model (78 fit; mla via V3b floor split; windowed
MoE combos are conservative pending the h-term refinement -- flagged).

Emits data/strategy_map_v4.md with v3 diffs + registered E3 candidates.
"""

import csv
import itertools
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PHASE = Path(__file__).resolve().parent.parent
P76 = PHASE.parent / "76_lever_latency_sweep"
P77 = PHASE.parent / "77_acceptance_map"
P78 = PHASE.parent / "78_cost_model"
sys.path.insert(0, str(P76 / "scripts"))
sys.path.insert(0, str(P78 / "scripts"))
from e2_strategy_map import best_speedup  # noqa: E402
import fit_cost_model as M  # noqa: E402
import predict_v3_mla  # noqa: E402,F401

BETA_MODEL = {"dense": "dense", "moe": "moe", "mla": "mla"}
CTX_OF = {2: 2048, 16: 16384, 32: 32768}
CELLS = {"dense": [(b, c) for b in (1, 8, 32) for c in (2, 16, 32)],
         "moe": [(b, c) for b in (4, 8, 32) for c in (2, 16, 32)],
         "mla": [(b, c) for b in (4, 8, 32) for c in (2, 16, 32)]}
OVERCAP = {("dense", (32, 32))}          # bf16 denominator thrash
# components: (token, beta-arm, spec-edits)
# kvq_fp8 EXCLUDED from selection: not draft-only implementable in the
# harness (P74 SHARED_KV) -- stays a pool-motivating reference, as in map v3.
COMPONENTS = {
    "dense": [("win512", "win512", dict(win=528)), ("win128", "win128", dict(win=144)),
              ("q_int4", "q_int4", dict(w=0.28, kappa="marlin")),
              ("q_fp8", "q_fp8", dict(w=0.5, kappa="fp8d")),
              ("skip125", "skip125", dict(ls=0.875))],
    "moe": [("win512", "win512", dict(win=528)), ("win128", "win128", dict(win=144)),
            ("q_fp8", "q_fp8", dict(w=0.5, kappa="fp8m")),
            ("skip125", "skip125", dict(ls=0.875)),
            ("lr50", "lr50", dict(comm_off=True, local=True))],
    "mla": [("win512", "win512", dict(win=528)),
            ("q_fp8", "q_fp8", dict(w=0.5, kappa="fp8m")),
            ("skip125", "skip125", dict(ls=0.875))],
}
CTX_LEVERS = {"win512", "win128", "lr50", "kvq_fp8"}   # for the mla-skip exclusion
# measured combo R arms: combo-name -> runner arm name
MEASURED_R = {("dense", "q_int4+win512"): "d_w4win",
              ("dense", "kvq_fp8+win512"): "d_kvqwin",
              ("moe", "lr50+win512"): "m_winlr",
              ("moe", "lr50+q_fp8+win512"): "m_winlrq",
              ("mla", "q_fp8+skip125"): "ds_skip125q"}   # canonical (sorted) names


def canon(name: str) -> str:
    return "+".join(sorted(name.split("+")))


def load_all():
    beta_s, beta_c = {}, {}
    for r in csv.DictReader((P77 / "data/beta.csv").open()):
        beta_s[(r["model"], r["arm"], int(r["ctx"]))] = float(r["beta_greedy"])
    for r in csv.DictReader((PHASE / "data/beta_combo.csv").open()):
        beta_c[(r["model"], canon(r["combo"]), int(r["ctx"]))] = float(r["beta_greedy"])
    Rs = defaultdict(dict)
    for row in csv.DictReader((P76 / "data/e1/summary.csv").open()):
        if int(row.get("overpool", 0)):
            continue
        Rs[row["group"]][(row["arm"], (int(row["batch"]), int(row["ctx"])))] = \
            float(row["ratio"])
    blob = json.loads((P78 / "data/fit.json").read_text())
    v3b = json.loads((P78 / "data/v3b_verify.json").read_text())
    return beta_s, beta_c, Rs, blob, v3b


def combo_R_measured(group, combo, b, ck, Rs):
    arm = MEASURED_R.get((group, combo))
    if not arm:
        return None
    vals = [json.loads(f.read_text())["mean_tpot_ms"] for r in (1, 2, 3, 4)
            if (f := P76 / f"data/e1/{group}_{arm}_b{b}_c{ck}k_r{r}.json").exists()]
    if not vals:
        return None
    t = statistics.median(vals)
    # denominator: reconstructed bf16 (dummy for skip combos)
    den_rows = []
    for row in csv.DictReader((P76 / "data/e1/summary.csv").open()):
        if (row["group"] == group and int(row["batch"]) == b
                and int(row["ctx"]) == ck and not int(row["overpool"])):
            want = "dummy" if "skip" in combo else "_bf16"
            if row["denom"].endswith(want if want != "_bf16" else "_bf16") and \
               ("dummy" in row["denom"]) == ("skip" in combo):
                den_rows.append(float(row["tpot_ms"]) / float(row["ratio"]))
    return t / statistics.median(den_rows) if den_rows else None


def term_R(group, spec, b, ck, blob, v3b):
    key = f"{'moe' if group == 'mla' else group}/additive"
    p = blob[key]["params"]
    kap = [f"kappa_{k}" for k in M.KAPPAS["moe" if group == "mla" else group]]
    arm_spec = dict(spec)
    if arm_spec.pop("comm_off", None):
        pass
    else:
        arm_spec["comm"] = group in ("moe", "mla")
    M.ARMS[(("moe" if group == "mla" else group), "_tmp")] = arm_spec
    M.ARMS[(("moe" if group == "mla" else group), "_tmpbf")] = (
        dict(comm=True) if group in ("moe", "mla") else dict())
    g = "moe" if group == "mla" else group
    if group == "mla":
        M.ARMS[("mla", "_tmp")] = arm_spec
        M.ARMS[("mla", "_tmpbf")] = dict(comm=True)
        pv = [0.0, 0.0, p["BW_GBs"], p["h_ms_per_Mtok"], p["c0"], p["c1"],
              p["kappa_fp8m"], p["kappa_fib"], p["kappa_a2atile"]]
        ls = arm_spec.get("ls", 1.0)
        f = v3b["f_step"] + 1.78 * ls
        t = M.predict("mla", pv, "_tmp", b, ck, "additive") + f
        d = M.predict("mla", pv, "_tmpbf", b, ck, "additive") + v3b["f_step"] + 1.78
        return t / d
    names = (["f0", "f1", "BW_GBs", "h_ms_per_Mtok"]
             + (["c0", "c1"] if g == "moe" else []) + kap)
    pv = [p[n] for n in names]
    t = M.predict(g, pv, "_tmp", b, ck, "additive")
    d = M.predict(g, pv, "_tmpbf" if g == "moe" else "bf16", b, ck, "additive")
    return t / d


def main() -> int:
    beta_s, beta_c, Rs, blob, v3b = load_all()
    out = ["# Strategy map v4 — composed configurations (search over measured physics)\n"]
    e3 = []
    for group in ("dense", "moe", "mla"):
        comps = COMPONENTS[group]
        # candidate configs: singles + pairs/triples (<=1 win, <=1 quant)
        cands = []
        for size in (1, 2, 3):
            for combo in itertools.combinations(comps, size):
                toks = [c[0] for c in combo]
                if sum(t.startswith("win") for t in toks) > 1:
                    continue
                if sum(t.startswith("q_") for t in toks) > 1:
                    continue
                cands.append(combo)
        out.append(f"\n## {group}\n")
        out.append("| cell | best config | speedup* | γ | source | v3 winner |")
        out.append("|---|---|---|---|---|---|")
        v3w = {  # from strategy_map_v3 (winners), for the diff column
            ("dense", (1, 2)): "w4marlin 1.28", ("dense", (8, 16)): "win128 1.20",
            ("dense", (32, 16)): "win128 1.56", ("moe", (32, 32)): "win 2.22",
            ("moe", (8, 16)): "win 1.27", ("mla", (32, 16)): "fp8marlin 1.13"}
        for cell in CELLS[group]:
            if (group, cell) in OVERCAP:
                out.append(f"| b{cell[0]}/{cell[1]}k | (bf16 over-capacity) | | | | |")
                continue
            b, ck = cell
            best = (1.0, "OFF", 0, "")
            for combo in cands:
                toks = sorted(c[0] for c in combo)
                name = "+".join(toks)
                ctx = CTX_OF[ck]
                # beta: measured combo > product law (with the mla exclusion)
                bm = beta_c.get((group, name, ctx))
                src_b = "measβ"
                if bm is None:
                    if (group == "mla" and ck >= 32
                            and any(t.startswith("skip") for t in toks)
                            and any(t in CTX_LEVERS for t in toks)):
                        continue                     # E0 destructive, unmeasured
                    parts = [beta_s.get((BETA_MODEL[group], c[1], ctx)) for c in combo]
                    if any(x is None for x in parts):
                        continue
                    bm = 1.0
                    for x in parts:
                        bm *= x
                    src_b = "prodβ"
                # R: measured combo > measured single > term model
                r = combo_R_measured(group, name, b, ck, Rs)
                src_r = "measR"
                if r is None and len(combo) == 1:
                    single_arm = {("dense", "win512"): "d_win", ("dense", "win128"): "d_win128",
                                  ("dense", "q_int4"): "d_w4marlin", ("dense", "q_fp8"): "d_fp8w8a8",
                                  ("dense", "kvq_fp8"): "d_kvq", ("dense", "skip125"): None,
                                  ("moe", "win512"): "m_win", ("moe", "win128"): "m_win128",
                                  ("moe", "q_fp8"): "m_fp8marlin", ("moe", "kvq_fp8"): "m_kvq",
                                  ("moe", "skip125"): None, ("moe", "lr50"): "m_localroute",
                                  ("mla", "win512"): "ds_win", ("mla", "q_fp8"): "ds_fp8marlin",
                                  ("mla", "kvq_fp8"): "ds_kvq", ("mla", "skip125"): None,
                                  }.get((group, toks[0]))
                    if single_arm:
                        r = Rs[group].get((single_arm, cell))
                if r is None:
                    spec = {}
                    for c in combo:
                        spec.update(c[2])
                    try:
                        r = term_R(group, spec, b, ck, blob, v3b)
                        src_r = "modelR"
                    except Exception:
                        continue
                s, g = best_speedup(min(bm, 0.995), r)
                if s > best[0]:
                    best = (s, name, g, f"{src_b}/{src_r}")
            prev = v3w.get((group, cell), "")
            out.append(f"| b{b}/{ck}k | **{best[1]}** | {best[0]:.2f}× | γ{best[2]} "
                       f"| {best[3]} | {prev} |")
            if best[0] > 1.35 and "+" in best[1]:
                e3.append((group, cell, best))
    out.append("\n## Registered E3 candidates (composed cells worth an e2e check)\n")
    for (g, cell, best) in sorted(e3, key=lambda x: -x[2][0])[:4]:
        out.append(f"- {g} b{cell[0]}/{cell[1]}k: **{best[1]}** predicted "
                   f"{best[0]:.2f}× (γ{best[2]}, {best[3]})")
    out.append("\n**Notes**: measβ/prodβ = measured-combo vs product-law β; "
               "measR/modelR = measured combo arm vs term-edit prediction "
               "(windowed-MoE modelR is conservative pending the h-term "
               "refinement). Deep-γ cells are rooflines (delivery ≤86% at γ6).")
    text = "\n".join(out)
    (PHASE / "data/strategy_map_v4.md").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
