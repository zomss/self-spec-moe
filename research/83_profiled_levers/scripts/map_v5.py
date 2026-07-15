#!/usr/bin/env python3
"""E4: strategy map v5 -- the exhaustive search with PROFILED lever columns.

Injects Phase-83's profiled betas into the 79 winner-hardening machinery:
  dense: ls3 (iter-greedy layer set {4,5,6}, beta 0.869); q_int4 -> the
         E3 calibrated (GPTQ) beta when measured (data/beta_gptq.csv),
         else the RTN 0.924 stands.
  moe:   flr50 (0.9531) / flr25 (0.8229) with lr cost structure;
         profiled skip6 (0.742, ls=42/48).
  mla:   unchanged (layer profile flat; OFF triply measured).

Provenance rules for the emitted map: flr cells at the 2k band carry the
E2b/E2c e2e verdicts (b4 DELIVERED 1.03x; b8/b32 chain-blocked); other
profiled winners are PRICED. Output: data/strategy_map_v5.md with the v4
diff and flip list.
"""

import csv
import itertools
import sys
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
P79 = PHASE.parent / "79_paper"
R80 = PHASE.parent / "80_lever_composition"
P76 = PHASE.parent / "76_lever_latency_sweep"
P78 = PHASE.parent / "78_cost_model"
for p in (R80, P76, P78, P79):
    sys.path.insert(0, str(p / "scripts"))
import search_v4 as SV  # noqa: E402
from e2_strategy_map import best_speedup  # noqa: E402

CTX_OF = SV.CTX_OF

# v4 winners (for the diff column)
V4 = {
    ("dense", (1, 2)): ("q_int4", 1.28), ("dense", (1, 16)): ("q_int4", 1.28),
    ("dense", (1, 32)): ("q_int4+win512", 1.31),
    ("dense", (8, 2)): ("q_int4", 1.30), ("dense", (8, 16)): ("q_int4+win512", 1.55),
    ("dense", (8, 32)): ("q_int4+win512", 1.61),
    ("dense", (32, 2)): ("q_int4+win128", 1.27),
    ("dense", (32, 16)): ("q_int4+win512", 1.91),
    ("moe", (4, 2)): ("lr50+q_fp8", 1.04), ("moe", (4, 16)): ("win128", 1.05),
    ("moe", (4, 32)): ("win128", 1.58),
    ("moe", (8, 2)): ("q_fp8", 1.06), ("moe", (8, 16)): ("win512", 1.27),
    ("moe", (8, 32)): ("win512", 1.37),
    ("moe", (32, 2)): ("q_fp8+win512", 1.07), ("moe", (32, 16)): ("win128", 1.44),
    ("moe", (32, 32)): ("win512", 2.22),
    # mla v4 argmaxes (the OFF verdict is the marginality reading, not the
    # argmax; compare argmax-to-argmax so unchanged cells don't count as flips)
    ("mla", (4, 2)): ("q_fp8", 1.08), ("mla", (4, 16)): ("skip125", 1.02),
    ("mla", (4, 32)): ("q_fp8", 1.07),
    ("mla", (8, 2)): ("skip125", 1.02), ("mla", (8, 16)): ("skip125", 1.03),
    ("mla", (8, 32)): ("win512", 1.05),
    ("mla", (32, 2)): ("q_fp8", 1.07), ("mla", (32, 16)): ("q_fp8", 1.13),
    ("mla", (32, 32)): ("q_fp8", 1.04),
}

# profiled beta values (Phase 83, measured offline; e2e verdicts in notes)
PROF_BETA = {"ls3_dense": 0.869, "flr50": 0.9531, "flr25": 0.8229,
             "skip6p_moe": 0.742}
E2E_NOTE = {("moe", "flr50", (4, 2)): "e2e DELIVERED 1.03x (E2c)",
            ("moe", "flr50", (8, 2)): "e2e 0.63x -- chain-blocked (E2c)",
            ("moe", "flr50", (32, 2)): "e2e 0.64x -- chain-blocked (E2c)"}


def gptq_beta():
    f = PHASE / "data/beta_gptq.csv"
    if not f.exists():
        return None
    for r in csv.DictReader(f.open()):
        if r["arm"] == "q_int4_gptq":
            return float(r["beta_greedy"])
    return None


def components(group, beta_s, ctx, q4):
    """(token, beta-getter, cost-spec) per group, profiled columns included."""
    def meas(arm):
        return beta_s.get((group, arm, ctx))
    C = []
    if group == "dense":
        C = [("win512", meas("win512"), dict(win=528)),
             ("win128", meas("win128"), dict(win=144)),
             ("q_int4", q4 or meas("q_int4"), dict(w=0.28, kappa="marlin")),
             ("q_fp8", meas("q_fp8"), dict(w=0.5, kappa="fp8d")),
             ("ls3", PROF_BETA["ls3_dense"], dict(ls=25 / 28)),
             ("skip125", meas("skip125"), dict(ls=0.875))]
    elif group == "moe":
        C = [("win512", meas("win512"), dict(win=528)),
             ("win128", meas("win128"), dict(win=144)),
             ("q_fp8", meas("q_fp8"), dict(w=0.5, kappa="fp8m")),
             ("flr50", PROF_BETA["flr50"], dict(comm_off=True, local=True)),
             ("flr25", PROF_BETA["flr25"], dict(comm_off=True, local=True)),
             ("skip6p", PROF_BETA["skip6p_moe"], dict(ls=42 / 48)),
             ("skip125", meas("skip125"), dict(ls=0.875))]
    else:
        C = [("win512", meas("win512"), dict(win=528)),
             ("q_fp8", meas("q_fp8"), dict(w=0.5, kappa="fp8m")),
             ("skip125", meas("skip125"), dict(ls=0.875)),
             ("lr50", meas("lr50"), dict(comm_off=True, local=True))]
    return [(t, b, s) for t, b, s in C if b is not None]


CLASS = ("win", "q_", "skip", "ls", "flr", "lr")


def klass(t):
    for k in ("win", "q_", "flr", "lr", "skip", "ls"):
        if t.startswith(k):
            return {"flr": "lr", "ls": "skip"}.get(k, k)
    return t


def cands(comps):
    out = []
    for size in (1, 2, 3):
        for combo in itertools.combinations(comps, size):
            ks = [klass(c[0]) for c in combo]
            if len(set(ks)) != len(ks):
                continue
            out.append(combo)
    return out


def main() -> int:
    beta_s, beta_c, Rs, blob, v3b = SV.load_all()
    q4 = gptq_beta()
    lines = ["# Strategy map v5 — profiled levers (Phase 83)",
             "", f"q_int4 beta source: "
             f"{'GPTQ-calibrated %.4f' % q4 if q4 else 'RTN 0.924 (E3 pending)'}.",
             "Profiled columns: dense ls3 0.869; moe flr50 0.9531 / flr25",
             "0.8229 / skip6p 0.742. mla unchanged. delivery=1 pricing;",
             "e2e verdicts noted where measured (E2b/E2c).", ""]
    flips = []
    groups = {"dense": [1, 8, 32], "moe": [4, 8, 32], "mla": [4, 8, 32]}
    for group, bs in groups.items():
        lines.append(f"\n## {group}\n")
        lines.append("| cell | v5 winner | speedup* | v4 | note |")
        lines.append("|---|---|---|---|---|")
        for b in bs:
            for ck in (2, 16, 32):
                if group == "dense" and (b, ck) == (32, 32):
                    lines.append("| b32/32k | (over-capacity) | | | |")
                    continue
                ctx = CTX_OF[ck]
                comps = components(group, beta_s, ctx, q4)
                best = (1.0, "OFF", 0)
                for combo in cands(comps):
                    toks = sorted(c[0] for c in combo)
                    name = "+".join(toks)
                    bm = beta_c.get((group, name, ctx))
                    if bm is None:
                        bm = 1.0
                        for c in combo:
                            bm *= c[1]
                    r = SV.combo_R_measured(group, name, b, ck, Rs)
                    if r is None and len(combo) == 1:
                        arm = {("dense", "win512"): "d_win", ("dense", "win128"): "d_win128",
                               ("dense", "q_int4"): "d_w4marlin", ("dense", "q_fp8"): "d_fp8w8a8",
                               ("moe", "win512"): "m_win", ("moe", "win128"): "m_win128",
                               ("moe", "q_fp8"): "m_fp8marlin",
                               ("moe", "flr50"): "m_localroute", ("moe", "flr25"): "m_localroute",
                               ("mla", "win512"): "ds_win", ("mla", "q_fp8"): "ds_fp8marlin",
                               }.get((group, toks[0]))
                        r = Rs[group].get((arm, (b, ck))) if arm else None
                    if r is None:
                        spec = {}
                        for c in combo:
                            spec.update(c[2])
                        try:
                            r = SV.term_R(group, spec, b, ck, blob, v3b)
                        except Exception:
                            continue
                    s, g = best_speedup(min(bm, 0.995), r)
                    if s > best[0]:
                        best = (s, name, g)
                v4w = V4.get((group, (b, ck)), ("?", 0))
                note = ""
                for c in best[1].split("+"):
                    note = E2E_NOTE.get((group, c, (b, ck)), note)
                flip = best[1] != v4w[0]
                if flip:
                    flips.append((group, (b, ck), v4w, best, note))
                marg = " (marginal/OFF region)" if best[0] <= 1.15 else ""
                lines.append(f"| b{b}/{ck}k | **{best[1]}**{marg} | {best[0]:.2f}× "
                             f"| {v4w[0]} {v4w[1]:.2f} | "
                             f"{'FLIP. ' if flip else ''}{note} |")
    lines.append(f"\n## Flips vs v4: {len(flips)}\n")
    for g, cell, v4w, best, note in flips:
        lines.append(f"- {g} b{cell[0]}/{cell[1]}k: {v4w[0]} {v4w[1]:.2f} -> "
                     f"**{best[1]} {best[0]:.2f}** {note}")
    text = "\n".join(lines)
    (PHASE / "data/strategy_map_v5.md").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
