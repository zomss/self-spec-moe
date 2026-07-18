#!/usr/bin/env python3
"""Map v6.3: reprice with the measured Humming-W4A8 realization (Phase 87).

Two additions over v6:
  1. dense (Q2.5-7B column): new config q_int4+win512+A8 with the Humming
     kernel-factor transfer -- R_w4a8h = R_w4win_combo x KFAC(b), where
     KFAC is the MEASURED Humming/Marlin e2e-implied-R ratio on Q3-8B
     (b8 .945, b16 .841) and wide-sigma extrapolation elsewhere.
     beta = GPTQ-combo beta x actfp8 factor .9922 (measured on Q3-8B;
     sigma widened for the no-QK-norm transfer). Ckpt UNBUILT -> any win
     goes to the measurement queue, not the map headline.
     (CutlassW4A8 factors measured 1.12-1.23x WORSE than Marlin -- the
     realization is dominated; not tabled.)
  2. dense_q38b: the measured Qwen3-8B column at 16k -- e2e-measured
     arms sampled directly (sigma 2%), model rows priced from
     e2e-implied R with term edits (as price_w4ffn).

Winner = argmax LCB05, P(win) over MC draws; OFF = 1.0 exactly.
Emits its own measurement queue.
"""
import random
import sys
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / "scripts"))
import map_v6 as M6  # noqa: E402
import search_v4 as SV  # noqa: E402

random.seed(63)
N_MC = 2000

# Humming/Marlin e2e-implied R ratio by batch (Q3-8B measured at b8/b16)
KFAC = {1: (1.05, 0.15), 4: (0.98, 0.12), 8: (0.945, 0.08),
        16: (0.88, 0.08), 32: (0.82, 0.15)}
A8BETA_FAC = (0.9922, 0.010)  # actfp8 measured Q3-8B; transfer sigma below

W4A8_ROW = ("q_int4+win512+A8humming",
            0.917 * (0.9427 / 0.924) * 0.9922, 0.022,
            ("w4a8h",), None, "fixed", 8,
            "kernel-factor transfer from 8B; CKPT UNBUILT (queue)")

_orig_resolve = M6.resolve_R


def resolve_R(group, name, spec, b, ck, Rs):
    if isinstance(spec, tuple) and spec and spec[0] == "w4a8h":
        base = _orig_resolve(group, name, "combo", b, ck, Rs)
        if base is None:
            return None
        f, fs = KFAC.get(b, (0.95, 0.20))
        return (base[0] * f, base[1] + fs)
    return _orig_resolve(group, name, spec, b, ck, Rs)


M6.resolve_R = resolve_R
M6.CONFIGS["dense"].insert(2, W4A8_ROW)

# ---------------- dense_q38b: the measured column ----------------
# name -> cell -> ("e2e", tau, speedup) or ("model", beta, R, Rsig)
MLP_SHARE = 0.783
Q38B = {
    "w4win": {(1, 16): ("e2e", 4.49, 1.42), (8, 16): ("e2e", 4.51, 1.81),
              (16, 16): ("e2e", 5.69, 1.79)},
    "w4a8win_humming": {(8, 16): ("e2e", 4.58, 1.90),
                        (16, 16): ("e2e", 6.02, 2.19)},
    "w4a8win_cutlass": {(8, 16): ("e2e", 4.54, 1.60),
                        (16, 16): ("e2e", 5.94, 1.78)},
}
# model rows built at runtime from implied R
WSHARE = {1: (0.70, 0.12), 8: (0.50, 0.18), 16: (0.45, 0.18)}
B87 = {"w4ffn25win": (0.8628 * 0.975, 0.25)}


def tau(beta, g):
    return (1 - beta ** (g + 1)) / (1 - beta) if beta < 1 else g + 1


def price_q38b(cell):
    b, ck = cell
    anchor = Q38B["w4win"][cell]
    R_imp = (anchor[1] / anchor[2] - 1) / (4 if b < 16 else 6)
    cands = []
    for name, cells in Q38B.items():
        if cell in cells:
            cands.append((name, ("e2e", cells[cell][2], 0.02)))
    if cell not in Q38B["w4a8win_humming"]:
        # model row: weight-bound cell, act-quant overhead factor
        f, fs = KFAC[b]
        cands.append(("w4a8win_humming*", ("modelR", 0.950, R_imp * f,
                                           fs + 0.06)))
    for name, (beta, frac) in B87.items():
        cands.append((name, ("ffn", beta, frac)))
    stats = {n: [] for n, _ in cands}
    wins = {n: 0 for n, _ in cands}
    wins["OFF"] = 0
    for _ in range(N_MC):
        best_v, best_n = 1.0, "OFF"
        smp = max(0.05, min(0.95, random.gauss(*WSHARE[b])))
        for name, spec in cands:
            if spec[0] == "e2e":
                v = random.gauss(spec[1], spec[2] * spec[1])
            elif spec[0] == "modelR":
                bs = min(0.995, random.gauss(spec[1], 0.012))
                R = max(0.02, random.gauss(spec[2], spec[3] * spec[2]))
                v = max(tau(bs, g) / (g * R + 1) for g in range(1, 9))
            else:  # ffn term edit
                bs = min(0.995, random.gauss(spec[1], 0.012))
                R = R_imp * (1 - smp * MLP_SHARE * spec[2]) \
                    * random.gauss(1.0, 0.06)
                v = max(tau(bs, g) / (g * R + 1) for g in range(1, 9))
            stats[name].append(v)
            if v > best_v:
                best_v, best_n = v, name
        wins[best_n] += 1
    rows = []
    for name, _ in cands:
        vs = sorted(stats[name])
        rows.append((vs[int(0.05 * N_MC)], vs[N_MC // 2],
                     wins[name] / N_MC, name))
    rows.sort(reverse=True)
    return rows


def main():
    _, _, Rs, _, _ = SV.load_all()
    lines = ["# Strategy map v6.3 — Humming-W4A8 reprice (Phase 87 fold)",
             "",
             "Delta vs v6: dense gains q_int4+win512+A8humming (kernel-"
             "factor transfer, ckpt unbuilt -> queue-gated); new measured "
             "Qwen3-8B column. Winner = argmax LCB05.", ""]
    queue = []
    lines.append("\n## dense (Qwen2.5-7B, repriced)\n")
    lines.append("| cell | winner (LCB05/med/P) | runner-up | note |")
    lines.append("|---|---|---|---|")
    for b in (1, 8, 32):
        for ck in (2, 16, 32):
            if ("dense", (b, ck)) in M6.INFEASIBLE:
                lines.append(f"| b{b}/{ck}k | INFEASIBLE (residency) | | |")
                continue
            ranked, p_off = M6.price_cell("dense", b, ck, Rs)
            w = ranked[0]
            ru = ranked[1] if len(ranked) > 1 else None
            mark = ""
            if w[3] == W4A8_ROW[0]:
                mark = " [QUEUE: ckpt unbuilt]"
                queue.append(f"Q2.5-7B W4A8 ckpt + e2e arm at b{b}/{ck}k "
                             f"(map winner by LCB {w[0]:.2f} vs "
                             f"{ru[3]} {ru[0]:.2f})")
            elif ru and ru[3] == W4A8_ROW[0] and w[0] - ru[0] < 0.10:
                mark = " [CONTESTED]"
                queue.append(f"Q2.5-7B W4A8 arm at b{b}/{ck}k CONTESTED "
                             f"(gap {w[0] - ru[0]:.2f} < model error; "
                             f"nominate-confirm rule)")
            lines.append(f"| b{b}/{ck}k | **{w[3]}** {w[0]:.2f}/{w[1]:.2f}/"
                         f"P{w[2]:.2f}{mark} | "
                         f"{ru[3] + f' {ru[0]:.2f}' if ru else '—'} | {w[4]} |")
    lines.append("\n## dense_q38b (Qwen3-8B, measured column, 16k)\n")
    lines.append("| cell | winner (LCB05/med/P) | runner-up | rest |")
    lines.append("|---|---|---|---|")
    for cell in [(1, 16), (8, 16), (16, 16)]:
        rows = price_q38b(cell)
        w, ru = rows[0], rows[1]
        rest = "; ".join(f"{r[3]} {r[0]:.2f}" for r in rows[2:])
        if w[3].endswith("*"):
            queue.append(f"Q3-8B w4a8win arm at b{cell[0]}/16k (model-R "
                         f"winner LCB {w[0]:.2f})")
        lines.append(f"| b{cell[0]}/16k | **{w[3]}** {w[0]:.2f}/{w[1]:.2f}/"
                     f"P{w[2]:.2f} | {ru[3]} {ru[0]:.2f}/P{ru[2]:.2f} | "
                     f"{rest} |")
    lines.append("\n## measurement queue (emitted)\n")
    if queue:
        for q in queue:
            lines.append(f"- {q}")
    else:
        lines.append("- (empty: no model-R config wins any cell)")
    lines.append("\nStanding queue items: w4a8 b32 8B cell unmeasured. "
                 "RESOLVED 2026-07-18: 32B W4A8 arm MEASURED -- b8 K5 1.22x "
                 "vs w4win 1.28x (no flip; kernel factor ~1.05 at 32B/TP2 "
                 "vs 0.945 at 8B -> scale/TP-dependent); b16/16k "
                 "capacity-infeasible at TP2 (KV 185k < 262k, 7th residency "
                 "incident).")
    text = "\n".join(lines)
    (PHASE / "data/strategy_map_v6_3.md").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
