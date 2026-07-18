#!/usr/bin/env python3
"""Price w4+ffn width-prune compositions on the measured Qwen3-8B cells.

Anchors (measured e2e, fixed chain phi=psi=0):
  R implied per cell from speedup = tau_e2e / (gamma*R + 1).
Candidates priced by term edit: ffn removes frac f of weight bytes
(pruned channels are also int4-quantized, so the edit multiplies the
weight-stream share s of R by (1-0.783*f); 0.783 = MLP share of layer
bytes). s is the uncertain term: weight-bound at b1, mixed at batch.
beta for combos MEASURED (beta_87.csv w4ffn*) x win512 factor .975
(the measured w4->w4win beta ratio on Q3-8B refs).
LCB05/P(win) via MC as map v6.
"""
import csv
import random
from pathlib import Path

random.seed(85)
PHASE = Path(__file__).resolve().parents[1]
N_MC = 2000
MLP_SHARE = 0.783
WINBETA = 0.975

# cell -> (gamma, tau_e2e, speedup) from research/86 e2e (w4win arms)
ANCHORS = {
    (1, 16): (4, 4.49, 1.42),
    (8, 16): (4, 4.51, 1.81),
    (16, 16): (6, 5.69, 1.79),
}
# measured competitors: name -> cell -> (tau, speedup)
MEASURED = {
    "w4win": {(1, 16): (4.49, 1.42), (8, 16): (4.51, 1.81),
              (16, 16): (5.69, 1.79)},
    "w4a8win_humming": {(8, 16): (4.58, 1.90), (16, 16): (6.02, 2.19)},
}
# weight-stream share of draft step time (mu, sigma) per batch
WSHARE = {1: (0.70, 0.12), 8: (0.50, 0.18), 16: (0.45, 0.18)}


def tau(beta, g):
    return (1 - beta ** (g + 1)) / (1 - beta) if beta < 1 else g + 1


def betas_from_csv():
    out = {}
    with (PHASE / "data/beta_87.csv").open() as f:
        for r in csv.DictReader(f):
            out[r["arm"]] = float(r["beta_greedy"])
    return out


def main():
    b87 = betas_from_csv()
    combos = []  # (name, beta_win, ffn_frac)
    for arm, frac in [("w4ffn125", 0.125), ("w4ffn25", 0.25),
                      ("w4ffn375", 0.375)]:
        if arm in b87:
            combos.append((arm + "win", b87[arm] * WINBETA, frac))
    print(f"{'cell':>8} | {'winner':<18} {'LCB05':>6} {'med':>6} {'P(win)':>6}"
          f" | runner-up")
    lines = []
    for cell, (g_anchor, tau_a, sp_a) in ANCHORS.items():
        b, ck = cell
        R_imp = (tau_a / sp_a - 1) / g_anchor
        cands = []
        for name, mcells in MEASURED.items():
            if cell in mcells:
                t, s = mcells[cell]
                cands.append((name, None, None, s, 0.02))  # measured e2e
        for name, beta, frac in combos:
            cands.append((name, beta, frac, None, None))
        stats = {n: [] for n, *_ in cands}
        wins = {n: 0 for n, *_ in cands}
        for _ in range(N_MC):
            best_v, best_n = 0.0, None
            smp = max(0.05, min(0.95, random.gauss(*WSHARE[b])))
            for name, beta, frac, sp_meas, ssig in cands:
                if sp_meas is not None:
                    v = random.gauss(sp_meas, ssig * sp_meas)
                else:
                    R = R_imp * (1 - smp * MLP_SHARE * frac) \
                        * random.gauss(1.0, 0.06)
                    bs = min(0.995, max(0.05, random.gauss(beta, 0.012)))
                    v = max(tau(bs, g) / (g * R + 1) for g in range(1, 9))
                stats[name].append(v)
                if v > best_v:
                    best_v, best_n = v, name
            wins[best_n] += 1
        rows = []
        for name, *_ in cands:
            vs = sorted(stats[name])
            rows.append((vs[int(0.05 * N_MC)], vs[N_MC // 2],
                         wins[name] / N_MC, name))
        rows.sort(reverse=True)
        w = rows[0]
        ru = rows[1] if len(rows) > 1 else None
        line = (f"b{b}/{ck}k | {w[3]:<18} {w[0]:6.2f} {w[1]:6.2f} {w[2]:6.2f}"
                + (f" | {ru[3]} LCB {ru[0]:.2f} P {ru[2]:.2f}" if ru else ""))
        print(line)
        lines.append("  " + line)
        for r in rows[2:]:
            print(f"         :  {r[3]:<18} {r[0]:6.2f} {r[1]:6.2f} {r[2]:6.2f}")
    return 0


if __name__ == "__main__":
    main()
