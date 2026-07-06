"""Phase 65: stage table from data/*.json + fixed/marginal cycle solve.

For each stage (base = Phase 64 references, f1, f12, f123) prints
tok/s (accept_len) per batch vs the no-spec references, and per stage
solves cycle_ms = F + K*D from the K=2 / K=4 points at that stage's
solve batch (b8 for base/f1/f12; b6 for f123, its largest resident
batch under the fp8-replica pool):
  cycle_ms(K, b) = batch * accept_len / tok_s * 1000
Usage: .venv/bin/python analyze.py
"""

import json
import os

DATA = os.path.join(os.path.dirname(__file__), os.pardir, "data")
P64DATA = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "64_window_e2e", "data"
)

# Phase-64 references (results_window_e2e.md); b6 measured this phase
# (filled from the nospec b6 arm when present).
NOSPEC = {8: 386.8, 12: 395.7, 32: 379.0}
_ns_b6 = os.path.join(DATA, "w72n_q30b_p65_f12_nospec_b6_nospec_cg_nospec.json")


def load(fnames):
    by_batch = {}
    for path in fnames:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        for r in d.get("results") or []:
            if "error" in r:
                by_batch[r.get("batch")] = dict(err=r["error"])
                continue
            by_batch[r["batch"]] = dict(
                tps=r["tok_s_mean"], std=r["tok_s_std"],
                al=r.get("accept_len"), suspect=r.get("suspect"),
            )
    return by_batch


ns6 = load([_ns_b6]).get(6)
if ns6:
    NOSPEC[6] = ns6["tps"]

STAGES = [
    ("base (P64)", 8, {
        2: [os.path.join(P64DATA, "w72n_q30b_p64_w512_spec_cg_K2.json")],
        4: [os.path.join(P64DATA, "w72n_q30b_p64_w512_spec_cg_K4.json")],
    }),
]
for st, solve_b in (("f1", 8), ("f12", 8), ("f123", 6), ("f123nl", 6)):
    STAGES.append((st, solve_b, {
        2: [os.path.join(DATA, f"w72n_q30b_p65_{st}_w512_spec_cg_K2.json"),
            os.path.join(DATA, f"w72n_q30b_p65_{st}_w512_b6_spec_cg_K2.json")],
        4: [os.path.join(DATA, f"w72n_q30b_p65_{st}_w512_spec_cg_K4.json"),
            os.path.join(DATA, f"w72n_q30b_p65_{st}_w512_b6_spec_cg_K4.json")],
    }))
STAGES.append(("f12 @b6", 6, {
    2: [os.path.join(DATA, "w72n_q30b_p65_f12_w512_b6_spec_cg_K2.json")],
    4: [os.path.join(DATA, "w72n_q30b_p65_f12_w512_b6_spec_cg_K4.json")],
}))
STAGES.append(("f123+skip", 6, {
    2: [os.path.join(DATA, "w72n_q30b_p65_f123_w512_b6s_spec_cg_K2.json")],
    4: [os.path.join(DATA, "w72n_q30b_p65_f123_w512_b6s_spec_cg_K4.json")],
}))


def cycle_ms(tps, al, batch):
    # per cycle the engine emits batch*accept_len tokens at tps tok/s.
    return batch * al / tps * 1000.0


def main():
    print("| stage | K | b6 tok/s (accept) | x | b8 tok/s (accept) | x "
          "| b12 tok/s (accept) | x |")
    print("|---|---|" + "---|---|" * 3)
    solves = []
    for label, solve_b, by_k in STAGES:
        pts = {}
        printed = False
        for k, fnames in sorted(by_k.items()):
            bb = load(fnames)
            if not bb:
                continue
            printed = True
            cells = []
            for b in (6, 8, 12):
                r = bb.get(b)
                if not r or "err" in r:
                    cells += ["--", "--"]
                    continue
                al = f" ({r['al']:.3f})" if r.get("al") else ""
                sus = "*" if r.get("suspect") else ""
                cells.append(f"{r['tps']:.1f}+-{r['std']:.1f}{al}{sus}")
                ref = NOSPEC.get(b)
                cells.append(f"{r['tps'] / ref:.2f}x" if ref else "--")
                if b == solve_b and r.get("al"):
                    pts[k] = cycle_ms(r["tps"], r["al"], b)
            print(f"| {label} | K={k} | " + " | ".join(cells) + " |")
        if printed and 2 in pts and 4 in pts:
            d = (pts[4] - pts[2]) / 2.0
            f = pts[2] - 2 * d
            solves.append((label, solve_b, f, d, pts[2], pts[4]))
    if solves:
        print("\n| stage | solve batch | cycle K2 ms | cycle K4 ms | fixed F ms "
              "| marginal D ms/draft-step |")
        print("|---|---|---|---|---|---|")
        for label, sb, f, d, c2, c4 in solves:
            print(f"| {label} | b{sb} | {c2:.1f} | {c4:.1f} | {f:.1f} "
                  f"| {d:.1f} |")
        print("\n(no-spec refs: b8 386.8 = 20.7 ms/step, b12 395.7; "
              f"b6 {NOSPEC.get(6, 'TBD')}; verify ~ a no-spec step)")


if __name__ == "__main__":
    main()
