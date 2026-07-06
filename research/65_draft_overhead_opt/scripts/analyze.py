"""Phase 65: stage table from data/*.json + fixed/marginal cycle solve.

For each stage (base = Phase 64 references, f1, f12, f123) prints
tok/s (accept_len) per batch vs the Phase-64 no-spec references, and for
b8 solves cycle_ms = F + K*D from the K=2 / K=4 points:
  cycle_ms(K, b) = batch * accept_len / tok_s * 1000
Usage: .venv/bin/python analyze.py
"""

import json
import os

DATA = os.path.join(os.path.dirname(__file__), os.pardir, "data")
P64DATA = os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, "64_window_e2e", "data"
)

# Phase-64 references (results_window_e2e.md).
NOSPEC = {8: 386.8, 12: 395.7, 32: 379.0}

STAGES = [
    ("base (P64)", {
        2: [os.path.join(P64DATA, "w72n_q30b_p64_w512_spec_cg_K2.json")],
        4: [os.path.join(P64DATA, "w72n_q30b_p64_w512_spec_cg_K4.json")],
    }),
]
for st in ("f1", "f12", "f123", "f123nl"):
    STAGES.append((st, {
        2: [os.path.join(DATA, f"w72n_q30b_p65_{st}_w512_spec_cg_K2.json")],
        4: [os.path.join(DATA, f"w72n_q30b_p65_{st}_w512_spec_cg_K4.json")],
    }))


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


def cycle_ms(tps, al, batch):
    # per cycle the engine emits batch*accept_len tokens at tps tok/s.
    return batch * al / tps * 1000.0


def main():
    print("| stage | K | b8 tok/s (accept) | x | b12 tok/s (accept) | x |")
    print("|---|---|---|---|---|---|")
    solves = {}
    for label, by_k in STAGES:
        pts = {}
        for k, fnames in sorted(by_k.items()):
            bb = load(fnames)
            if not bb:
                continue
            cells = []
            for b in (8, 12):
                r = bb.get(b)
                if not r or "err" in r:
                    cells += ["--", "--"]
                    continue
                al = f" ({r['al']:.3f})" if r.get("al") else ""
                sus = "*" if r.get("suspect") else ""
                cells.append(f"{r['tps']:.1f}+-{r['std']:.1f}{al}{sus}")
                cells.append(f"{r['tps'] / NOSPEC[b]:.2f}x")
                if b == 8 and r.get("al"):
                    pts[k] = cycle_ms(r["tps"], r["al"], b)
            print(f"| {label} | K={k} | " + " | ".join(cells) + " |")
        if 2 in pts and 4 in pts:
            d = (pts[4] - pts[2]) / 2.0
            f = pts[2] - 2 * d
            solves[label] = (f, d, pts[2], pts[4])
    if solves:
        print("\n| stage | b8 cycle K2 ms | b8 cycle K4 ms | fixed F ms "
              "| marginal D ms/draft-step |")
        print("|---|---|---|---|---|")
        for label, (f, d, c2, c4) in solves.items():
            print(f"| {label} | {c2:.1f} | {c4:.1f} | {f:.1f} | {d:.1f} |")
        print("\n(no-spec refs: b8 386.8 = 20.7 ms/step, b12 395.7 = "
              "30.3 ms/step; verify ~ a no-spec step)")


if __name__ == "__main__":
    main()
