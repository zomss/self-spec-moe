"""Phase 66: arm table from data/*.json + fixed/marginal cycle solve.

Prints tok/s (accept_len) per batch vs the Phase-64 no-spec references and,
where a K=2/K=4 pair exists at the same batch, solves
cycle_ms = F + K*D with cycle_ms(K, b) = batch * accept_len / tok_s * 1000.
Usage: .venv/bin/python analyze.py
"""

import glob
import json
import os

DATA = os.path.join(os.path.dirname(__file__), os.pardir, "data")

# Phase-64 no-spec references (results_window_e2e.md); P65 b6.
NOSPEC = {6: 287.6, 8: 386.8, 12: 395.7, 32: 379.0}

ARMS = [
    ("A ep bf16", "w72n_q30b_p66_aep*_spec_cg_K{k}.json"),
    ("B fp8 replica", "w72n_q30b_p66_brep*_spec_cg_K{k}.json"),
    ("C node-local", "w72n_q30b_p66_cnl*_spec_cg_K{k}.json"),
]


def load(pattern):
    by_batch = {}
    for path in sorted(glob.glob(os.path.join(DATA, pattern))):
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
    return batch * al / tps * 1000.0


def main():
    batches = sorted(NOSPEC)
    hdr = " | ".join(f"b{b} tok/s (accept) | x" for b in batches)
    print(f"| arm | K | {hdr} |")
    print("|---|---|" + "---|---|" * len(batches))
    solves = []
    for label, pat in ARMS:
        pts = {}  # (k, batch) -> cycle
        for k in (2, 4):
            bb = load(pat.format(k=k))
            if not bb:
                continue
            cells = []
            for b in batches:
                r = bb.get(b)
                if not r or "err" in r:
                    cells += ["--", "--"]
                    continue
                al = f" ({r['al']:.3f})" if r.get("al") else ""
                sus = "*" if r.get("suspect") else ""
                cells.append(f"{r['tps']:.1f}+-{r['std']:.1f}{al}{sus}")
                ref = NOSPEC.get(b)
                cells.append(f"{r['tps'] / ref:.2f}x" if ref else "--")
                if r.get("al"):
                    pts[(k, b)] = cycle_ms(r["tps"], r["al"], b)
            print(f"| {label} | K={k} | " + " | ".join(cells) + " |")
        for b in batches:
            if (2, b) in pts and (4, b) in pts:
                d = (pts[(4, b)] - pts[(2, b)]) / 2.0
                f = pts[(2, b)] - 2 * d
                solves.append((label, b, f, d, pts[(2, b)], pts[(4, b)]))
    if solves:
        print("\n| arm | batch | cycle K2 ms | cycle K4 ms | fixed F ms "
              "| marginal D ms/step |")
        print("|---|---|---|---|---|---|")
        for label, b, f, d, c2, c4 in solves:
            print(f"| {label} | b{b} | {c2:.1f} | {c4:.1f} | {f:.1f} | {d:.1f} |")
    print("\n(no-spec refs P64/P65: " +
          ", ".join(f"b{b} {v}" for b, v in sorted(NOSPEC.items())) + ")")


if __name__ == "__main__":
    main()
