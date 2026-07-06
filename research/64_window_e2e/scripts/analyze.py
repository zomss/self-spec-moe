"""Phase 64: build the tok/s (accept_len) x batch table from data/*.json.

Split invocations of an arm (resident points vs the over-pool _b32 launch)
are merged per arm. Usage: .venv/bin/python analyze.py
"""

import json
import os

DATA = os.path.join(os.path.dirname(__file__), os.pardir, "data")

ARMS = [
    ("no-spec", [
        "w72n_q30b_p64_nospec_nospec_cg_nospec.json",
        "w72n_q30b_p64_nospec_b12_nospec_cg_nospec.json",
    ]),
    ("W512 K=2", [
        "w72n_q30b_p64_w512_spec_cg_K2.json",
        "w72n_q30b_p64_w512_b32_spec_cg_K2.json",
    ]),
    ("W512 K=4", [
        "w72n_q30b_p64_w512_spec_cg_K4.json",
        "w72n_q30b_p64_w512_b32_spec_cg_K4.json",
    ]),
    ("W256 K=4", [
        "w72n_q30b_p64_w256_spec_cg_K4.json",
        "w72n_q30b_p64_w256_b32_spec_cg_K4.json",
    ]),
    ("EAGLE3 K=1", [
        "w72n_q30b_p64_eagle_spec_cg_K1.json",
    ]),
]


def load_arm(fnames):
    by_batch = {}
    for fname in fnames:
        path = os.path.join(DATA, fname)
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


def main():
    rows = [(label, load_arm(fnames)) for label, fnames in ARMS]
    rows = [(label, bb) for label, bb in rows if bb]
    base = dict(rows).get("no-spec", {})
    batches = sorted({b for _, bb in rows for b in bb})
    print("| arm |" + "".join(
        f" b{b} tok/s (accept) | vs nospec |" for b in batches))
    print("|---|" + "---|---|" * len(batches))
    for label, bb in rows:
        cells = []
        for b in batches:
            r = bb.get(b)
            if not r or "err" in r:
                cells += [(r or {}).get("err", "--")[:24], "--"]
                continue
            al = f" ({r['al']:.3f})" if r.get("al") else ""
            sus = "*" if r.get("suspect") else ""
            cells.append(f"{r['tps']:.1f}+-{r['std']:.1f}{al}{sus}")
            nb = base.get(b)
            if label != "no-spec" and nb and "err" not in nb and nb.get("tps"):
                cells.append(f"{r['tps'] / nb['tps']:.2f}x")
            else:
                cells.append("--")
        print(f"| {label} | " + " | ".join(cells) + " |")
    print("\n(*) = harness 'suspect' flag (noisy slope or tiny decode delta).")


if __name__ == "__main__":
    main()
