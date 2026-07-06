"""Phase 64: build the tok/s (accept_len) x {b8,b32} table from data/*.json.

Usage: .venv/bin/python analyze.py
"""

import json
import os

DATA = os.path.join(os.path.dirname(__file__), os.pardir, "data")

ARMS = [
    ("nospec", "w72n_q30b_p64_nospec_nospec_cg_nospec.json", "no-spec"),
    ("w512k2", "w72n_q30b_p64_w512_spec_cg_K2.json", "W512 K=2"),
    ("w512k4", "w72n_q30b_p64_w512_spec_cg_K4.json", "W512 K=4"),
    ("w256k4", "w72n_q30b_p64_w256_spec_cg_K4.json", "W256 K=4"),
    ("eagle_k1", "w72n_q30b_p64_eagle_spec_cg_K1.json", "EAGLE3 K=1"),
]


def main():
    rows = {}
    for arm, fname, label in ARMS:
        path = os.path.join(DATA, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        by_batch = {}
        for r in d.get("results") or []:
            if "error" in r:
                by_batch[r.get("batch")] = dict(err=r["error"])
                continue
            by_batch[r["batch"]] = dict(
                tps=r["tok_s_mean"], std=r["tok_s_std"],
                al=r.get("accept_len"), suspect=r.get("suspect"),
            )
        rows[arm] = (label, by_batch)

    base = rows.get("nospec", (None, {}))[1]
    batches = sorted({b for _, bb in rows.values() for b in bb})
    hdr = "| arm |" + "".join(
        f" b{b} tok/s (accept) | vs nospec |" for b in batches
    )
    print(hdr)
    print("|---|" + "---|---|" * len(batches))
    for arm, fname, label in ARMS:
        if arm not in rows:
            continue
        label, bb = rows[arm]
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
            if arm != "nospec" and nb and "err" not in nb and nb["tps"]:
                cells.append(f"{r['tps'] / nb['tps']:.2f}x")
            else:
                cells.append("--")
        print(f"| {label} | " + " | ".join(cells) + " |")
    print("\n(*) = harness 'suspect' flag (noisy slope or tiny decode delta).")


if __name__ == "__main__":
    main()
