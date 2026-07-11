#!/usr/bin/env python3
"""E1 analyzer: TPOT tables + lever ratios R = TPOT(arm)/TPOT(denominator).

Reads the bench-serve JSONs written by e1_sweep.sh
(<group>_<arm>_b<B>_c<C>k_r<r>.json), aggregates runs, ratios each arm against
its matched denominator (L0; dummy-L0 for the dummy-load skip arms; the
FLASHMLA-pinned bf16 for MLA fp8-KV -- E0 findings), and runs the
pre-registered sanity gates:

  S1 cross-check: dense bf16 b1/2k TPOT ~= P75-E1 slope (5.86 ms, +-15%)
  S2 window functional: win/L0 < 0.97 at 32k, else the window is a silent
     no-op at kernel level (config plumbing alone proved nothing -- E0)
  S3 skip50 linearity: skip50/L0dummy in [0.42, 0.68] at b32/2k
  S4 run spread: max|run-mean|/mean > 5% -> flag cell as noisy

Outputs: markdown tables to stdout + data/e1/summary.csv.
"""

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

FNAME = re.compile(r"^(?P<group>[a-z]+)_(?P<arm>[a-z0-9_]+)_b(?P<b>\d+)_c(?P<c>\d+)k_r(?P<r>\d+)\.json$")

DENOM = {
    "dense": {"_default": "d_bf16", "d_skip50": "d_bf16dummy", "d_skip25": "d_bf16dummy"},
    "moe": {"_default": "m_bf16", "m_skip50": "m_bf16dummy"},
    "mla": {"_default": "ds_bf16", "ds_skip50": "ds_bf16dummy", "ds_kvq": "ds_bf16fmla"},
    "pcie": {"_default": "p_bf16", "p_skip50": "p_bf16dummy"},
}
P75_E1_BF16_MS = 5.86  # dense bf16 b1/2k decode-step, P75 E1 slope (cross-check)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--metric", default="mean_tpot_ms",
                    choices=["mean_tpot_ms", "median_tpot_ms"])
    a = ap.parse_args()
    d = Path(a.data)

    # cells[group][arm][(b, c)] = [tpot_run1, tpot_run2, ...]
    cells: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for f in sorted(d.glob("*.json")):
        m = FNAME.match(f.name)
        if not m:
            continue
        try:
            j = json.loads(f.read_text())
            tpot = float(j[a.metric])
        except Exception as e:
            print(f"!! unreadable {f.name}: {e}")
            continue
        if tpot <= 0 or j.get("completed", 1) == 0:
            print(f"!! empty/failed result {f.name} (tpot={tpot})")
            continue
        cells[m["group"]][m["arm"]][(int(m["b"]), int(m["c"]))].append(tpot)

    if not cells:
        print(f"no results under {d}")
        return 1

    # over-pool cells from runner sidecars: "PREEMPT b<B>_c<C>k <N>" in *.meta
    overpool: dict = defaultdict(set)
    for f in d.glob("*.meta"):
        try:
            group, arm = f.stem.split("_", 1)
        except ValueError:
            continue
        for line in f.read_text(errors="replace").splitlines():
            m = re.match(r"(?:OVERCAP|PREEMPT) b(\d+)_c(\d+)k (\d+)", line)
            if m:
                overpool[(group, arm)].add((int(m[1]), int(m[2])))

    flags: list[str] = []
    rows_csv: list[dict] = []
    for group in cells:
        arms = cells[group]
        cols = sorted({cell for arm in arms.values() for cell in arm})
        hdr = ["arm"] + [f"b{b}/c{c}k" for b, c in cols]

        def table(title, fmt):
            print(f"\n## {group}: {title}\n")
            print("| " + " | ".join(hdr) + " |")
            print("|" + "|".join("---" for _ in hdr) + "|")
            for arm in arms:
                cellsvals = []
                for cell in cols:
                    runs = arms[arm].get(cell, [])
                    cellsvals.append(fmt(arm, cell, runs) if runs else "-")
                print("| " + " | ".join([arm] + cellsvals) + " |")

        def agg(runs):
            # median-of-runs: DP4 serving shows bimodal per-run TPOT (request
            # placement across ranks + prefix-cache-aware routing clumping);
            # the median rejects placement-outlier runs, the mean does not.
            return statistics.median(runs)

        def denom_of(arm):
            g = DENOM.get(group, {"_default": None})
            return g.get(arm, g["_default"])

        # S4 spread / S5 over-pool + TPOT table
        def fmt_tpot(arm, cell, runs):
            m = agg(runs)
            if cell in overpool.get((group, arm), ()):
                flags.append(f"S5 OVER-POOL {group}/{arm} b{cell[0]}/c{cell[1]}k (preempted)")
                return f"{m:.2f}!"
            if len(runs) > 1 and max(abs(r - m) for r in runs) / m > 0.05:
                flags.append(f"S4 NOISY {group}/{arm} b{cell[0]}/c{cell[1]}k runs={runs}")
                return f"{m:.2f}*"
            return f"{m:.2f}"

        def fmt_ratio(arm, cell, runs):
            dn = denom_of(arm)
            if dn == arm or dn is None:
                return "1.000"
            druns = arms.get(dn, {}).get(cell, [])
            if not druns:
                return "?"
            bad = (cell in overpool.get((group, arm), ())
                   or cell in overpool.get((group, dn), ()))
            r = agg(runs) / agg(druns)
            rows_csv.append(dict(group=group, arm=arm, batch=cell[0], ctx=cell[1],
                                 tpot_ms=round(agg(runs), 3), runs=len(runs),
                                 denom=dn, ratio=round(r, 4), overpool=int(bad)))
            return f"{r:.3f}!" if bad else f"{r:.3f}"

        table("TPOT ms (mean over runs; * = >5% run spread)", fmt_tpot)
        table("ratio R = arm / denominator (README predictions)", fmt_ratio)

        # sanity gates
        if group == "dense":
            runs = arms.get("d_bf16", {}).get((1, 2), [])
            if runs:
                v = agg(runs)
                if abs(v - P75_E1_BF16_MS) / P75_E1_BF16_MS > 0.15:
                    flags.append(f"S1 CROSS-CHECK FAIL dense bf16 b1/2k {v:.2f}ms vs P75 {P75_E1_BF16_MS}ms")
                else:
                    flags.append(f"S1 cross-check OK: {v:.2f}ms vs P75 {P75_E1_BF16_MS}ms")
        for win, l0 in (("d_win", "d_bf16"), ("m_win", "m_bf16"), ("ds_win", "ds_bf16"),
                        ("p_win", "p_bf16")):
            if win in arms and l0 in arms:
                r32 = [(agg(arms[win][c]) / agg(arms[l0][c]))
                       for c in arms[win] if c[1] == 32 and c in arms[l0]]
                if r32 and min(r32) > 0.97:
                    flags.append(f"S2 WINDOW NO-OP? {group}/{win} best 32k ratio {min(r32):.3f} "
                                 "(config plumbed but kernel unaffected)")
        for sk, dn in (("d_skip50", "d_bf16dummy"), ("m_skip50", "m_bf16dummy"),
                       ("ds_skip50", "ds_bf16dummy")):
            if sk in arms and dn in arms:
                c = (32, 2)
                if c in arms[sk] and c in arms[dn]:
                    r = agg(arms[sk][c]) / agg(arms[dn][c])
                    if not 0.42 <= r <= 0.68:
                        flags.append(f"S3 SKIP NONLINEAR {group}/{sk} b32/2k ratio {r:.3f} (want ~0.5)")

    print("\n## sanity gates\n")
    for fl in flags or ["(none)"]:
        print(f"- {fl}")

    out = d / "summary.csv"
    if rows_csv:
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_csv[0].keys()))
            w.writeheader()
            w.writerows(rows_csv)
        print(f"\nwrote {out} ({len(rows_csv)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
