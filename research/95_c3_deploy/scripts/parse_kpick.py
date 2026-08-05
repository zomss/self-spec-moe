#!/usr/bin/env python3
"""Parse [kpick] per-step decisions and test the R8 gate-failure hypotheses.

  [kpick] step=N n_run=N cell=bB/cC f=F K=K

Reports the armed duty, the live-f trajectory, and -- the discriminator --
how often the batch band (n_run.bit_length()) changes, since a band change
clears every per-request accept-EMA back to optimistic 1.0
(scheduler.py:1306-1308) and re-arms the policy.
"""
import re
import sys
from collections import Counter
from pathlib import Path

PAT = re.compile(
    r"\[kpick\] step=(\d+) n_run=(\d+) cell=b(\S+?)/c(\S+?) f=([\d.]+) K=(\d+)")


def main(path):
    rows = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = PAT.search(line)
        if m:
            rows.append((int(m.group(1)), int(m.group(2)), m.group(3),
                         m.group(4), float(m.group(5)), int(m.group(6))))
    if not rows:
        print(f"no [kpick] lines in {path}")
        return
    n = len(rows)
    kdist = Counter(r[5] for r in rows)
    cells = Counter(f"b{r[2]}/c{r[3]}" for r in rows)
    armed = sum(1 for r in rows if r[5] > 0)
    bands = [r[1].bit_length() for r in rows]
    band_changes = sum(1 for a, b in zip(bands, bands[1:]) if a != b)
    nrun = Counter(r[1] for r in rows)

    print(f"steps: {n}")
    print(f"armed duty: {armed}/{n} = {armed/n*100:.1f}%   K distribution: "
          + ", ".join(f"K{k}={c} ({c/n*100:.1f}%)" for k, c in sorted(kdist.items())))
    print(f"cells visited: {dict(cells.most_common(4))}")
    print(f"n_run values: {dict(nrun.most_common(6))}")
    print(f"batch band changes: {band_changes} over {n} steps "
          f"({band_changes/n*100:.1f}% of steps)  bands seen: "
          f"{sorted(set(bands))}")

    f = [r[4] for r in rows]
    q = lambda p: sorted(f)[int(p * (len(f) - 1))]          # noqa: E731
    print(f"live f: min {min(f):.3f}  p25 {q(.25):.3f}  median {q(.5):.3f}  "
          f"p75 {q(.75):.3f}  max {max(f):.3f}")
    print(f"  fraction of steps with f >= 0.95 (near-optimistic): "
          f"{sum(1 for x in f if x >= 0.95)/n*100:.1f}%")

    # first 40 steps and a mid-run window, to see reset-rearm cycling
    print("\n  step  n_run band     f   K")
    for r in rows[:15] + [None] + rows[n // 2:n // 2 + 15]:
        if r is None:
            print("  ...")
            continue
        print(f"  {r[0]:5d} {r[1]:5d} {r[1].bit_length():5d} {r[4]:6.3f} {r[5]:3d}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "research/95_c3_deploy/logs/diag_r8.log")
