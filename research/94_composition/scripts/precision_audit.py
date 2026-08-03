#!/usr/bin/env python3
"""Is the search's reported regret resolvable, or is it content noise?

Regret is (S_oracle_best - S_pick)/S_oracle_best. When the strategy
picks the oracle's OWN argmax the two are the same measurement, so the
term is exactly 0 and carries no noise at all. Only cells where the
pick differs from the argmax can be corrupted by a content draw.

This splits every strategy's regret into those two parts, so the
headline number can be stated with the share that is structurally
exact.
"""
import importlib.util
import json
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P))
spec = importlib.util.spec_from_file_location("bl", P / "baselines.py")
bl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bl)

OUT = Path("/data/smcho/self-spec-moe/paper/data")


def main():
    report = {}
    for arch in ("dense", "llama"):
        cells, ar = bl.load(arch)
        if not cells:
            continue
        orc = bl.oracle(cells, ar) if hasattr(bl, "oracle") else None
        # oracle best per cell, straight from the measured arms
        orc = {}
        for cell, arms in cells.items():
            orc[cell] = max(((t / ar[cell], n, K)
                             for (n, K), (t, a) in arms.items()), default=None)
        scorers = bl.make_scorers()
        rows = {}
        for name, fn in scorers.items():
            found = bl.rank_and_pick(cells, ar, fn, 5)
            exact = tot = 0
            noisy_terms = []
            for cell, o in orc.items():
                f = found.get(cell)
                if not f:
                    continue
                tot += 1
                same = (f[1], f[2]) == (o[1], o[2])
                if same:
                    exact += 1
                else:
                    lbl = cell if isinstance(cell, str) else \
                        f"b{cell[0]}/c{cell[1]}"
                    noisy_terms.append((lbl, (o[0] - f[0]) / o[0] * 100,
                                        f"{f[1]}-K{f[2]}", f"{o[1]}-K{o[2]}"))
            mean = sum(t[1] for t in noisy_terms) / tot if tot else 0
            rows[name] = {"cells": tot, "exact_argmax": exact,
                          "mean_regret_pct": round(mean, 3),
                          "noisy_cells": [
                              {"cell": c, "regret_pct": round(r, 2),
                               "picked": p, "oracle": o}
                              for c, r, p, o in noisy_terms]}
        report[arch] = rows
        print(f"\n=== {arch}: regret decomposition at confirm-5")
        print(f"  {'strategy':16s} {'cells':>5s} {'exact argmax':>13s} "
              f"{'mean regret':>12s}  {'noise-exposed cells'}")
        for name, r in sorted(rows.items(), key=lambda kv: kv[1]["mean_regret_pct"]):
            ne = len(r["noisy_cells"])
            print(f"  {name:16s} {r['cells']:5d} {r['exact_argmax']:>8d}/{r['cells']:<4d} "
                  f"{r['mean_regret_pct']:11.2f}%  {ne} "
                  f"({', '.join(n['cell'] for n in r['noisy_cells']) or '-'})")
    OUT.mkdir(exist_ok=True)
    (OUT / "c2_precision_audit.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT}/c2_precision_audit.json")


if __name__ == "__main__":
    main()
