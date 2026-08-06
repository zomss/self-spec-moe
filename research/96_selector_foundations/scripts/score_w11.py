#!/usr/bin/env python3
"""W11 scorer: llama + q3_32b Stage-B surfaces under equal work.

Per cell: best S over the canonical spec arms vs the off anchor, with
2nd-highest-round episode rejection. Reports P-W11a..e, and isolates
the ANCHOR term on b1 cells (drain-immune by construction, so any b1
change is autotune/protocol, not drain).
"""
import json
import sys
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
W11 = PHASE / "data" / "w11"
SB = PHASE.parent / "93_c1_grid/data"
ARMS = {
    "llama": ["w4a16_k2", "w4a16_k4", "w8int8_k2", "win512_k2", "win2048_k4"],
    "q3_32b": ["w4gptq_k4", "w4gptq_k6", "win512_k4", "skipb2_k4"],
}


def cells(p):
    return {(c["rid"], c["batch"]): c
            for c in json.load(open(p))["cells"] if "toks" in c}


def robust(rs):
    s = sorted(rs, reverse=True)
    ref = s[1] if len(s) > 1 else s[0]
    keep = [r for r in rs if r >= 0.95 * ref]
    return sum(keep) / len(keep) if len(keep) >= 2 else None


def score(arch):
    arms = ARMS[arch]
    need = [W11 / f"w11_{arch}_{a}.json" for a in ["off"] + arms]
    missing = [p.name for p in need if not p.exists()]
    if missing:
        print(f"[{arch}] MISSING {missing}")
        return None
    off, sboff = cells(need[0]), cells(SB / f"stageb_{arch}_off.json")
    spec = {a: cells(W11 / f"w11_{arch}_{a}.json") for a in arms}
    sb = {a: cells(SB / f"stageb_{arch}_{a}.json") for a in arms
          if (SB / f"stageb_{arch}_{a}.json").exists()}

    rows, wins, sbwins, flips, fell, winners = [], 0, 0, [], [], set()
    for k in sorted(off, key=lambda x: (x[0], x[1])):
        o = robust(off[k]["all"])
        if not o:
            continue
        best = besta = None
        for a in arms:
            if k in spec[a]:
                s = robust(spec[a][k]["all"])
                if s and (best is None or s / o > best):
                    best, besta = s / o, a
        if best is None:
            continue
        sbb = None
        for a, c in sb.items():
            if k in c and k in sboff and sboff[k]["toks"]:
                v = c[k]["toks"] / sboff[k]["toks"]
                sbb = v if sbb is None else max(sbb, v)
        if best > 1.0:
            wins += 1
            winners.add(besta)
        if sbb and sbb > 1.0:
            sbwins += 1
        if sbb and sbb <= 1.0 < best:
            flips.append(k)
        if sbb and best <= 1.0 < sbb:
            fell.append(k)
        rows.append({"rid": k[0], "batch": k[1], "S": round(best, 4),
                     "arm": besta,
                     "S_stageB": round(sbb, 4) if sbb else None,
                     "delta_pct": round(100 * (best - sbb) / sbb, 1)
                     if sbb else None})
    # anchor isolation on b1 (drain-immune)
    anch = []
    for k in off:
        if k[1] == 1 and k in sboff and sboff[k]["toks"]:
            anch.append(100 * (off[k]["toks"] - sboff[k]["toks"])
                        / sboff[k]["toks"])
    res = {"arch": arch, "rows": rows, "wins": wins, "wins_stageB": sbwins,
           "winning_configs": sorted(winners), "flips_up": flips,
           "wins_fell": fell,
           "b1_anchor_shift_mean_pct": round(sum(anch) / len(anch), 2)
           if anch else None,
           "b1_anchor_shift_max_pct": round(max(anch, key=abs), 2)
           if anch else None}
    print(f"\n=== {arch}: Stage-B {sbwins}/{len(rows)} -> W11 {wins}/{len(rows)}")
    for r in rows:
        print(f"  {r['rid']:>5} b{r['batch']:<4}{r['S']:>8.3f}{r['arm']:>13}"
              f"{r['S_stageB'] if r['S_stageB'] else float('nan'):>8.3f}"
              f"{r['delta_pct'] if r['delta_pct'] is not None else float('nan'):>8.1f}%")
    print(f"  winning configs: {len(winners)} {sorted(winners)}")
    print(f"  flips up: {len(flips)} {flips}")
    print(f"  wins fell: {len(fell)} {fell}")
    print(f"  b1 anchor shift: mean {res['b1_anchor_shift_mean_pct']}% "
          f"max {res['b1_anchor_shift_max_pct']}%  "
          f"(b1 is drain-immune -> this is the autotune/protocol term)")
    return res


def main():
    out = {}
    for arch in (sys.argv[1:] or ["llama", "q3_32b"]):
        r = score(arch)
        if r:
            out[arch] = r
    if out:
        (W11 / "w11_scored.json").write_text(json.dumps(out, indent=1))
        print("\nsaved ->", W11 / "w11_scored.json")


if __name__ == "__main__":
    main()
