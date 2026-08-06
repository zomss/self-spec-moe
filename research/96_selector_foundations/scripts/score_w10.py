#!/usr/bin/env python3
"""W10 scorer: MoE Stage-B surface under equal work (w10_moe_remeasure.md).

Per cell: best S over the four spec arms vs the off anchor, with
2nd-highest-round episode rejection. Compares against the Stage-B
(natural-EOS, tune-on) values and reports P-W10a..d.
"""
import json
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
W10 = PHASE / "data" / "w10"
SB = PHASE.parent / "93_c1_grid/data"
ARMS = ["w4a16_k2", "w4a16_k3", "win2048_k3", "win8192_k3"]
# W9b screen: Stage-B cells flagged contaminated (drain + arms diverged)
FLAGGED_WINS = {("R6", 32), ("R1", 64), ("R2", 32), ("R8", 8)}
FLAGGED_LOSSES = {("R6", 64), ("R8", 64), ("R5", 8), ("R5cot", 8), ("R2", 8)}


def rounds(path):
    d = json.load(open(path))
    return {(c["rid"], c["batch"]): c for c in d["cells"] if "toks" in c}


def robust(rs):
    """2nd-highest anchor, reject rounds >5% below, need >=2 survivors."""
    if not rs:
        return None
    s = sorted(rs, reverse=True)
    ref = s[1] if len(s) > 1 else s[0]
    keep = [r for r in rs if r >= 0.95 * ref]
    if len(keep) < 2:
        return None
    return sum(keep) / len(keep)


def main():
    off = rounds(W10 / "w10_moe_off.json")
    spec = {}
    for a in ARMS:
        p = W10 / f"w10_moe_{a}.json"
        if p.exists():
            spec[a] = rounds(p)
    sb_off = rounds(SB / "stageb_moe_off.json")
    sb = {a: rounds(SB / f"stageb_moe_{a}.json") for a in ARMS
          if (SB / f"stageb_moe_{a}.json").exists()}

    rows, wins, flips, lost = [], 0, [], []
    for key in sorted(off, key=lambda k: (k[0], k[1])):
        o = robust(off[key]["all"])
        if not o:
            continue
        best, best_arm = None, None
        for a, cells in spec.items():
            if key not in cells:
                continue
            s = robust(cells[key]["all"])
            if s and (best is None or s / o > best):
                best, best_arm = s / o, a
        if best is None:
            continue
        # Stage-B comparison
        sb_best = None
        if key in sb_off:
            for a, cells in sb.items():
                if key in cells and cells[key]["toks"] and sb_off[key]["toks"]:
                    v = cells[key]["toks"] / sb_off[key]["toks"]
                    sb_best = v if sb_best is None else max(sb_best, v)
        row = {"rid": key[0], "batch": key[1], "S": round(best, 4),
               "arm": best_arm, "S_stageB": round(sb_best, 4) if sb_best else None,
               "flagged": key in FLAGGED_WINS or key in FLAGGED_LOSSES}
        if best > 1.0:
            wins += 1
        if sb_best:
            row["delta_pct"] = round(100 * (best - sb_best) / sb_best, 1)
            if sb_best <= 1.0 < best:
                flips.append(row)
            if best <= 1.0 < sb_best:
                lost.append(row)
        rows.append(row)

    print(f"{'cell':<12}{'S(W10)':>9}{'arm':>13}{'S(StageB)':>11}"
          f"{'delta%':>9}  flag")
    for r in rows:
        mark = "WIN" if r["S"] > 1.0 else ""
        print(f"{r['rid']:>5} b{r['batch']:<5}{r['S']:>9.3f}"
              f"{str(r['arm']):>13}"
              f"{r['S_stageB'] if r['S_stageB'] else float('nan'):>11.3f}"
              f"{r.get('delta_pct', float('nan')):>9.1f}"
              f"  {'FLAGGED' if r['flagged'] else '':<8}{mark}")

    sb_wins = sum(1 for r in rows if r["S_stageB"] and r["S_stageB"] > 1.0)
    print(f"\nP-W10a  win count: Stage-B {sb_wins}/{len(rows)} -> "
          f"W10 {wins}/{len(rows)}   (predicted 2-5)")
    r6 = next((r for r in rows if (r["rid"], r["batch"]) == ("R6", 32)), None)
    if r6:
        print(f"P-W10b  R6 b32: W10 {r6['S']:.3f} vs Stage-B "
              f"{r6['S_stageB']} (W7 measured 0.857) -> "
              f"{'CONFIRMED' if r6['S'] < 1.0 else 'REFUTED'}")
    print(f"P-W10c  losses that flipped UP: {len(flips)} "
          f"{[(f['rid'], f['batch']) for f in flips]} -> "
          f"{'CONFIRMED' if flips else 'REFUTED (artifact one-directional)'}")
    print(f"        wins that fell: {len(lost)} "
          f"{[(f['rid'], f['batch']) for f in lost]}")
    hb = [r for r in rows if r["S"] > 1.0 and r["batch"] >= 32]
    print(f"P-W10d  surviving wins at b>=32: {len(hb)}/{wins}")
    out = {"rows": rows, "wins_w10": wins, "wins_stageB": sb_wins,
           "flips_up": flips, "wins_lost": lost}
    (W10 / "w10_scored.json").write_text(json.dumps(out, indent=1))
    print("\nsaved ->", W10 / "w10_scored.json")


if __name__ == "__main__":
    main()
