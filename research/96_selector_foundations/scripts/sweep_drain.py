#!/usr/bin/env python3
"""W9 follow-up: sweep prior phases for e2e verdicts contaminated by the
drain artifact (results_w9.md).

Contamination requires ALL of:
  (a) batch > 1            -- b1 cells run prompts SEQUENTIALLY in
                              run_grid.py, so there is no drain at all
  (b) requests finish at DIFFERENT times (clip_ratio < 1: not every
      request is ceiling-clipped) -- otherwise all lengths are equal
  (c) the two arms' length distributions DIVERGE (the actual W9
      signature: out_p95 differed 1332 vs 925 while p50 matched)
  (d) the verdict is near 1, where a tail artifact can flip it

(a)-(c) make a cell's e2e ratio unsound; (d) makes it consequential.
Cells failing (d) but meeting (a)-(c) are reported as "unsound but
verdict-robust" -- the ratio is wrong, the sign is not.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NEAR1_LO, NEAR1_HI = 0.90, 1.15
TAIL_TOL = 0.10          # |p95 ratio - 1| above this = arms diverged


def cells(p):
    try:
        d = json.load(open(p))
    except Exception:
        return None, {}
    out = {}
    for c in d.get("cells", []):
        if "toks" in c and "out_tok_p95" in c:
            out[(c.get("rid"), c.get("batch"))] = c
    return d, out


def scan(pairs, label):
    rows = []
    for offp, specs in pairs:
        _, oc = cells(offp)
        if not oc:
            continue
        for sp in specs:
            d, sc = cells(sp)
            if not sc:
                continue
            for key in sorted(set(oc) & set(sc), key=lambda k: (str(k[0]), k[1])):
                o, s = oc[key], sc[key]
                b = key[1]
                if not o["toks"] or not s["toks"]:
                    continue
                S = s["toks"] / o["toks"]
                p95o, p95s = o["out_tok_p95"], s["out_tok_p95"]
                tail = abs(p95s / p95o - 1) if p95o else 0.0
                drain = (b > 1 and (o["clip_ratio"] < 0.999
                                    or s["clip_ratio"] < 0.999))
                diverged = tail > TAIL_TOL
                near1 = NEAR1_LO <= S <= NEAR1_HI
                if not (drain and diverged):
                    continue
                rows.append({
                    "set": label, "spec": sp.name, "rid": key[0], "b": b,
                    "S": round(S, 4), "near1": near1,
                    "p95_off": p95o, "p95_spec": p95s,
                    "tail_gap_pct": round(100 * tail, 1),
                    "clip_off": o["clip_ratio"], "clip_spec": s["clip_ratio"],
                    "p50_off": o["out_tok_p50"], "p50_spec": s["out_tok_p50"],
                    "severity": "SUSPECT" if near1 else "unsound-but-robust"})
    return rows


def main():
    P93 = ROOT / "research/93_c1_grid/data"
    P95 = ROOT / "research/95_c3_deploy/data"
    P96 = ROOT / "research/96_selector_foundations/data"
    pairs, seen = [], set()
    for offp in sorted(P93.glob("stageb_*_off.json")):
        arch = offp.name[len("stageb_"):-len("_off.json")]
        sp = [p for p in sorted(P93.glob(f"stageb_{arch}_*.json"))
              if p != offp and "CORRUPT" not in p.name]
        if sp:
            pairs.append((offp, sp))
            seen.add(arch)
    grid = [(P93 / "grid_smoke_mla_plain.json",
             [p for p in sorted(P93.glob("grid_smoke_mla_*.json"))
              if "plain" not in p.name])]
    e3 = [(P95 / "e3_mla_off_s0.json", [P95 / "e3_mla_uncond_s0.json"])]
    w7 = [(P96 / "w7/w7_mla_off_lane0.json",
           [P96 / "w7/w7_mla_uncond_lane0.json"]),
          (P96 / "w7/w7_moe_off_boot1.json",
           [P96 / "w7/w7_moe_uncond_boot1.json"])]

    rows = (scan(pairs, "93/stageb") + scan(grid, "93/smoke")
            + scan(e3, "95/e3") + scan(w7, "96/w7"))
    rows.sort(key=lambda r: (r["severity"] != "SUSPECT",
                             abs(r["S"] - 1.0)))
    print(f"Screened Stage-B archs: {sorted(seen)}")
    print(f"Contaminated cells (drain + arms diverged): {len(rows)}\n")
    hdr = (f"{'set':<11}{'artifact':<34}{'rid':>5}{'b':>4}{'S':>8}"
           f"{'p95 off/spec':>16}{'gap%':>7}  verdict")
    print(hdr)
    for r in rows:
        print(f"{r['set']:<11}{r['spec'][:33]:<34}{str(r['rid']):>5}"
              f"{r['b']:>4}{r['S']:>8.3f}"
              f"{str(r['p95_off']) + '/' + str(r['p95_spec']):>16}"
              f"{r['tail_gap_pct']:>7.1f}  {r['severity']}")
    outp = ROOT / "research/96_selector_foundations/data/w9/drain_sweep.json"
    outp.write_text(json.dumps(rows, indent=1))
    n_sus = sum(1 for r in rows if r["severity"] == "SUSPECT")
    print(f"\nSUSPECT (near-1 AND contaminated): {n_sus}")
    print("saved ->", outp)


if __name__ == "__main__":
    main()
