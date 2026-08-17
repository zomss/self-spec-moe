#!/usr/bin/env python3
"""Campaign-1 scoring under barrier v2 (amendment 1).

Per (cell, batch, arm) against same-batch stock:

* wall ratio        = stock wall / arm wall (headline for LI/LIO/SS)
* per-token ratio   = stock (wall/out) / arm (wall/out)
* LO corrected      = per-token ratio divided by the context factor
  [(1-s) + s * C_arm/C_stock], where C is the token-weighted mean
  context position Sum_i o_i*(p_i+(o_i+1)/2) / Sum_i o_i and s = 0.15
  is the context-scaling share of per-token cost at batch 8 from the
  phase-98 Nsight account (attention+KV share between 10.9% at b1 and
  18.9% at b32). One estimator, every arm, stock included (trivially 1).
* amended gates: 3a length-sanity (violations reported, cells marked),
  3b divergence diagnostic, cap-hit fractions.

Reads data/campaign1/*.json; writes campaign1_scored.json and prints a
table. Rerunnable at any point mid-campaign; scores whatever has landed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR.parent / "data" / "campaign1"
sys.path.insert(0, str(SCRIPT_DIR))

import w100_protocol as P  # noqa: E402

CTX_SHARE = 0.15  # registered estimator constant, amendment 1


def _ctx_pos(requests) -> float:
    num = sum(r["out_toks"] * (r["prompt_toks"] + (r["out_toks"] + 1) / 2)
              for r in requests)
    den = sum(r["out_toks"] for r in requests)
    return num / den if den else 0.0


def main() -> None:
    recs = {}
    for p in sorted(OUT.glob("*__b*.json")):
        r = json.loads(p.read_text())
        recs[(r["arm"], r["batch"])] = r
    scored = []
    for (arm, batch), r in sorted(recs.items(), key=lambda kv: (
            kv[0][1], kv[0][0] != "stock", kv[0][0])):
        if arm == "stock":
            continue
        stock = recs.get(("stock", batch))
        if stock is None:
            continue
        for label, e in r["cells"].items():
            se = stock["cells"][label]
            pt_arm = e["wall_s"] / e["total_out_toks"]
            pt_stock = se["wall_s"] / se["total_out_toks"]
            row = {
                "cell": label, "batch": batch, "arm": arm,
                "wall_ratio": round(se["wall_s"] / e["wall_s"], 4),
                "per_token_ratio": round(pt_stock / pt_arm, 4),
                "cap_fraction": round(sum(
                    q["finish"] == "length" for q in e["requests"])
                    / e["n"], 4),
            }
            if e["temperature"] == 0.0:
                ref_o = {q["prompt_index"]: q["out_toks"]
                         for q in se["requests"]}
                got_o = {q["prompt_index"]: q["out_toks"]
                         for q in e["requests"]}
                try:
                    row["gate_3a"] = P.length_sanity_gate(
                        label, ref_o, got_o)
                    row["gate_3a"]["verdict"] = "pass"
                except P.ProtocolViolation as exc:
                    row["gate_3a"] = {"verdict": "FAIL",
                                      "reason": str(exc)}
                mean_out = (se["total_out_toks"] / se["n"])
                row["gate_3b"] = P.divergence_diagnostic(
                    {q["prompt_index"]: q["sha256"]
                     for q in se["requests"]},
                    {q["prompt_index"]: q["sha256"]
                     for q in e["requests"]}, mean_out)
            if e["cell"] == "LO":
                c_ratio = _ctx_pos(e["requests"]) / _ctx_pos(
                    se["requests"])
                factor = (1 - CTX_SHARE) + CTX_SHARE * c_ratio
                row["lo_ctx_ratio"] = round(c_ratio, 4)
                row["score"] = round(
                    row["per_token_ratio"] / factor, 4)
            else:
                row["score"] = row["wall_ratio"]
            scored.append(row)
    (OUT / "campaign1_scored.json").write_text(
        json.dumps({"ctx_share": CTX_SHARE, "rows": scored}, indent=1))
    cells_order = ["SS", "LI", "LIO", "LO", "LO_T1"]
    for batch in sorted({r["batch"] for r in scored}):
        print(f"\n== batch {batch} (score vs stock; LO ctx-corrected) ==")
        arms = sorted({r["arm"] for r in scored if r["batch"] == batch})
        hdr = "arm".ljust(12) + "".join(c.rjust(9) for c in cells_order)
        print(hdr)
        for arm in arms:
            vals = {r["cell"]: r for r in scored
                    if r["batch"] == batch and r["arm"] == arm}
            line = arm.ljust(12)
            for c in cells_order:
                v = vals.get(c)
                line += (f"{v['score']:9.3f}" if v else " " * 9)
            flags = [c for c in cells_order
                     if vals.get(c, {}).get("gate_3a", {}).get(
                         "verdict") == "FAIL"]
            if flags:
                line += "   3a-FAIL:" + ",".join(flags)
            print(line)


if __name__ == "__main__":
    main()
