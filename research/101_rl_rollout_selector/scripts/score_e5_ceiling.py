# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E5 — the LO ceiling re-derived under the registered request set.

E4 invalidated the section-45 ceiling: that lattice ran `n = batch`
(unbackfilled), which E3 measured carrying +-23% draw noise, so its per-point
winners and the +7.28% figure derived from them cannot be trusted. This
re-scores LO from the `n = 4 x batch` re-measurement.

Reports what a selector could win at this cell, against three baselines that
answer different questions:

  vs stock          is speculation worth anything here
  vs best static    what SELECTION buys over one fixed configuration
  vs W4-only        what our lattice buys over EfficientRollout's single lever

The third is the one that matters for the phase's claim: `woff/skip0` in the
quantized family is their W4 drafter with no other lever applied.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve()
DATA = HERE.parents[1] / "data" / "e5_lattice_lo_b8_n32"
OLD = HERE.parents[2] / "98_selector_demo" / "data" / "g98_lat_lo_b8"

W4_ONLY = "woff/skip0"  # EfficientRollout's lever: quantized draft, nothing else


def load(directory: Path, quant: str = "w4a16-quantized") -> dict[str, dict]:
    out = {}
    for path in sorted(Path(directory).glob("*.json")):
        if path.name == "summary.json":
            continue
        try:
            rec = json.loads(path.read_text())
        except ValueError:
            continue
        if rec.get("record_type") != "w98_refined_lo":
            continue
        cell = rec["cell"]
        if "/" in cell and rec["config"].get("quant") != quant:
            continue
        out[cell if "/" not in cell else cell.split("/", 1)[1]] = rec
    return out


def per_token(rec: dict) -> float:
    return rec["wall_s"] / rec["total_out_tokens"]


def main() -> int:
    new = load(DATA)
    if "stock" not in new:
        print("stock missing; E5 incomplete")
        return 1
    ref = per_token(new["stock"])
    armed = {k: v for k, v in new.items() if k not in ("stock", "off")}
    if len(armed) < 5:
        print(f"only {len(armed)} armed arms so far; run incomplete")
        return 1
    score = {k: ref / per_token(v) for k, v in armed.items()}
    order = sorted(score, key=lambda k: -score[k])

    print(f"E5 LO lattice, n=32 registered protocol ({len(armed)} arms)")
    print(f"  stock {new['stock']['tokens_per_s']:.1f} tok/s")
    for k in order:
        tag = " <- EfficientRollout's lever" if k == W4_ONLY else ""
        print(f"   {k:14s} {score[k]:.4f}{tag}")

    best = order[0]
    print("\nCEILINGS at LO b8, n=32:")
    print(f"  best arm vs stock            {score[best]:.4f}   ({best})")
    if W4_ONLY in score:
        print(f"  best arm vs W4-only          {score[best] / score[W4_ONLY]:.4f}")
    if "off" in new:
        print(f"  off  vs stock                {ref / per_token(new['off']):.4f}")
    spread = score[best] / score[order[-1]] - 1.0
    print(f"  spread best/worst            {spread * 100:+.1f}%")

    # what SELECTION could buy: this is one cell, so a per-cell selector has
    # nothing to choose -- the honest ceiling is only defined across points.
    print("\n  NOTE: at a single (cell, batch) point the selector's ceiling over")
    print("  the best static is zero by construction. The number that matters is")
    print("  the arm SPREAD, which bounds what a wrong pick costs, and the")
    print("  margin over W4-only, which is what the extra lever axes buy.")

    if OLD.is_dir():
        old = load(OLD)
        if "stock" in old:
            oref = per_token(old["stock"])
            oscore = {
                k: oref / per_token(v)
                for k, v in old.items()
                if k not in ("stock", "off")
            }
            common = sorted(set(score) & set(oscore))
            print(f"\n  against section 45 (n=8), {len(common)} shared arms:")
            oorder = sorted(oscore, key=lambda k: -oscore[k])
            print(f"    s45 best {oorder[0]} {oscore[oorder[0]]:.4f}")
            print(f"    E5  best {best} {score[best]:.4f}")
            moved = sum(
                1
                for k in common
                if (sorted(common, key=lambda x: -score[x]).index(k))
                != (sorted(common, key=lambda x: -oscore[x]).index(k))
            )
            print(f"    arms whose rank changed: {moved} of {len(common)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
