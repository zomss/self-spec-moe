#!/usr/bin/env python3
"""E2: compose the E1 cost map R with measured accept lengths -> strategy map.

speedup(lever, cell) = max_gamma  tau_beta(gamma) / (gamma * R(lever, cell) + 1)

- Formula validated end-to-end in P75 E3 (delivered ~90% of prediction at b1
  dense). It assumes the verify's K+1 tokens ride one ~memory-bound target step;
  at compute-bound cells (short ctx x high batch) it is a ROOFLINE (P74: real
  b64 self-spec lost outright) -- those cells are marked.
- tau_beta(gamma) = (1 - beta^(gamma+1)) / (1 - beta): geometric accept model,
  validated against real multi-token accept in P24 B1 (ratio 1.00 at k<=4).
- beta per lever from prior-phase measurements (see BETA table). Layer-skip has
  NO credible beta (P17: acceptance collapses super-linearly) -> we report the
  BREAK-EVEN beta it would need to (a) reach 1.0x, (b) beat the cell winner.
- kvq is NOT draft-only in the current harness (P74 SHARED_KV) -> excluded from
  winner selection, shown as reference.

Reads data/e1/summary.csv; writes the map to stdout (markdown).
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

GAMMAS = range(1, 9)

# (beta, in_selection, source)
BETA = {
    "d_w4marlin": (0.91, True, "P75 tau-fit (3.52/4.71/5.96 @ g=3/5/7); P22 int4 0.90-0.92"),
    "d_w4machete": (0.91, True, "same weights as marlin"),
    "d_fp8w8a8": (0.95, True, "P18 fp8 anchor 0.954; P74 fp8 draft 4.88/5 @K4"),
    "d_win": (0.94, True, "P74 window accept 4.76/5 @K4 (4.79 @32k)"),
    "d_kvq": (0.91, False, "P74 kvq 4.56/5 @K4 -- GLOBAL lever, not draft-only"),
    "d_bf16": (0.99, False, "plain self-draft reference (accept ~K+1)"),
    "m_fp8marlin": (0.95, True, "P18 0.954 (Qwen3-30B, rejection-sampling vs bf16)"),
    "m_fp8block": (0.95, True, "same quant error class as fp8marlin"),
    "m_win": (0.94, True, "P74 window accept (same model/ctx family)"),
    "m_localroute": (0.85, True, "P24 B1 k-sweep per-pos ~0.85 (0.5E cache; range 0.78-0.92)"),
    "m_kvq": (0.91, False, "P74 -- GLOBAL lever"),
    "m_bf16": (0.99, False, "plain self-draft reference"),
}
SKIP_ARMS = {"d_skip50", "d_skip25", "m_skip50"}  # break-even analysis only
IGNORE = {"d_bf16dummy", "m_bf16dummy", "m_skipa2a"}  # denominators / timing stand-in

# compute-bound roofline cells (P74: verify's B*(K+1) tokens are NOT free there)
def compute_bound(batch: int, ctx: int) -> bool:
    return batch >= 32 and ctx <= 2


def tau(beta: float, g: int) -> float:
    return (1 - beta ** (g + 1)) / (1 - beta)


def best_speedup(beta: float, r: float) -> tuple[float, int]:
    cand = [(tau(beta, g) / (g * r + 1), g) for g in GAMMAS]
    return max(cand)


def breakeven_beta(r: float, target: float) -> float | None:
    """Smallest beta (over gamma) whose best speedup reaches `target`."""
    lo, hi = 0.30, 0.999
    if best_speedup(hi, r)[0] < target:
        return None
    for _ in range(40):
        mid = (lo + hi) / 2
        if best_speedup(mid, r)[0] >= target:
            hi = mid
        else:
            lo = mid
    return hi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    path = Path(a.csv or Path(__file__).resolve().parent.parent / "data/e1/summary.csv")

    R: dict = defaultdict(dict)   # group -> (arm,(b,c)) -> ratio
    for row in csv.DictReader(path.open()):
        if int(row.get("overpool", 0)):
            continue
        R[row["group"]][(row["arm"], (int(row["batch"]), int(row["ctx"])))] = float(row["ratio"])

    for group in R:
        cellset = sorted({cell for (_, cell) in R[group]})
        arms = sorted({arm for (arm, _) in R[group]})
        sel = [x for x in arms if BETA.get(x, (0, False, ""))[1]]
        refs = [x for x in arms if x in BETA and not BETA[x][1]]
        skips = [x for x in arms if x in SKIP_ARMS]

        print(f"\n## {group}: strategy map  (winner | speedup* | gamma*)\n")
        hdr = ["cell"] + sel + ["WINNER"] + refs
        print("| " + " | ".join(hdr) + " |")
        print("|" + "|".join("---" for _ in hdr) + "|")
        for cell in cellset:
            b, c = cell
            row, best = [], (0.0, "-", 0)
            for armname in sel:
                r = R[group].get((armname, cell))
                if r is None:
                    row.append("-")
                    continue
                s, g = best_speedup(BETA[armname][0], r)
                row.append(f"{s:.2f} (g{g})")
                if s > best[0]:
                    best = (s, armname, g)
            win = (f"**{best[1].split('_', 1)[1]} {best[0]:.2f}x**" if best[0] > 1.0
                   else f"none ({best[1].split('_', 1)[1]} {best[0]:.2f}x)")
            if compute_bound(b, c):
                win += " †"
            refcols = []
            for armname in refs:
                r = R[group].get((armname, cell))
                refcols.append(f"{best_speedup(BETA[armname][0], r)[0]:.2f}" if r else "-")
            print(f"| b{b}/{c}k | " + " | ".join(row) + f" | {win} | " + " | ".join(refcols) + " |")

        if skips:
            print(f"\n### {group}: layer-skip break-even beta (P17: measured accept COLLAPSES)\n")
            hdr = ["cell"] + [f"{s} beta>=1.0x" for s in skips] + [f"{s} beta to beat winner" for s in skips]
            print("| " + " | ".join(hdr) + " |")
            print("|" + "|".join("---" for _ in hdr) + "|")
            for cell in cellset:
                b, c = cell
                col1, col2 = [], []
                winner_s = 0.0
                for armname in sel:
                    r = R[group].get((armname, cell))
                    if r is not None:
                        winner_s = max(winner_s, best_speedup(BETA[armname][0], r)[0])
                for s in skips:
                    r = R[group].get((s, cell))
                    if r is None:
                        col1.append("-"); col2.append("-")
                        continue
                    be1 = breakeven_beta(r, 1.0)
                    bew = breakeven_beta(r, winner_s)
                    col1.append(f"{be1:.2f}" if be1 else ">0.999")
                    col2.append(f"{bew:.2f}" if bew else ">0.999")
                print(f"| b{b}/{c}k | " + " | ".join(col1 + col2) + " |")

    print("\n**Legend**: speedup* = max over gamma of tau_beta(gamma)/(gamma*R+1) "
          "(P75-validated roofline; ~90% delivered at b1 dense, P75 E3). "
          "† = compute-bound cell -- formula optimistic there (P74: real high-batch "
          "self-spec lost; verify tokens are not free). kvq/bf16 columns are "
          "REFERENCE only (kvq is global, not draft-only; bf16 R=1 shows plain "
          "self-drafting never pays).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
