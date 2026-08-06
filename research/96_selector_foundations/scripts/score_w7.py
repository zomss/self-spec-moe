#!/usr/bin/env python3
"""W7 scorer: MLA/MoE Round-2 completeness (w7_completeness.md).

Per (arch, rid, batch) cell and arm: pool the per-iter rates from both
boots, reference = 2nd-highest pooled round, reject rounds >5% below
reference; certified iff each boot retains >=2 rounds and the boot
means of surviving rounds agree within 2%. S = uncond/off certified
means (same-cell). Verdicts against P-W7a..e.
"""
import json
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
W7 = PHASE / "data" / "w7"
K = 4

ARMS = {
    "mla": {n: [W7 / f"w7_mla_{n}_lane{g}.json" for g in (0, 1)]
            for n in ("off", "uncond")},
    "moe": {n: [W7 / f"w7_moe_{n}_boot{b}.json" for b in (1, 2)]
            for n in ("off", "uncond")},
}
# banked C2 R at b32/ctx2000 (95/data/policy_*_wnone.json), K=4
BANKED_R_B32 = {"mla": 0.929, "moe": 0.951}


def cells_of(path):
    d = json.load(open(path))
    assert d.get("tune") is False, f"{path.name}: not a notune artifact"
    return {(c["rid"], c["batch"]): c for c in d["cells"]
            if "toks" in c}


def certify(rounds_a, rounds_b):
    """Episode rejection + cross-boot certification. Returns
    (mean, note) -- mean is None when not certifiable."""
    pooled = sorted(rounds_a + rounds_b, reverse=True)
    if len(pooled) < 4:
        return None, "too-few-rounds"
    ref = pooled[1]                      # 2nd-highest anchor
    keep_a = [r for r in rounds_a if r >= 0.95 * ref]
    keep_b = [r for r in rounds_b if r >= 0.95 * ref]
    if len(keep_a) < 2 or len(keep_b) < 2:
        return None, (f"episode-suppressed "
                      f"(kept {len(keep_a)}/{len(rounds_a)} + "
                      f"{len(keep_b)}/{len(rounds_b)})")
    ma, mb = sum(keep_a) / len(keep_a), sum(keep_b) / len(keep_b)
    if abs(ma - mb) / max(ma, mb) > 0.02:
        return None, f"boots-disagree ({ma:.1f} vs {mb:.1f})"
    n_rej = len(rounds_a) - len(keep_a) + len(rounds_b) - len(keep_b)
    return (ma + mb) / 2, f"ok ({n_rej} rounds rejected)"


def main():
    report = {}
    for arch, arms in ARMS.items():
        data = {}
        for arm, paths in arms.items():
            missing = [p.name for p in paths if not p.exists()]
            if missing:
                print(f"[W7] {arch}/{arm}: missing {missing}")
                continue
            data[arm] = [cells_of(p) for p in paths]
        if len(data) < 2:
            report[arch] = {"status": "incomplete"}
            continue
        rows = []
        keys = sorted(set(data["off"][0]) & set(data["uncond"][0]),
                      key=lambda k: (k[0], k[1]))
        for key in keys:
            rid, b = key
            row = {"rid": rid, "batch": b}
            means = {}
            for arm in ("off", "uncond"):
                a, bb = data[arm]
                if key not in a or key not in bb:
                    row[f"{arm}_note"] = "cell-missing"
                    continue
                m, note = certify(a[key]["all"], bb[key]["all"])
                means[arm] = m
                row[f"{arm}_mean"] = round(m, 1) if m else None
                row[f"{arm}_note"] = note
            tau = data["uncond"][0].get(key, {}).get("accept")
            tau2 = data["uncond"][1].get(key, {}).get("accept")
            row["tau"] = tau
            row["tau_boot2"] = tau2
            if means.get("off") and means.get("uncond"):
                S = means["uncond"] / means["off"]
                row["S"] = round(S, 4)
                if tau and S > 0:
                    row["R_hat"] = round((tau / S - 1) / K, 3)
            rows.append(row)
        report[arch] = {"cells": rows}

        # per-boot S at b32 (P-W7c needs BOTH boots >= 1.02)
        for row in rows:
            if row["batch"] != 32 or "S" not in row:
                continue
            per_boot = []
            for i in (0, 1):
                o = data["off"][i].get((row["rid"], 32))
                u = data["uncond"][i].get((row["rid"], 32))
                if o and u:
                    om = sorted(o["all"], reverse=True)[:2]
                    um = sorted(u["all"], reverse=True)[:2]
                    per_boot.append(round(
                        (sum(um) / 2) / (sum(om) / 2), 4))
            row["S_per_boot_best2"] = per_boot

    out = W7 / "w7_scored.json"
    out.write_text(json.dumps(report, indent=1))
    for arch, r in report.items():
        print(f"\n=== {arch} ===")
        if "cells" not in r:
            print(" incomplete")
            continue
        for row in r["cells"]:
            s = row.get("S")
            verdict = ""
            if s is not None:
                if row["batch"] < 32:
                    verdict = ("OFF-confirmed" if s < 1
                               else "PREDICTION-VIOLATED")
                elif row["batch"] == 32:
                    pb = row.get("S_per_boot_best2", [])
                    armed = s >= 1.02 and len(pb) == 2 and all(
                        x >= 1.02 for x in pb)
                    verdict = "ARM-candidate" if armed else "tie->OFF"
                else:
                    verdict = "map-only"
            print(f" {row['rid']:>4} b{row['batch']:<3} "
                  f"off={row.get('off_mean')} "
                  f"uncond={row.get('uncond_mean')} S={s} "
                  f"tau={row.get('tau')} R^={row.get('R_hat', '')} "
                  f"{verdict}  [{row.get('off_note')}|"
                  f"{row.get('uncond_note')}]")
    print("\n[W7] saved ->", out)


if __name__ == "__main__":
    main()
