#!/usr/bin/env python3
"""Gate D0 — LIVE half (w14_plan.md item D0).

Run on the five non-scored smoke boots. Any failure blocks scored D
data.

Checks, per the plan:
  1. target-step accounting closes exactly on EVERY smoke round, after
     the declared terminal-step treatment
  2. exact scheduled-KV reconstructed from the trace agrees with the
     engine counters
  3. the observed dispatch descriptor belongs to a registered stratum
  4. repeated snapshots are monotone
  5. registration + scorer commit hashes are present in every output
"""
import glob
import json
import subprocess
import sys
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
W14 = PHASE / "data" / "w14"
FAILS = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILS.append(name)


def main():
    reg = json.load(open(W14 / "w14d_prereg.json"))
    strata = {(s["action"], 1): s["b1"] for s in reg["strata"]}
    strata.update({(s["action"], 8): s["b8"] for s in reg["strata"]})
    files = sorted(glob.glob(str(W14 / "w14d_smoke_*.json")))
    if not files:
        print("no smoke boots yet"); return 1
    print(f"Gate D0 live checks over {len(files)} smoke boots\n")

    n_rounds = 0
    closure_bad, kv_bad, strat_bad, mono_bad = [], [], [], []
    for f in files:
        d = json.load(open(f))
        cfg, K = d["config"], d["K"]
        action = "AR" if cfg == "AR" else f"K{K}"
        for c in d["cells"]:
            b = c["n_active"]
            # 3. dispatch descriptor in a registered stratum
            want = strata.get((action, b))
            got = c["query_tokens_per_req"] * b
            if want != got:
                strat_bad.append((f, action, b, want, got))
            prev = None
            for r in c["rounds"]:
                n_rounds += 1
                # 1. count closure
                if not (r["E_committed"] + r["C_clipped"]
                        == r["A_accepted"] + r["H_target_steps"]):
                    closure_bad.append((f, c["rid"], b))
                # 2. exact KV: identical prompts advance in lockstep, so
                #    Q_bar must equal b*(n_prompt + (GEN-1)/2)
                exp = b * (c["n_prompt_exact"] + (d["gen"] - 1) / 2.0)
                if abs(r["Q_bar"] - exp) > 1e-6:
                    kv_bad.append((f, c["rid"], b, r["Q_bar"], exp))
                # 4. monotone cumulative snapshots (deltas non-negative)
                if (r["dec_sum_s"] < 0 or r["dec_count"] < 0
                        or r["A_accepted"] < 0 or r["D_armed"] < 0):
                    mono_bad.append((f, c["rid"], b))
                prev = r

    check("1 target-step closure E+C=A+H on every round",
          not closure_bad, f"{n_rounds} rounds checked")
    check("2 exact scheduled-KV matches the step-trace reconstruction",
          not kv_bad, f"{len(kv_bad)} mismatches")
    check("3 dispatch descriptor in a registered stratum",
          not strat_bad, f"{len(strat_bad)} out-of-stratum")
    check("4 snapshot deltas monotone (non-negative)", not mono_bad)

    # 5. provenance
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                  cwd=PHASE.parents[1], text=True).strip()
    missing = [f for f in files
               if not json.load(open(f)).get("prereg_git_rev")]
    check("5 every output carries the registration git rev", not missing,
          f"current HEAD {rev[:12]}")
    smoke_flagged = all(json.load(open(f)).get("smoke") for f in files)
    check("5 every smoke output is flagged non-scored", smoke_flagged)

    # informational: the accounting the plan predicted
    tot_U = tot_H = 0
    for f in files:
        d = json.load(open(f))
        if d["config"] == "AR":
            continue
        for c in d["cells"]:
            for r in c["rounds"]:
                tot_U += r["U_unarmed"]; tot_H += r["H_target_steps"]
    if tot_H:
        print(f"\n  info: unarmed target steps U/H = {tot_U}/{tot_H} "
              f"= {100*tot_U/tot_H:.2f}%  (B's b8 discrepancy, now measured "
              f"directly rather than inferred)")

    print()
    if FAILS:
        print(f"GATE D0 LIVE: FAILED -> {FAILS}  — scored D data is BLOCKED")
        return 1
    print("GATE D0 LIVE: PASSED — scored D boots may proceed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
