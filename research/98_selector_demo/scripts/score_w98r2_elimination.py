# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Score D1-prime's second clause: zero false eliminations.

The preregistered claim has two clauses -- interval coverage (scored by
`round2_result.json`) and ZERO FALSE ELIMINATIONS under the sound rule
`(K+1)/q_lo < 1 + epsilon_arm` (`w98_cost_model.eliminate`). Round 1 reported
its rule NOT EXERCISED for lack of a scored surface; Round 2's held-out cells
carry `verify_s`, so the rule is evaluated here on measurements.

`q` is the per-step relative cost of the armed configuration against the
verify-only baseline. To make the test as elimination-favorable as possible
(the hardest version of a zero-elimination claim), q_lo substitutes the
model's OPTIMISTIC draft-chain bound into the FULL armed step, keeps every
measured overhead, and normalises by verify alone:

    q_lo = (mean_armed_step - draft_chain_measured + draft_chain_lo) / verify

A false elimination would need `eliminate(K, q_lo)` true while
`eliminate(K, q_measured)` is false. Writes d1p_false_elimination.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from w98_cost_model import eliminate, s_max  # noqa: E402

REPO_ROOT = SCRIPT_DIR.parents[2]
G98C = REPO_ROOT / "research/98_selector_demo/data/g98_c"
K_DEPTH = 4
EPSILON_ARM = 0.015


def main() -> int:
    result = json.loads((G98C / "round2_result.json").read_text())
    rows_out = []
    eliminations = false_eliminations = 0
    min_s_max = None
    for row in result["rows"]:
        cell_file = G98C / "heldout" / (row["cell"].replace("/", "_") + ".json")
        obs = json.loads(cell_file.read_text())["observations"][row["regime"]]
        verify = obs["verify_s"]
        overhead_step = obs["mean_armed_step_s"] - obs["draft_chain_s"]
        q_lo = (overhead_step + row["lo"]) / verify
        q_meas = (overhead_step + row["measured_s"]) / verify
        s = s_max(K_DEPTH, q_lo)
        kill = eliminate(K_DEPTH, q_lo, EPSILON_ARM)
        false_kill = kill and not eliminate(K_DEPTH, q_meas, EPSILON_ARM)
        eliminations += kill
        false_eliminations += false_kill
        min_s_max = s if min_s_max is None else min(min_s_max, s)
        rows_out.append(
            {
                "cell": row["cell"],
                "regime": row["regime"],
                "q_lo": round(q_lo, 4),
                "q_measured": round(q_meas, 4),
                "s_max": round(s, 4),
                "eliminated": kill,
                "false_elimination": false_kill,
            }
        )
    record = {
        "schema_version": 1,
        "record_type": "w98r2_d1p_false_elimination",
        "package_id": result["package_id"],
        "rule": "(K+1)/q_lo < 1 + epsilon_arm",
        "k_depth": K_DEPTH,
        "epsilon_arm": EPSILON_ARM,
        "q_definition": (
            "(mean_armed_step - draft_chain_measured + draft_chain_lo) / verify"
            " -- optimistic bound in the full armed step over verify-only,"
            " the most elimination-favorable defensible normalisation"
        ),
        "rows": rows_out,
        "eliminations": eliminations,
        "false_eliminations": false_eliminations,
        "min_s_max": round(min_s_max, 4),
        "elimination_boundary_s_max": 1.0 + EPSILON_ARM,
    }
    out = G98C / "d1p_false_elimination.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "eliminations": eliminations,
                "false_eliminations": false_eliminations,
                "min_s_max": record["min_s_max"],
                "boundary": record["elimination_boundary_s_max"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
