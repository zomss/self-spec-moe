# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Realized acceptance and armed fraction from refined-LO traces.

The scored runs record throughput; the traces record why. This pulls the two
quantities that decide whether a family difference is cost or acceptance:

* **armed fraction** -- what share of target steps actually ran a draft. The
  refined-LO record flagged this as a live confound ("the LO runs use the
  live K/OFF ladder, which may park steps"), which would make any inversion
  of measured time by acceptance invalid. It is checked here rather than
  assumed.
* **acceptance per drafted token** -- ``A_accepted / (armed * K)``, the
  quantity the selector's Round 2 consumes.

Counts come from the same filters the D2 campaign used: engine-step records
with no exclusion reasons and a nonzero target-step count.

Traces are large and are not committed, so this writes the derived counts to
a record that outlives them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

K_ARMED = 4
STEP_RECORD = '"koff_engine_step"'


def counts(path: Path) -> dict[str, Any]:
    steps = armed = accepted = committed = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            # Cheap prefilter: these files reach hundreds of megabytes and
            # only a minority of lines are engine steps.
            if STEP_RECORD not in line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("record_type") != "koff_engine_step":
                continue
            if record.get("exclusion_reasons"):
                continue
            counters = record.get("counters") or {}
            if not counters.get("H_target_steps"):
                continue
            steps += counters["H_target_steps"]
            armed += counters.get("D_armed") or 0
            accepted += counters.get("A_accepted") or 0
            committed += counters.get("E_committed") or 0
    return {
        "target_steps": steps,
        "armed_steps": armed,
        "armed_fraction": round(armed / steps, 6) if steps else 0.0,
        "accepted_per_drafted": (
            round(accepted / (armed * K_ARMED), 6) if armed else 0.0
        ),
        "committed_per_step": round(committed / steps, 6) if steps else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dirs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms: dict[str, Any] = {}
    for directory in args.trace_dirs:
        for path in sorted(Path(directory).glob("*.jsonl")):
            key = f"{directory.parent.name}/{path.stem}"
            arms[key] = counts(path)
            print(f"{key:44s} {json.dumps(arms[key])}", flush=True)
    args.output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "w98_lo_acceptance",
                "k_armed": K_ARMED,
                "arms": arms,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
