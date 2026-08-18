# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step 1 -- does the lattice still reorder across weight versions at equal work?

Section 15 reported Spearman **+0.286** between the two draft weight
versions' rankings of the same seven configurations, against **+0.902** for a
change of workload, and concluded that the weight version reorders the
lattice more than the task does. Section 17 then found the prediction ranks
`skip8` fifth where the cell measures it second.

Both were measured under **natural EOS**, where each arm emits a different
number of tokens -- 104,530 to 159,170 against the parked arm's 134,139, a
52% spread. Section 21 established that a comparison holding the machine
fixed while the workload moves is measuring both, and retired two
conclusions that had been drawn that way. Neither section 15's reorder nor
section 17's misranking has been re-taken since.

This scores the same seven configurations in both families under equal work
(`ignore_eos`, fixed budget), where every arm emits exactly the same tokens
over the same prompts, and reports the two protocols side by side.

One property makes the comparison direct: with identical generation lengths
the Campaign-1 context correction becomes a common factor across arms, so
the corrected and raw rankings coincide and `score_vs_off` can be read as a
ranking without further argument.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import score_w98_refined_lo as base  # noqa: E402

DATA = SCRIPT_DIR.parent / "data"
OFF_KEY = "off"

PROTOCOLS = {
    "equal_work": {
        "target-matching": [DATA / "g98_lo_ew_bf16"],
        "w4a16-quantized": [DATA / "g98_lo_ew_q4"],
    },
    "natural_eos": {
        "target-matching": [DATA / "g98_lo_b8", DATA / "g98_lo_sweep"],
        "w4a16-quantized": [DATA / "g98_lo_q4"],
    },
}


def arm_key(cell: str) -> str:
    """`quant/window/skip` -> `window/skip`, so families align."""
    return "/".join(cell.split("/")[1:])


def equal_work_gate(scored: dict[str, Any]) -> dict[str, Any]:
    """Every arm must have emitted the same tokens, with no cap hit."""
    totals = {c: r["total_out_tokens"] for c, r in scored["arms"].items()}
    caps = {c: r["cap_hits"] for c, r in scored["arms"].items() if r["cap_hits"]}
    contexts = {c: r["context_ratio"] for c, r in scored["arms"].items()}
    unique = sorted(set(totals.values()))
    return {
        "token_counts": unique,
        "identical_work": len(unique) == 1,
        "cap_hits": caps,
        "max_context_ratio_deviation": round(
            max(abs(v - 1.0) for v in contexts.values()), 6
        ),
        "ok": len(unique) == 1 and not caps,
    }


def rank_correlation(a: dict[str, float], b: dict[str, float]) -> dict[str, Any]:
    from scipy.stats import spearmanr

    keys = sorted(set(a) & set(b))
    rho, p = spearmanr([a[k] for k in keys], [b[k] for k in keys])
    return {
        "arms": keys,
        "n": len(keys),
        "spearman": round(float(rho), 4),
        "p_value": round(float(p), 4),
    }


def main() -> int:
    out: dict[str, Any] = {"record_type": "w98_ew_lattice_score", "protocols": {}}
    ratios: dict[str, dict[str, dict[str, float]]] = {}
    for protocol, families in PROTOCOLS.items():
        entry: dict[str, Any] = {}
        ratios[protocol] = {}
        for family, directories in families.items():
            # A family is scorable only once its own parked boot exists;
            # partial directories are skipped rather than half-scored.
            if not any((Path(d) / f"{OFF_KEY}.json").is_file() for d in directories):
                continue
            scored = base.score(list(directories))
            entry[family] = scored
            if protocol == "equal_work":
                entry.setdefault("gates", {})[family] = equal_work_gate(scored)
            ratios[protocol][family] = {
                arm_key(c): r["score_vs_off"]
                for c, r in scored["arms"].items()
                if c != OFF_KEY
            }
        families_present = [f for f in families if f in entry]
        if len(families_present) == 2:
            entry["denominator_check"] = base.compare(
                [entry[f] for f in families_present]
            )
            entry["family_rank_correlation"] = rank_correlation(
                ratios[protocol][families_present[0]],
                ratios[protocol][families_present[1]],
            )
        out["protocols"][protocol] = entry

    # The contamination, per arm: what natural EOS added or removed.
    contamination = {}
    for family in ("target-matching", "w4a16-quantized"):
        rows = {}
        for arm in sorted(
            set(ratios.get("equal_work", {}).get(family, {}))
            & set(ratios.get("natural_eos", {}).get(family, {}))
        ):
            natural = ratios["natural_eos"][family][arm]
            equal = ratios["equal_work"][family][arm]
            rows[arm] = {
                "natural_eos": natural,
                "equal_work": equal,
                "inflation_pct": round((natural / equal - 1.0) * 100, 2),
            }
        if rows:
            contamination[family] = rows
    out["contamination"] = contamination

    target = DATA / "g98_fit" / "ew_lattice_score.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
