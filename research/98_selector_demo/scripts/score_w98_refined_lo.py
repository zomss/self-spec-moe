# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Score refined-grid LO runs in Campaign 1's currency.

Raw throughput cannot rank these arms. Under natural EOS the arms do not
emit identical tokens -- verification runs the target at different batch
shapes, FP reduction order changes, near-tie argmaxes diverge and EOS lands
elsewhere -- so output totals span 28% and a tok/s ranking is partly a
ranking of how much each arm happened to generate.

Campaign 1's amendment 1 fixes this and is reproduced exactly here: cost per
token against the parked arm, divided by the context factor

    (1 - s) + s * C_arm / C_ref,    s = CTX_SHARE = 0.15

where ``C`` is the token-weighted mean context position. It charges an arm
for generating at deeper context rather than crediting it, and it is one
estimator applied to every arm including the reference, where it is
trivially 1.

Each weight-version family is scored against **its own** parked arm, and the
two parked arms are then compared against each other -- the check that says
whether the families may be put in one table at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_w98_refined_lo as lo  # noqa: E402
import w98_artifacts as artifacts  # noqa: E402

CTX_SHARE = 0.15  # registered estimator constant, Campaign 1 amendment 1
OFF_KEY = "off"


def prompt_lengths(count: int) -> dict[str, int]:
    """Prompt token counts by request id, in the order the runner adds them."""
    return {f"{lo.CELL}-{i}": len(t) for i, t in enumerate(lo.prompts(count))}


def mean_context(record: dict[str, Any], prompts: dict[str, int]) -> float:
    """Token-weighted mean context position over a run's generated tokens."""
    numerator = denominator = 0.0
    for request, out in record["out_tokens_by_request"].items():
        prompt = prompts[request]
        numerator += out * (prompt + (out + 1) / 2.0)
        denominator += out
    return numerator / denominator if denominator else 0.0


def score(directories: list[Path]) -> dict[str, Any]:
    """Score one weight-version family, whose arms may span directories."""
    records = {}
    for directory in directories:
        for path in sorted(Path(directory).glob("*.json")):
            if path.name == "summary.json":
                continue
            try:
                record = artifacts.read(path)
            except artifacts.ArtifactError:
                continue  # derived records share the directory
            if record.get("record_type") != "w98_refined_lo":
                continue
            if record["cell"] in records:
                raise RuntimeError(f"{path}: {record['cell']} measured twice")
            records[record["cell"]] = record
    if OFF_KEY not in records:
        raise RuntimeError(f"{directories}: no parked arm to score against")
    reference = records[OFF_KEY]
    prompts = prompt_lengths(reference["batch"])
    ref_context = mean_context(reference, prompts)
    ref_per_token = reference["wall_s"] / reference["total_out_tokens"]
    rows = {}
    for cell, record in records.items():
        per_token = record["wall_s"] / record["total_out_tokens"]
        context_ratio = mean_context(record, prompts) / ref_context
        factor = (1 - CTX_SHARE) + CTX_SHARE * context_ratio
        rows[cell] = {
            "tokens_per_s": record["tokens_per_s"],
            "total_out_tokens": record["total_out_tokens"],
            "per_token_ms": round(per_token * 1000, 3),
            "context_ratio": round(context_ratio, 4),
            "cap_hits": record["cap_hits"],
            "score_vs_off": round((ref_per_token / per_token) / factor, 4),
        }
    return {
        "record_type": "w98_refined_lo_score",
        "directories": [str(d) for d in directories],
        "ctx_share": CTX_SHARE,
        "reference_per_token_ms": round(ref_per_token * 1000, 3),
        "reference_tokens_per_s": reference["tokens_per_s"],
        "batch": reference["batch"],
        "arms": rows,
    }


def compare(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Whether the families' parked arms agree well enough to share a table.

    Anything measured against OFF on this box carries a 9-16% band, so two
    families scored against two separate parked boots are only comparable if
    those boots actually reproduced. This reports the gap rather than
    assuming it away.
    """
    references = [s["reference_per_token_ms"] for s in scores]
    spread = max(references) / min(references) - 1.0
    return {
        "reference_per_token_ms": references,
        "reference_spread_pct": round(spread * 100, 3),
        "families_comparable": bool(spread < 0.02),
        "note": "OFF on this box carries a 9-16% round-to-round band; "
        "this is the measured gap between the two families' own parked boots",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        action="append",
        required=True,
        help="comma-separated directories holding ONE weight version's arms",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    scores = [score([Path(d) for d in family.split(",")]) for family in args.family]
    result = {
        "record_type": "w98_refined_lo_score_set",
        "families": scores,
        "reference_agreement": compare(scores) if len(scores) > 1 else None,
    }
    if args.output:
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    for entry in scores:
        print(f"\n== {', '.join(entry['directories'])}  (batch {entry['batch']}) ==")
        print(f"{'arm':34s} {'tok/s':>9} {'tokens':>9} {'per-tok':>9} {'vs OFF':>8}")
        for cell, row in sorted(
            entry["arms"].items(), key=lambda kv: -kv[1]["score_vs_off"]
        ):
            print(
                f"{cell:34s} {row['tokens_per_s']:9.1f} "
                f"{row['total_out_tokens']:9d} {row['per_token_ms']:9.3f} "
                f"{row['score_vs_off']:8.3f}"
            )
    if result["reference_agreement"]:
        print(f"\n{json.dumps(result['reference_agreement'], indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
