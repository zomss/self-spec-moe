#!/usr/bin/env python3
"""Derive a fail-closed arm/disarm rule and re-score D3 under it.

The D3 selector armed at every regime because its pick was the argmax of a
predicted-rate map; it had no way to decline. At R4 that cost 24% (522 tok/s
against OFF's 685) even though the map itself predicted a margin of only
1.026x there -- the thinnest in the map. This script builds the rule that
reads that signal, and re-scores the EXISTING G98-E grid under it.

No new measurement. The grid already contains OFF and all 30 armed
configurations at all six regimes, on both instrument arms, so changing the
selection rule is pure analysis over measured cells.

## The rule

Within a regime the selector's decision is a ratio of two predicted rates,

    m(R) = max_cell rate_hat(cell, R) / rate_hat(off, R)
         = tau_hat * verify / (d_hat + verify)

so any error common to both arms cancels; what survives is acceptance error
and the DIFFERENTIAL cost error between the draft chain and verify. Arm only
when the lower confidence bound on that margin clears unity:

    LCB(R) = m(R) * exp(-z * sigma_lnm) > 1

## Where the threshold comes from

Both error terms are estimated from data that predates the D3 grid and is
disjoint from its content:

* `sigma_tau` -- G98-D measured every acceptance cell on content seeds 4 AND
  5. Their spread is the content-draw variation of tau at the deployed depth.
  D3 evaluates on seeds 2-3, so this is the right scale for "the acceptance I
  predicted vs the acceptance a new content draw delivers".
* `sigma_cost` -- D1' held-out relative errors (48 rows, `round2_result.json`),
  the cost model's own measured accuracy.

This is NOT a preregistered rule: the D3 outcome was already known when it
was written, and it is reported as a post-hoc rule with a pre-D3-data-only
derivation. The defence against fitting the threshold to the answer is the
sweep -- `--sweep` reports the verdict as a function of the threshold across
its whole plausible range, so the reader can see whether the conclusion is
knife-edge or flat.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import score_w98_g98e_d3 as scorer

PHASE = Path(__file__).resolve().parent.parent
OFF_KEY = "off"
DEPLOY_K = 4
Z_ONE_SIDED_95 = 1.645


# --------------------------------------------------------------------------
# pre-D3 error terms
# --------------------------------------------------------------------------


def _cells(directory: Path):
    for path in sorted(directory.glob("*.json")):
        if path.name.endswith(".telemetry.json"):
            continue
        yield json.loads(path.read_text())


def _config_key(cfg: Mapping[str, Any]) -> str:
    window = cfg["window"]
    window_key = "woff" if window in (None, "off", 0) else f"w{window}"
    return f"{cfg['quant']}/{window_key}/skip{cfg['skip_count']}"


def _tau_at_deploy(obs: Mapping[str, Any]) -> float | None:
    """tau at K=DEPLOY_K from the KMAX=8 stream (same read as the map)."""
    profile = obs.get("tau_profile") or {}
    armed = 0
    positions = [0] * DEPLOY_K
    for entry in (profile.get("buckets") or {}).values():
        armed += int(entry.get("armed_steps", 0))
        pos = entry.get("pos_accepted") or []
        for index in range(min(DEPLOY_K, len(pos))):
            positions[index] += int(pos[index])
    if not armed:
        return None
    return 1.0 + sum(positions) / armed


def acceptance_seed_sigma(repo: Path) -> dict[str, Any]:
    """Content-draw sd of tau, from G98-D's seed-4 vs seed-5 pairs.

    The map predicts with the mean of the two seeds and D3 measures a fresh
    draw, so the discrepancy carries the full content sd rather than the
    standard error of the mean. For a matched pair the unbiased estimate is
    `sigma^2 = mean(diff^2) / 2` on the log scale.
    """
    by_key: dict[tuple[str, str], dict[int, float]] = {}
    for subdir in ("singles", "confirm"):
        directory = repo / "data" / "g98_d" / subdir
        if not directory.is_dir():
            continue
        for record in _cells(directory):
            cell = _config_key(record["config"])
            seed = int(record["content_seed"])
            for regime, obs in record["observations"].items():
                tau = _tau_at_deploy(obs)
                if tau and tau > 0:
                    by_key.setdefault((cell, regime), {})[seed] = tau

    log_diffs: list[float] = []
    pairs = 0
    for (_cell, _regime), by_seed in sorted(by_key.items()):
        if 4 in by_seed and 5 in by_seed:
            log_diffs.append(math.log(by_seed[4] / by_seed[5]))
            pairs += 1
    if not log_diffs:
        raise SystemExit("no matched seed-4/seed-5 acceptance pairs found")

    sigma = math.sqrt(sum(d * d for d in log_diffs) / (2 * len(log_diffs)))
    absolute = sorted(abs(d) for d in log_diffs)
    return {
        "sigma_log": sigma,
        "pairs": pairs,
        "median_abs_log_diff": statistics.median(absolute),
        "p95_abs_log_diff": absolute[min(len(absolute) - 1, int(0.95 * len(absolute)))],
        "source": "G98-D seeds 4 vs 5, tau at K=4",
    }


def cost_sigma(repo: Path) -> dict[str, Any]:
    """Relative sd of the cost model, from D1' held-out rows."""
    rows = json.loads((repo / "data" / "g98_c" / "round2_result.json").read_text())
    errors = [abs(float(r["rel_error"])) for r in rows["rows"]]
    # rel_error is a signed-magnitude relative miss; use the RMS as the sd.
    sigma = math.sqrt(sum(e * e for e in errors) / len(errors))
    return {
        "sigma_rel": sigma,
        "rows": len(errors),
        "median_abs": statistics.median(errors),
        "max_abs": max(errors),
        "source": "D1' held-out rows (round2_result.json)",
    }


# --------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------


def margin_table(predictions: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """regime -> predicted argmax, its margin over OFF, and the cost shares."""
    predicted = predictions["predicted_decode_tokens_per_s"]
    detail = predictions["detail"]
    off = predicted[OFF_KEY]

    out: dict[str, dict[str, Any]] = {}
    for regime in sorted(off):
        armed = {
            cell: by_regime[regime]
            for cell, by_regime in predicted.items()
            if cell != OFF_KEY and regime in by_regime
        }
        if not armed:
            continue
        pick = max(armed, key=armed.get)
        cell_detail = detail[pick][regime]
        d_hat = float(cell_detail["d_hat_s"])
        verify = float(cell_detail["verify_s"])
        out[regime] = {
            "pick": pick,
            "margin": armed[pick] / off[regime],
            "draft_share": d_hat / (d_hat + verify),
            "tau_hat_at_k4": float(cell_detail["tau_hat_at_k4"]),
        }
    return out


def sigma_log_margin(draft_share: float, sig_tau: float, sig_cost: float) -> float:
    """sd of log(margin).

    d log m = d log tau + (d/(d+v)) * (d log v - d log d): a cost error common
    to the draft chain and verify cancels, and only the differential survives,
    scaled by the draft's share of the armed step.
    """
    cost_term = draft_share * math.sqrt(2.0) * sig_cost
    return math.hypot(sig_tau, cost_term)


def apply_rule(
    margins: Mapping[str, Mapping[str, Any]],
    sig_tau: float,
    sig_cost: float,
    z: float,
) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    for regime, row in margins.items():
        sigma = sigma_log_margin(row["draft_share"], sig_tau, sig_cost)
        lcb = row["margin"] * math.exp(-z * sigma)
        decisions[regime] = {
            "pick": row["pick"] if lcb > 1.0 else OFF_KEY,
            "armed": lcb > 1.0,
            "margin": round(row["margin"], 6),
            "sigma_log_margin": round(sigma, 6),
            "lcb": round(lcb, 6),
            "threshold_equivalent": round(math.exp(z * sigma), 6),
        }
    return decisions


def choice_from_decisions(decisions: Mapping[str, Mapping[str, Any]]):
    return {regime: row["pick"] for regime, row in decisions.items()}


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def score_choice(
    grid: Mapping[str, Mapping[str, float]],
    regimes: list[str],
    choice: Mapping[str, str],
) -> dict[str, Any]:
    """Score an explicit per-regime choice against the measured grid."""
    weights = {r: 1.0 / len(regimes) for r in regimes}
    rate = scorer.aggregate(grid, choice, weights)
    omni_pick = scorer.omniscient_choice(grid, regimes)
    omni = scorer.aggregate(grid, omni_pick, weights)
    statics = {
        cell: scorer.aggregate(grid, {r: cell for r in regimes}, weights)
        for cell, by_regime in grid.items()
        if all(r in by_regime for r in regimes)
    }
    best_static = max(statics, key=statics.get)
    beaten = {
        cell: rate >= value * (1.0 + scorer.LCB_MARGIN)
        for cell, value in statics.items()
    }
    return {
        "rate": round(rate, 6),
        "omniscient_rate": round(omni, 6),
        "omniscient_fraction": round(rate / omni, 6),
        "meets_90pct": rate / omni >= scorer.OMNISCIENT_TARGET,
        "over_off": round(rate / statics[OFF_KEY], 6),
        "best_static_cell": best_static,
        "best_static_rate": round(statics[best_static], 6),
        "over_best_static": round(rate / statics[best_static], 6),
        "beats_every_static_by_lcb_margin": all(beaten.values()),
        "statics_not_beaten": sorted(c for c, ok in beaten.items() if not ok),
        "choice": dict(choice),
        "per_regime_rate": {r: round(grid[choice[r]][r], 4) for r in regimes},
    }


def frontier(
    grid: Mapping[str, Mapping[str, float]],
    regimes: list[str],
    choice: Mapping[str, str],
    steps: int = 21,
) -> dict[str, Any]:
    """The same 126 mixes D3 reported, under a fixed per-regime choice.

    The fail-closed decision is a function of the prediction map alone, not of
    the mix, so one choice is scored across every mix -- which is what makes
    the comparison against the as-measured frontier apples to apples.
    """
    rows: list[dict[str, Any]] = []
    others = len(regimes) - 1
    for regime in regimes:
        for index in range(steps):
            share = index / (steps - 1)
            rest = (1.0 - share) / others if others else 0.0
            weights = {r: (share if r == regime else rest) for r in regimes}
            rate = scorer.aggregate(grid, choice, weights)
            omni_pick = scorer.omniscient_choice(grid, regimes)
            omni = scorer.aggregate(grid, omni_pick, weights)
            statics = {
                cell: scorer.aggregate(grid, {r: cell for r in regimes}, weights)
                for cell in grid
            }
            best_static = max(statics.values())
            rows.append(
                {
                    "concentrated_regime": regime,
                    "share": round(share, 4),
                    "omniscient_fraction": round(rate / omni, 6),
                    "meets_90pct": rate / omni >= scorer.OMNISCIENT_TARGET,
                    "over_off": round(rate / statics[OFF_KEY], 6),
                    "over_best_static": round(rate / best_static, 6),
                    "beats_every_static": (
                        rate >= best_static * (1.0 + scorer.LCB_MARGIN)
                    ),
                }
            )
    return {
        "mixes": len(rows),
        "meets_90pct": sum(1 for r in rows if r["meets_90pct"]),
        "beats_every_static": sum(1 for r in rows if r["beats_every_static"]),
        "rows": rows,
    }


def sweep(
    grid: Mapping[str, Mapping[str, float]],
    regimes: list[str],
    margins: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Verdict as a function of a bare margin threshold, 1.00 to 2.00.

    This is the robustness check that the derived threshold is not doing the
    work: if the aggregate is flat across a wide band, the result does not
    depend on where in that band the rule lands.
    """
    rows: list[dict[str, Any]] = []
    step = 0.01
    theta = 1.0
    while theta <= 2.0001:
        choice = {
            regime: (baseline[regime] if row["margin"] > theta else OFF_KEY)
            for regime, row in margins.items()
        }
        scored = score_choice(grid, regimes, choice)
        rows.append(
            {
                "threshold": round(theta, 4),
                "armed_regimes": sorted(
                    r for r, c in choice.items() if c != OFF_KEY
                ),
                "rate": scored["rate"],
                "omniscient_fraction": scored["omniscient_fraction"],
                "over_off": scored["over_off"],
                "over_best_static": scored["over_best_static"],
                "beats_every_static": scored["beats_every_static_by_lcb_margin"],
            }
        )
        theta += step
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=PHASE)
    parser.add_argument("--grid", type=Path, default=PHASE / "data/g98_e/grid")
    parser.add_argument(
        "--predictions", type=Path, default=PHASE / "data/g98_e/d3_predictions.json"
    )
    parser.add_argument(
        "--out", type=Path, default=PHASE / "data/g98_e/d3_failclosed.json"
    )
    parser.add_argument("--z", type=float, default=Z_ONE_SIDED_95)
    args = parser.parse_args()

    predictions = json.loads(args.predictions.read_text())
    sig_tau_info = acceptance_seed_sigma(args.repo)
    sig_cost_info = cost_sigma(args.repo)
    sig_tau = sig_tau_info["sigma_log"]
    sig_cost = sig_cost_info["sigma_rel"]

    margins = margin_table(predictions)
    decisions = apply_rule(margins, sig_tau, sig_cost, args.z)
    fail_closed_choice = choice_from_decisions(decisions)
    argmax_choice = {r: row["pick"] for r, row in margins.items()}

    grids = scorer.load_grid(args.grid)
    arms: dict[str, Any] = {}
    for arm, grid in sorted(grids.items()):
        regimes = scorer.regimes_of(grid)
        complete = {
            cell: by_regime
            for cell, by_regime in grid.items()
            if all(r in by_regime for r in regimes)
        }
        arms[arm] = {
            "regimes": regimes,
            "as_measured_argmax": score_choice(complete, regimes, argmax_choice),
            "fail_closed": score_choice(complete, regimes, fail_closed_choice),
            "frontier_as_measured": frontier(complete, regimes, argmax_choice),
            "frontier_fail_closed": frontier(complete, regimes, fail_closed_choice),
            "threshold_sweep": sweep(complete, regimes, margins, argmax_choice),
        }

    record = {
        "record_type": "w98d3_failclosed",
        "schema_version": 1,
        "rule": (
            "arm iff LCB(predicted margin over OFF) > 1, "
            "LCB = margin * exp(-z * sigma_log_margin)"
        ),
        "provenance": (
            "POST-HOC rule with a pre-D3-data-only derivation. The D3 outcome "
            "was known when this was written; the threshold sweep is the "
            "robustness check, not a preregistration."
        ),
        "z": args.z,
        "error_terms": {
            "acceptance": sig_tau_info,
            "cost": sig_cost_info,
        },
        "margins": {r: dict(v) for r, v in margins.items()},
        "decisions": decisions,
        "arms": arms,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    print(f"sigma_tau  (log, content draw) = {sig_tau:.4f}  "
          f"({sig_tau_info['pairs']} matched pairs)")
    print(f"sigma_cost (relative, D1')     = {sig_cost:.4f}  "
          f"({sig_cost_info['rows']} held-out rows)")
    print()
    print("regime  margin   sigma   LCB     equiv-threshold  decision")
    for regime in sorted(decisions):
        d = decisions[regime]
        verdict = "ARM" if d["armed"] else "OFF (fail closed)"
        print(
            f"{regime:6s} {d['margin']:7.4f} {d['sigma_log_margin']:7.4f} "
            f"{d['lcb']:7.4f} {d['threshold_equivalent']:15.4f}  {verdict}"
        )
    print()
    for arm, payload in sorted(arms.items()):
        a = payload["as_measured_argmax"]
        f = payload["fail_closed"]
        print(f"[{arm}]")
        print(
            f"  as measured : {a['rate']:8.1f} tok/s  "
            f"{a['omniscient_fraction']*100:5.1f}% of omniscient  "
            f"{a['over_off']:.3f}x OFF  {a['over_best_static']:.3f}x best static"
        )
        print(
            f"  fail-closed : {f['rate']:8.1f} tok/s  "
            f"{f['omniscient_fraction']*100:5.1f}% of omniscient  "
            f"{f['over_off']:.3f}x OFF  {f['over_best_static']:.3f}x best static"
        )
        print(
            f"  beats every static: {a['beats_every_static_by_lcb_margin']} "
            f"-> {f['beats_every_static_by_lcb_margin']}"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
