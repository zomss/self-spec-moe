#!/usr/bin/env python3
"""The certified region: where the product-of-singles surrogate is provably usable.

D2(b) found that the product of per-layer retentions ranks layer-skip sets
EXACTLY right at k=4 and k=8 (Spearman +1.000) and INVERTS at k=16. Read as a
negative result that is the non-additivity theorem again. Read as a positive
one it says something the search can actually use: the surrogate has a
validity DOMAIN, and inside it the ordering is not approximate but correct.

For that to be an algorithmic foundation rather than an observation, the
domain must be **certifiable from single-lever measurements alone** -- you
have to know you are inside it before you measure the composition, otherwise
the certificate is worth nothing.

## The certificate

The product surrogate is a multiplicative-independence approximation: it
assumes each lever's damage to acceptance is independent of the others. That
is a first-order approximation in the total damage, so it should hold while
the damage is small and fail once the levers are destroying enough of the
draft to interact. Define the **damage budget** of a composition from its
singles:

    D(set) = sum_i -log( retention_i ),   retention_i = tau(lever_i) / tau(base)

D is computable from single-lever profiles -- the same data the cost model
already needs -- so it is known BEFORE any composed measurement. The claim
under test:

    there exists D* such that for every composition with D <= D*, the
    product surrogate ranks correctly.

This script measures D* on the full G98-D factorial: base, all singles, and
22 compositions spanning quantization x window x skip, at six regimes and
two content seeds.

## What is measured vs assumed

Everything here is measured acceptance from G98-D. The only modelling choice
is which acceptance statistic to use: results are reported for BOTH `tau_eff`
(the full KMAX=8 stream, the statistic D2(a)/D2(b) worked in) and `tau_k4`
(the deployed depth the selector actually arms at), and the region must
survive both to count.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PHASE = Path(__file__).resolve().parent.parent
BASE_CELL = "target-matching/woff/skip0"
DEPLOY_K = 4


def _cells(directory: Path):
    for path in sorted(directory.glob("*.json")):
        if path.name.endswith(".telemetry.json"):
            continue
        yield json.loads(path.read_text())


def _config_key(cfg: Mapping[str, Any]) -> str:
    window = cfg["window"]
    window_key = "woff" if window in (None, "off", 0) else f"w{window}"
    return f"{cfg['quant']}/{window_key}/skip{cfg['skip_count']}"


def _tau_k4(obs: Mapping[str, Any]) -> float | None:
    profile = obs.get("tau_profile") or {}
    armed = 0
    positions = [0] * DEPLOY_K
    for entry in (profile.get("buckets") or {}).values():
        armed += int(entry.get("armed_steps", 0))
        pos = entry.get("pos_accepted") or []
        for index in range(min(DEPLOY_K, len(pos))):
            positions[index] += int(pos[index])
    return 1.0 + sum(positions) / armed if armed else None


def load_acceptance(repo: Path, statistic: str) -> dict[tuple[str, str, int], float]:
    """(cell, regime, seed) -> acceptance."""
    out: dict[tuple[str, str, int], float] = {}
    for subdir in ("singles", "confirm"):
        directory = repo / "data" / "g98_d" / subdir
        for record in _cells(directory):
            cell = _config_key(record["config"])
            seed = int(record["content_seed"])
            for regime, obs in record["observations"].items():
                value = (
                    float(obs["closure"]["tau_eff"])
                    if statistic == "tau_eff"
                    else _tau_k4(obs)
                )
                if value and value > 0:
                    out[(cell, regime, seed)] = value
    return out


def levers_of(cell: str) -> dict[str, str]:
    """Active (non-identity) levers of a cell, as lever -> single-lever cell."""
    quant, window, skip = cell.split("/")
    active: dict[str, str] = {}
    if quant != "target-matching":
        active["quant"] = f"{quant}/woff/skip0"
    if window != "woff":
        active["window"] = f"target-matching/{window}/skip0"
    if skip != "skip0":
        active["skip"] = f"target-matching/woff/{skip}"
    return active


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None

    def rank(vs: Sequence[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vs[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            shared = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = shared
            i = j + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    )
    return num / den if den else None


def build_rows(
    acceptance: Mapping[tuple[str, str, int], float],
) -> list[dict[str, Any]]:
    """One row per (composition, regime, seed) with its damage budget and error."""
    cells = sorted({c for c, _r, _s in acceptance})
    regimes = sorted({r for _c, r, _s in acceptance})
    seeds = sorted({s for _c, _r, s in acceptance})

    rows: list[dict[str, Any]] = []
    for cell, regime, seed in itertools.product(cells, regimes, seeds):
        active = levers_of(cell)
        if len(active) < 2:
            continue  # a single lever is its own prediction
        key = (cell, regime, seed)
        base_key = (BASE_CELL, regime, seed)
        if key not in acceptance or base_key not in acceptance:
            continue
        base = acceptance[base_key]
        retentions: dict[str, float] = {}
        ok = True
        for lever, single_cell in active.items():
            single = acceptance.get((single_cell, regime, seed))
            if not single:
                ok = False
                break
            retentions[lever] = single / base
        if not ok:
            continue
        product = 1.0
        for value in retentions.values():
            product *= value
        predicted = base * product
        measured = acceptance[key]
        damage = sum(-math.log(v) for v in retentions.values())
        rows.append(
            {
                "cell": cell,
                "regime": regime,
                "seed": seed,
                "n_levers": len(active),
                "retentions": {k: round(v, 6) for k, v in retentions.items()},
                "damage": round(damage, 6),
                "predicted_tau": round(predicted, 6),
                "measured_tau": round(measured, 6),
                "log_error": round(math.log(measured / predicted), 6),
            }
        )
    return rows


def ranking_by_budget(
    rows: Sequence[Mapping[str, Any]], budgets: Sequence[float]
) -> list[dict[str, Any]]:
    """Spearman(predicted, measured) within each (regime, seed), restricted to
    compositions whose damage budget is at or below the cap.

    Ranking is the quantity the search actually consumes: the surrogate only
    has to ORDER the shortlist, never to predict a value.
    """
    out: list[dict[str, Any]] = []
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["regime"], row["seed"]), []).append(row)

    for cap in budgets:
        rhos: list[float] = []
        sizes: list[int] = []
        inversions = 0
        comparable = 0
        for _key, group in sorted(groups.items()):
            kept = [r for r in group if r["damage"] <= cap]
            sizes.append(len(kept))
            rho = spearman(
                [r["predicted_tau"] for r in kept], [r["measured_tau"] for r in kept]
            )
            if rho is not None:
                rhos.append(rho)
            # top-1: does the surrogate's argmax measure best?
            if len(kept) >= 2:
                comparable += 1
                pick = max(kept, key=lambda r: r["predicted_tau"])
                truth = max(kept, key=lambda r: r["measured_tau"])
                if pick["cell"] != truth["cell"]:
                    inversions += 1
        out.append(
            {
                "damage_cap": round(cap, 4),
                "mean_spearman": round(statistics.mean(rhos), 6) if rhos else None,
                "min_spearman": round(min(rhos), 6) if rhos else None,
                "groups_scored": len(rhos),
                "mean_kept": round(statistics.mean(sizes), 3) if sizes else 0,
                "top1_miss": inversions,
                "top1_comparable": comparable,
            }
        )
    return out


def recall_and_regret(
    rows: Sequence[Mapping[str, Any]], budgets: Sequence[float], max_m: int = 6
) -> list[dict[str, Any]]:
    """recall@m and top-1 acceptance regret, per damage cap.

    Top-1 correctness is the wrong quantity to certify: the search never
    trusts its own top pick, it CONFIRMS a shortlist. What it needs is a
    guarantee that the shortlist CONTAINS the optimum -- recall@m -- and a
    bound on what it loses when the guarantee is not met.
    """
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["regime"], row["seed"]), []).append(row)

    out: list[dict[str, Any]] = []
    for cap in budgets:
        hits = {m: 0 for m in range(1, max_m + 1)}
        scored = 0
        regrets: list[float] = []
        for _key, group in sorted(groups.items()):
            kept = [r for r in group if r["damage"] <= cap]
            if len(kept) < 2:
                continue
            scored += 1
            order = sorted(kept, key=lambda r: -r["predicted_tau"])
            truth = max(kept, key=lambda r: r["measured_tau"])
            for m in range(1, max_m + 1):
                if any(r["cell"] == truth["cell"] for r in order[:m]):
                    hits[m] += 1
            regrets.append(
                1.0 - order[0]["measured_tau"] / truth["measured_tau"]
            )
        if not scored:
            continue
        out.append(
            {
                "damage_cap": round(cap, 4),
                "groups": scored,
                "recall_at": {m: round(hits[m] / scored, 6) for m in hits},
                "top1_regret_mean": round(statistics.mean(regrets), 6),
                "top1_regret_max": round(max(regrets), 6),
            }
        )
    return out


DAMAGE_BANDS = ((0.0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 1.6), (1.6, 1e9))


def band_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Value error and ranking quality WITHIN each damage band.

    The decisive test of the damage-budget certificate. Restricting to a band
    removes the cumulative-sample confound in `ranking_by_budget`: if the
    certificate gated ranking validity, rho would fall in the high-damage
    bands. It does not -- but the value error rises sharply, which is the
    result that survives.
    """
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["regime"], row["seed"]), []).append(row)

    bands = []
    for lo, hi in DAMAGE_BANDS:
        members = [r for r in rows if lo <= r["damage"] < hi]
        if not members:
            continue
        rhos: list[float] = []
        hits = 0
        scored = 0
        for _key, group in sorted(groups.items()):
            kept = [r for r in group if lo <= r["damage"] < hi]
            if len(kept) < 3:
                continue
            rho = spearman(
                [r["predicted_tau"] for r in kept], [r["measured_tau"] for r in kept]
            )
            if rho is None:
                continue
            rhos.append(rho)
            scored += 1
            order = sorted(kept, key=lambda r: -r["predicted_tau"])
            truth = max(kept, key=lambda r: r["measured_tau"])
            if any(r["cell"] == truth["cell"] for r in order[:4]):
                hits += 1
        errs = [abs(r["log_error"]) for r in members]
        signed = [r["log_error"] for r in members]
        bands.append(
            {
                "lo": lo,
                "hi": None if hi > 1e8 else hi,
                "n": len(members),
                "median_abs_log_error": round(statistics.median(errs), 6),
                "max_abs_log_error": round(max(errs), 6),
                "mean_signed_log_error": round(statistics.mean(signed), 6),
                "groups_scored": scored,
                "mean_spearman": round(statistics.mean(rhos), 6) if rhos else None,
                "min_spearman": round(min(rhos), 6) if rhos else None,
                "recall_at_4": f"{hits}/{scored}" if scored else None,
            }
        )
    return {
        "spearman_damage_vs_abs_error": round(
            spearman(
                [r["damage"] for r in rows], [abs(r["log_error"]) for r in rows]
            )
            or 0.0,
            6,
        ),
        "bands": bands,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=PHASE)
    parser.add_argument(
        "--out", type=Path, default=PHASE / "data/g98_d/certified_region.json"
    )
    args = parser.parse_args()

    budgets = [round(0.05 * i, 4) for i in range(1, 61)]
    record: dict[str, Any] = {
        "record_type": "w98_certified_region",
        "schema_version": 1,
        "base_cell": BASE_CELL,
        "certificate": "D(set) = sum_i -log(retention_i), computable from singles",
        "statistics": {},
    }

    for statistic in ("tau_eff", "tau_k4"):
        acceptance = load_acceptance(args.repo, statistic)
        rows = build_rows(acceptance)
        curve = ranking_by_budget(rows, budgets)
        recall = recall_and_regret(rows, budgets)
        record["statistics"][statistic] = {
            "rows": rows,
            "n_rows": len(rows),
            "budget_curve": curve,
            "recall_curve": recall,
            "band_analysis": band_analysis(rows),
        }

        print(f"=== {statistic} ===  {len(rows)} composed observations")
        print("  cap   mean-rho  min-rho  kept  top1-miss/comparable")
        shown = 0
        for point in curve:
            if point["mean_spearman"] is None or point["mean_kept"] < 2:
                continue
            shown += 1
            if shown % 4 and point["damage_cap"] not in (0.5, 1.0, 1.5, 2.0, 3.0):
                continue
            print(
                f"  {point['damage_cap']:4.2f} {point['mean_spearman']:9.4f} "
                f"{point['min_spearman']:8.4f} {point['mean_kept']:6.2f} "
                f"  {point['top1_miss']}/{point['top1_comparable']}"
            )

        errs = [abs(r["log_error"]) for r in rows]
        print(
            f"  |log error|: median {statistics.median(errs):.4f}  "
            f"max {max(errs):.4f}"
        )
        print("  cap  groups  recall@1 @2  @3  @4   top1-regret mean/max")
        for point in recall:
            if point["damage_cap"] not in (0.5, 1.0, 1.5, 2.0, 3.0):
                continue
            r = point["recall_at"]
            print(
                f"  {point['damage_cap']:4.2f} {point['groups']:6d}  "
                f"{r[1]:.3f} {r[2]:.3f} {r[3]:.3f} {r[4]:.3f}   "
                f"{point['top1_regret_mean']:.4f} / {point['top1_regret_max']:.4f}"
            )
        print()

    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
