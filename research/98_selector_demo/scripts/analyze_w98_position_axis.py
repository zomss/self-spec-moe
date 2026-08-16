#!/usr/bin/env python3
"""Does the optimum move WITHIN a request? The u-axis, read for switching.

D2(c) established that acceptance varies with generated-suffix position --
107 of 264 adjacent bucket pairs separate by interval. That is a statement
about tau. The switching question is different and stronger: does the
RANKING of configurations move with position, so that a request should
change configuration partway through generating?

This separates compiled per-regime policy from runtime switching. A
per-regime selector picks once per (batch, context) and holds; only a
runtime engine can change configuration as a request generates.

Analysis over the measured G98-D factorial; no new measurement. Two things
to know about its reach:

* the u-axis instrument carries bucket edges at **[256, 1024, 3072]**, but
  every G98-D and G98-E boot ran a 640-token budget, so buckets 2 and 3 are
  permanently EMPTY. What follows covers u < 640 only;
* which is why `research/99_kv_pressure/` runs 2048-8192 token outputs -- the
  instrument was built for that range and has never been fed it.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from analyze_w98_certified_region import spearman

PHASE = Path(__file__).resolve().parent.parent
REGIMES = ("R1", "R4", "R5", "R5cot", "R6", "R8")


def _config_key(cfg: dict[str, Any]) -> str:
    window = cfg["window"]
    window_key = "woff" if window in (None, "off", 0) else f"w{window}"
    return f"{cfg['quant']}/{window_key}/skip{cfg['skip_count']}"


def collect(repo: Path) -> tuple[dict, dict, list]:
    """(regime -> bucket -> [accepted, armed]), (regime -> cell -> bucket -> ...)."""
    pooled: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    per_cell: dict = defaultdict(dict)
    edges: list = []
    for path in glob.glob(str(repo / "data/g98_d/*/*.json")):
        if path.endswith(".telemetry.json"):
            continue
        record = json.loads(Path(path).read_text())
        if record.get("record_type") != "w98d2_cell":
            continue
        cell = _config_key(record["config"])
        for regime, obs in record["observations"].items():
            profile = obs.get("tau_profile") or {}
            if profile.get("u_edges"):
                edges = profile["u_edges"]
            slot = per_cell[regime].setdefault(cell, defaultdict(lambda: [0, 0]))
            for bucket, entry in (profile.get("buckets") or {}).items():
                accepted = sum(entry.get("pos_accepted") or [])
                armed = int(entry.get("armed_steps", 0))
                pooled[regime][bucket][0] += accepted
                pooled[regime][bucket][1] += armed
                slot[bucket][0] += accepted
                slot[bucket][1] += armed
    return pooled, per_cell, edges


def tau(pair: list[int]) -> float | None:
    return 1.0 + pair[0] / pair[1] if pair[1] else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=PHASE)
    parser.add_argument(
        "--out", type=Path, default=PHASE / "data/g98_d/position_axis.json"
    )
    args = parser.parse_args()

    pooled, per_cell, edges = collect(args.repo)
    populated = sorted({b for r in pooled.values() for b in r})

    record: dict[str, Any] = {
        "record_type": "w98_position_axis",
        "schema_version": 1,
        "u_edges": edges,
        "populated_buckets": populated,
        "note": (
            "buckets beyond the first two are empty: every boot ran a "
            "640-token budget against edges built for 3072+"
        ),
        "regimes": {},
    }

    print(f"u_edges = {edges}   populated buckets = {populated}")
    print()
    print("regime   tau(u<256)  tau(256-1024)   delta   rank rho   argmax moves?")
    for regime in REGIMES:
        pool = pooled.get(regime, {})
        t0, t1 = tau(pool.get("0", [0, 0])), tau(pool.get("1", [0, 0]))
        xs, ys, names = [], [], []
        for cell, buckets in sorted(per_cell.get(regime, {}).items()):
            a, b = tau(buckets.get("0", [0, 0])), tau(buckets.get("1", [0, 0]))
            if a and b:
                xs.append(a)
                ys.append(b)
                names.append(cell)
        if not (t0 and t1 and len(xs) >= 3):
            continue
        rho = spearman(xs, ys)
        best0 = names[max(range(len(xs)), key=lambda i: xs[i])]
        best1 = names[max(range(len(ys)), key=lambda i: ys[i])]
        moves = best0 != best1
        record["regimes"][regime] = {
            "tau_bucket0": round(t0, 6),
            "tau_bucket1": round(t1, 6),
            "delta_pct": round((t1 / t0 - 1) * 100, 4),
            "cells": len(xs),
            "rank_spearman": round(rho, 6) if rho is not None else None,
            "argmax_bucket0": best0,
            "argmax_bucket1": best1,
            "argmax_moves": moves,
        }
        print(
            f"{regime:6s} {t0:11.3f} {t1:14.3f} {(t1 / t0 - 1) * 100:+7.1f}% "
            f"{rho:+9.4f}   {'YES' if moves else 'no'}"
        )

    moved = sum(1 for v in record["regimes"].values() if v["argmax_moves"])
    record["argmax_moves_in_regimes"] = moved
    record["verdict"] = (
        "acceptance moves with position but the RANKING does not: no "
        "within-request switching opportunity is visible below u=640"
        if moved == 0
        else f"argmax moves in {moved} regimes"
    )
    print(f"\n{record['verdict']}")

    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
