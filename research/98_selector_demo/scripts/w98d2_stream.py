# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bridge: w98-d2 koff traces -> the frozen G98-0 accounting rows. CPU-only.

`w98_accounting` consumes `StepRow`s and produces closure-checked,
position-resolved, u-binned acceptance with request-level bootstrap
intervals. The `w98-d2` boot scope emits exactly the fields it needs, so
this module is a translation and nothing else -- no estimation lives here.

Field mapping, and the one derivation:

    request_id  <- request_id
    suffix_len  <- generated_suffix_len   (the request's OWN u, not the
                                           batch min/max)
    armed       <- the step carried a draft
    proposed    <- drafted_tokens
    accepted    <- accepted_draft_tokens
    clipped     <- accepted + 1 - committed_tokens

The clip derivation inverts `row_emitted` (emitted = accepted + 1 - clipped),
so a stream that closes here closes under the frozen identity E + C = A + H
rather than under a private one.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from w98_accounting import StepRow, W98AccountingError, bin_rows, bootstrap_tau

# Provisional per the preregistration; the FINAL edges are a scored output
# of the first D2 run, placed where tau(w, g, u) actually crosses.
PROVISIONAL_U_EDGES = (256, 1024, 3072)


def rows_from_records(records: Iterable[dict[str, Any]]) -> list[StepRow]:
    """Translate w98-d2 step records into accounting rows."""
    rows: list[StepRow] = []
    for record in records:
        acceptance = record.get("acceptance_rows")
        if acceptance is None:
            raise W98AccountingError(
                "step record carries no acceptance_rows; was it booted under "
                "the w98-d2 scope?"
            )
        armed = bool((record.get("counters") or {}).get("D_armed"))
        for row in acceptance:
            accepted = int(row["accepted_draft_tokens"])
            committed = int(row["committed_tokens"])
            clipped = accepted + 1 - committed
            if clipped < 0:
                raise W98AccountingError(
                    f"request {row['request_id']!r} committed {committed} "
                    f"tokens against accepted {accepted}: negative clip"
                )
            rows.append(
                StepRow(
                    request_id=str(row["request_id"]),
                    suffix_len=int(row["generated_suffix_len"]),
                    armed=armed,
                    proposed=int(row["drafted_tokens"]) if armed else 0,
                    accepted=accepted if armed else 0,
                    clipped=clipped,
                )
            )
    return rows


def read_trace(trace_path: Path) -> list[dict[str, Any]]:
    """Clean pure-decode step records of one boot, armed or priming."""
    out: list[dict[str, Any]] = []
    with Path(trace_path).open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("record_type") != "koff_engine_step":
                continue
            if record.get("exclusion_reasons"):
                continue
            if not (record.get("counters") or {}).get("H_target_steps"):
                continue
            out.append(record)
    return out


def rows_from_trace(trace_path: Path) -> list[StepRow]:
    """Accounting rows for one boot's trace."""
    return rows_from_records(read_trace(trace_path))


def tau_profile(
    rows: Sequence[StepRow],
    kmax: int,
    u_edges: Sequence[int] = PROVISIONAL_U_EDGES,
    *,
    depth: int | None = None,
    resamples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Per-u-bucket tau with request-level bootstrap intervals.

    Estimation is delegated entirely to `w98_accounting`; this only shapes
    the result for the campaign record.
    """
    depth = kmax if depth is None else depth
    buckets = bin_rows(rows, kmax, u_edges)
    intervals = bootstrap_tau(
        rows, kmax=kmax, u_edges=u_edges, k=depth, n_boot=resamples, seed=seed
    )
    per_bucket: dict[str, Any] = {}
    for index in sorted(buckets):
        counters = buckets[index]
        entry: dict[str, Any] = {
            "armed_steps": counters.armed_steps,
            "unarmed_steps": counters.unarmed_steps,
            "clipped_excluded": counters.clipped_excluded,
            "pos_accepted": list(counters.pos_accepted),
        }
        interval = intervals.get(index)
        if interval is not None:
            entry.update(
                {
                    "tau": round(interval.point, 6),
                    "tau_lo": round(interval.lo, 6),
                    "tau_hi": round(interval.hi, 6),
                    "undefined_draws": interval.undefined_draws,
                }
            )
        per_bucket[str(index)] = entry
    return {
        "kmax": kmax,
        "depth": depth,
        "u_edges": list(u_edges),
        "buckets": per_bucket,
    }
