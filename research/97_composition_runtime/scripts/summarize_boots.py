#!/usr/bin/env python3
"""Build the Phase 97 four-class proxy boot resource ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED = ("baseline", "weight_q", "kv_q", "weight_q_kv_q")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def read_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "source": str(path)}
    record = json.loads(path.read_text())
    record["source"] = str(path)
    return record


def compact(record: dict[str, Any]) -> dict[str, Any]:
    cache = record.get("cache") or {}
    smoke = record.get("smoke") or {}
    engine_args = record.get("engine_args") or {}
    after_boot = record.get("gpu_after_boot") or {}
    after_shutdown = record.get("gpu_after_shutdown") or {}
    sampler = record.get("memory_sampler", {})
    return {
        "boot_class": record.get("boot_class"),
        "status": record.get("status"),
        "source": record.get("source"),
        "gpu_physical": record.get("gpu_physical"),
        "weight_q": record.get("weight_q"),
        "kv_q": record.get("kv_q"),
        "draft_execution": record.get("draft_execution", "compiled"),
        "target_execution": record.get("target_execution", "compiled"),
        "max_num_batched_tokens": engine_args.get("max_num_batched_tokens"),
        "boot_seconds": record.get("boot_seconds"),
        "num_gpu_blocks": cache.get("num_gpu_blocks"),
        "block_size": cache.get("block_size"),
        "kv_cache_size_tokens": cache.get("kv_cache_size_tokens"),
        "kv_cache_max_concurrency": cache.get("kv_cache_max_concurrency"),
        "gpu_after_boot_mib": after_boot.get("used_mib"),
        "sampled_peak_mib": sampler.get("peak_used_mib"),
        "gpu_after_shutdown_mib": after_shutdown.get("used_mib"),
        "smoke_output_tokens": smoke.get("output_tokens"),
        "accepted_tokens": smoke.get("counters", {}).get("accepted_tokens"),
        "draft_tokens": smoke.get("counters", {}).get("draft_tokens"),
        "failure_stage": record.get("failure_stage"),
        "observed_elapsed_before_stop_seconds_gte": record.get(
            "observed_elapsed_before_stop_seconds_gte"
        ),
        "observed_gpu_during_stall": record.get("observed_gpu_during_stall"),
        "termination": record.get("termination"),
        "error": record.get("error"),
    }


def main() -> int:
    args = parse_args()
    records = [read_record(args.input_dir / f"{name}.json") for name in EXPECTED]
    rows = [compact(record) for record in records]
    baseline_tokens = next(
        (
            row["kv_cache_size_tokens"]
            for row in rows
            if row["boot_class"] == "baseline" and row["kv_cache_size_tokens"]
        ),
        None,
    )
    for row in rows:
        tokens = row["kv_cache_size_tokens"]
        row["kv_capacity_vs_baseline"] = (
            round(tokens / baseline_tokens, 6)
            if tokens is not None and baseline_tokens
            else None
        )

    diagnostics = []
    diagnostic_paths = sorted(
        (args.input_dir.parent / "diagnostics").glob("weight_q_kv_q_mbt*.json")
    )
    for diagnostic_path in diagnostic_paths:
        diagnostic = read_record(diagnostic_path)
        diagnostic_row = compact(diagnostic)
        tokens = diagnostic_row["kv_cache_size_tokens"]
        diagnostic_row["kv_capacity_vs_baseline"] = (
            round(tokens / baseline_tokens, 6)
            if tokens is not None and baseline_tokens
            else None
        )
        diagnostics.append(diagnostic_row)

    protocol = {
        "classes": EXPECTED,
        "model": "Qwen/Qwen3-8B",
        "k": 4,
        "max_model_len": 20480,
        "max_num_seqs": 32,
        "gpu_memory_utilization": 0.90,
        "note": "static non-scored lower-bound proxies",
    }
    protocol_hash = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()
    ledger = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": all(row["status"] == "booted" for row in rows),
        "non_scored": True,
        "static_proxy": True,
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "classes": rows,
        "diagnostics": diagnostics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(f"[P97] proxy ledger complete={ledger['complete']} -> {args.out}")
    return 0 if ledger["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
