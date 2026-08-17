#!/usr/bin/env python3
"""Compute the W100 registration barrier (digest pin of the protocol).

Hashes every registered artifact and writes
`data/registration/w100_barrier.json` with per-file sha256 and a short
combined digest (sha256 over the sorted per-file digests, first 16 hex),
phase-98 style. The barrier file is committed BEFORE the first scored
boot; the commit hash is then recorded in results documents.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PHASE = Path(__file__).resolve().parent.parent

PINNED = [
    "w100_prereg.md",
    "final_eval_design.md",
    "README.md",
    "dataset_map.md",
    "baseline_survey.md",
    "scripts/w100_eval_datasets.py",
    "scripts/w100_protocol.py",
    "scripts/generate_w100_prompt_manifest.py",
    "scripts/run_w100_pilot.py",
    "scripts/analyze_w100_pilot.py",
    "data/registration/w100_prompts.jsonl.gz",
    "data/registration/w100_prompt_manifest.json",
    "data/pilot/pilot_lengths.json",
    "data/pilot/pilot_feasibility.json",
]


def main() -> None:
    files = {}
    for rel in PINNED:
        p = PHASE / rel
        files[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    combined = hashlib.sha256(
        "\n".join(f"{k}  {v}" for k, v in sorted(files.items())).encode()
    ).hexdigest()[:16]
    barrier = {
        "barrier_id": "w100-final-grid-barrier-v1",
        "registered_date": "2026-08-16",
        "digest": combined,
        "files": files,
        "rule": ("no scored boot before the commit carrying this file; "
                 "changes to pinned artifacts require a superseding "
                 "barrier"),
    }
    dest = PHASE / "data" / "registration" / "w100_barrier.json"
    dest.write_text(json.dumps(barrier, indent=1, sort_keys=True))
    print(f"barrier digest {combined} over {len(files)} files -> {dest}")


if __name__ == "__main__":
    main()
