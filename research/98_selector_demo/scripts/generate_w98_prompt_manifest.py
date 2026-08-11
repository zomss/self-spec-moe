#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze the phase-98 prompt-token bundle, CPU-only and deterministic.

Reuses Phase 97's frozen generator implementation verbatim
(``research/97_composition_runtime/scripts/generate_p4_prompt_manifest.py``)
and overrides only the output paths and the content seeds, so the freeze
semantics — dataset pinning, tokenizer verification, canonical record
encoding, gzip determinism — are identical rather than reimplemented.

Content seeds are **2 and 3**, not 0 and 1. ``load_regime`` draws at offset
``seed * n`` into each source dataset, so Phase 97's screen (seeds 0 and 1,
offsets 0 and 32 at n=32) and this phase (offsets 64 and 96) carry disjoint
prompt sets. Phase 98 requires fresh phase-local prompts precisely so its
acceptance measurements are not taken on the content the Phase 97 screen
already used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
P97_SCRIPTS = REPO_ROOT / "research/97_composition_runtime/scripts"
sys.path.insert(0, str(P97_SCRIPTS))

import generate_p4_prompt_manifest as p97  # noqa: E402

MANIFEST_PATH = PHASE_DIR / "data/prereg/w98_prompt_manifest.json"
BUNDLE_PATH = PHASE_DIR / "data/prereg/w98_prompt_tokens.jsonl.gz"
MANIFEST_ID = "w98-six-regime-prompts-v1"
CONTENT_SEEDS = (2, 3)
FROZEN_DATE = "2026-08-11"


def generate(hf_home: Path, tokenizer_snapshot: Path) -> dict[str, object]:
    """Write the phase-98 manifest and bundle.

    Args:
        hf_home: The pinned HF cache root.
        tokenizer_snapshot: The pinned Qwen3-8B tokenizer snapshot.

    Returns:
        A summary with the manifest and bundle digests.

    Raises:
        AssertionError: If the reused generator did not honour the overrides.
    """
    p97.MANIFEST_PATH = MANIFEST_PATH
    p97.BUNDLE_PATH = BUNDLE_PATH
    p97.CONTENT_SEEDS = CONTENT_SEEDS
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = p97.generate(hf_home, tokenizer_snapshot)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["prompt_plan"]["content_seeds"] == list(CONTENT_SEEDS), (
        "override did not reach the prompt plan"
    )
    manifest["manifest_id"] = MANIFEST_ID
    manifest["frozen_date"] = FROZEN_DATE
    manifest["phase"] = 98
    manifest["derived_from"] = {
        "generator": (
            "research/97_composition_runtime/scripts/generate_p4_prompt_manifest.py"
        ),
        "reused_verbatim": True,
        "overrides": ["output paths", "content_seeds"],
        "disjoint_from_phase_97_seeds": [0, 1],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        "bundle_path": str(BUNDLE_PATH.relative_to(REPO_ROOT)),
        "bundle_sha256": hashlib.sha256(BUNDLE_PATH.read_bytes()).hexdigest(),
        "content_seeds": list(CONTENT_SEEDS),
        "prompt_records": manifest["prompt_plan"]["total_prompt_records"],
        "generator_summary": summary,
    }


def main() -> None:
    """Run the offline phase-98 prompt freeze."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=Path(os.environ.get("HF_HOME", "/data/smcho/huggingface")),
    )
    parser.add_argument(
        "--tokenizer-snapshot",
        type=Path,
        default=(
            Path("/data/smcho/huggingface/hub/models--Qwen--Qwen3-8B/snapshots")
            / p97.TOKENIZER_REVISION
        ),
    )
    args = parser.parse_args()
    print(json.dumps(generate(args.hf_home, args.tokenizer_snapshot), indent=2))


if __name__ == "__main__":
    main()
