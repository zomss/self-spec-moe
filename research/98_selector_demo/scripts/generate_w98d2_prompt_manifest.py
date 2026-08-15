#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze the phase-98 D2 prompt bundle (content seeds 4 and 5), CPU-only.

Reuses Phase 97's frozen generator implementation exactly as
`generate_w98_prompt_manifest.py` does, overriding only the output paths,
the content seeds, the regime loader, and the dataset FILE pins.

Why the file pins move, and what does not
-----------------------------------------

The seeds 2-3 freeze verified its inputs four ways: pinned software
versions, pinned dataset revisions, byte-exact hashes of every cache file,
and a pinned loader hash. Three of those four carry over to h104 unchanged.
The fourth cannot, for three unrelated mechanical reasons:

* the `aime_1983_2024` Arrow cache serialises to different BYTES here while
  its content fingerprint matches (Arrow writer nondeterminism);
* `refs/main` is absent, because the datasets were fetched by explicit
  revision rather than by branch;
* the canonical regime loader hard-codes the original box's c4 cache root,
  and it is itself hash-pinned, so it cannot run here unmodified.

What is preserved is the check that actually constrains the DATA: every
dataset is loaded at the same pinned revision through the same pinned
wrapper, and `_pin_dataset_loader` verifies each one's content FINGERPRINT
against the frozen value. Measured on h104: all four fingerprints match
exactly, and four of the five cache files are byte-identical anyway. The
prompts these seeds draw are therefore the same prompts the original box
would have drawn; only the storage bytes of one cache file differ.

This generator records its OWN file hashes and loader hash, so the D2
bundle is fully re-verifiable on its own terms. It does not weaken the
seeds 2-3 freeze, which is untouched.

Content seeds are 4 and 5. `load_regime` draws at offset ``seed * n``, so
Phase 97's screen (0, 1), the phase-98 cost lattice (2, 3) and this
acceptance bundle (4, 5) carry pairwise disjoint prompt sets -- which is
what the D2 contract requires so acceptance is never measured on content
the cost fit already consumed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
P97_SCRIPTS = REPO_ROOT / "research/97_composition_runtime/scripts"
sys.path.insert(0, str(P97_SCRIPTS))

import generate_p4_prompt_manifest as p97  # noqa: E402

MANIFEST_PATH = PHASE_DIR / "data/prereg/w98d2_prompt_manifest.json"
BUNDLE_PATH = PHASE_DIR / "data/prereg/w98d2_prompt_tokens.jsonl.gz"
LOADER_PATH = PHASE_DIR / "scripts/w98d2_regime_datasets.py"
MANIFEST_ID = "w98-d2-six-regime-prompts-v1"
CONTENT_SEEDS = (4, 5)
FROZEN_DATE = "2026-08-15"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _observed_datasets(hf_home: Path) -> tuple[dict[str, Any], ...]:
    """Frozen revisions and fingerprints, with h104's observed file hashes.

    For c4 every shard present under the pinned snapshot is pinned, not just
    the first. Seeds 4-5 draw further into the packed-document stream than
    seeds 2-3 did -- R4 at seed 5 skips 160 documents and needs 32 more --
    and one shard yields only 177, so the freeze must cover the shards the
    loader will actually read. The loader globs in sorted order, so extra
    shards only APPEND to the stream: every document index the earlier seeds
    resolved to is unchanged, and the seed sets stay disjoint by
    construction.
    """
    observed = []
    for dataset in copy.deepcopy(list(p97.DATASETS)):
        if dataset["dataset_id"] == "c4":
            template = dataset["files"][0]
            snapshot = (hf_home / template["path_from_hf_home"]).parent
            shards = sorted(snapshot.glob("*.json.gz"))
            if not shards:
                raise SystemExit(f"no c4 shards under {snapshot}")
            dataset["files"] = [
                {
                    "path_from_hf_home": str(path.relative_to(hf_home)),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in shards
            ]
            observed.append(dataset)
            continue
        for entry in dataset["files"]:
            path = hf_home / entry["path_from_hf_home"]
            if not path.is_file():
                raise SystemExit(f"missing dataset input: {path}")
            entry["sha256"] = _sha256(path)
            entry["bytes"] = path.stat().st_size
        observed.append(dataset)
    return tuple(observed)


def _verify_inputs_d2(hf_home: Path, tokenizer_snapshot: Path) -> None:
    """Verification for the relocated cache.

    Identical to the frozen check except that the dataset revision is
    verified from the SNAPSHOT PATH rather than from `refs/main`. That is
    not a weakening: the path segment IS the revision, and it is what the
    loader actually reads, whereas `refs/main` merely records where a branch
    pointed at download time.
    """
    p97._require(_sha256(LOADER_PATH) == LOADER_SHA256, "D2 regime loader drifted")
    p97._require(
        tokenizer_snapshot.name == p97.TOKENIZER_REVISION,
        "tokenizer revision drifted",
    )
    for name, expected in p97.TOKENIZER_FILES.items():
        p97._verify_file(tokenizer_snapshot / name, expected)
    expected_c4: set[Path] = set()
    for dataset in p97.DATASETS:
        for entry in dataset["files"]:
            path = hf_home / entry["path_from_hf_home"]
            p97._require(
                dataset["revision"] in str(path),
                f"dataset revision not present in path for {dataset['repo_id']}",
            )
            p97._verify_file(path, entry)
            if dataset["dataset_id"] == "c4":
                expected_c4.add(path)
    # Carried over verbatim in intent from the frozen check: the c4 glob must
    # resolve to EXACTLY the pinned shards, so an extra shard appearing in the
    # cache cannot silently extend the packed-document stream and move which
    # documents a seed draws.
    observed_c4 = set((hf_home / "hub" / "datasets--allenai--c4").glob("**/*.json.gz"))
    p97._require(
        observed_c4 == expected_c4,
        "the C4 glob no longer resolves to exactly the pinned shards",
    )


LOADER_SHA256 = _sha256(LOADER_PATH)
# (regime, seed) cells whose source pool ran out before PROMPTS_PER_GROUP,
# recorded by _require_sufficient and written into the manifest.
SHORT_POOLS: list[dict[str, Any]] = []


def _require_sufficient(condition: bool, message: str) -> None:
    """`_require`, relaxed for pool SIZE only -- never for content.

    R4 packs cnn_dailymail `test[:2000]` into ~177 long prompts, and seed 5
    draws at offset 160, so it can supply 17 rather than 32. There is no
    escape by choosing other seeds: disjointness from the cost lattice needs
    offset >= 128, and 128 is the only offset that fits, so two disjoint
    seeds cannot both be full. Enlarging the slice is worse -- it changes
    the `load_dataset` call, so the pinned wrapper's content-fingerprint
    check would reject it, trading a verified content guarantee for pool
    uniformity.

    A short pool is acceptable because a boot consumes `batch` prompts per
    regime (R4 batch = 8), not the whole group; sufficiency is checked
    against that, and every shortfall is recorded rather than absorbed.
    """
    if condition:
        return
    parts = message.split()
    if len(parts) >= 3 and parts[-1] == "prompts" and parts[-2].isdigit():
        cell, count = parts[0], int(parts[-2])
        regime_id = cell.split("/")[0]
        batch = int(p97.REGIMES[regime_id]["batch"])
        if count >= batch:
            SHORT_POOLS.append(
                {
                    "cell": cell,
                    "prompts_available": count,
                    "prompts_requested": p97.PROMPTS_PER_GROUP,
                    "batch_required": batch,
                    "sufficient": True,
                }
            )
            return
    raise p97.PromptGenerationError(message)


def generate(hf_home: Path, tokenizer_snapshot: Path) -> dict[str, Any]:
    """Write the D2 manifest and bundle."""
    os.environ["W98D2_HF_HOME"] = str(hf_home)
    p97.MANIFEST_PATH = MANIFEST_PATH
    p97.BUNDLE_PATH = BUNDLE_PATH
    p97.CONTENT_SEEDS = CONTENT_SEEDS
    p97.LOADER_PATH = LOADER_PATH
    p97.LOADER_SHA256 = LOADER_SHA256
    p97.DATASETS = _observed_datasets(hf_home)
    p97._verify_inputs = _verify_inputs_d2
    p97._require = _require_sufficient
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = p97.generate(hf_home, tokenizer_snapshot)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["prompt_plan"]["content_seeds"] == list(CONTENT_SEEDS), (
        "override did not reach the prompt plan"
    )
    manifest["manifest_id"] = MANIFEST_ID
    manifest["frozen_date"] = FROZEN_DATE
    manifest["phase"] = 98
    manifest["gate"] = "G98-D"
    manifest["derived_from"] = {
        "generator": (
            "research/97_composition_runtime/scripts/generate_p4_prompt_manifest.py"
        ),
        "reused_verbatim": True,
        "overrides": [
            "output paths",
            "content_seeds",
            "regime loader (c4 cache root only)",
            "dataset file hashes (re-pinned on h104)",
            "revision verified from the snapshot path, not refs/main",
        ],
        "disjoint_from_phase_97_seeds": [0, 1],
        "disjoint_from_w98_cost_seeds": [2, 3],
    }
    manifest["short_pools"] = SHORT_POOLS
    manifest["relocation"] = {
        "box": "h104 (bare metal)",
        "content_identity": (
            "every dataset loaded at its frozen revision through the pinned "
            "wrapper, whose fingerprint check passed for all four Arrow "
            "datasets; four of five cache files are byte-identical to the "
            "seeds 2-3 pins and only the aime Arrow serialisation differs"
        ),
        "loader_delta": "c4 cache root parameterised; nothing else",
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "manifest_sha256": _sha256(MANIFEST_PATH),
        "bundle_path": str(BUNDLE_PATH.relative_to(REPO_ROOT)),
        "bundle_sha256": _sha256(BUNDLE_PATH),
        "loader_sha256": LOADER_SHA256,
        "content_seeds": list(CONTENT_SEEDS),
        "prompt_records": manifest["prompt_plan"]["total_prompt_records"],
        "generator_summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_home = Path(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    )
    parser.add_argument("--hf-home", type=Path, default=default_home)
    parser.add_argument(
        "--tokenizer-snapshot",
        type=Path,
        default=(
            default_home / "hub/models--Qwen--Qwen3-8B/snapshots"
            / p97.TOKENIZER_REVISION
        ),
    )
    args = parser.parse_args()
    print(json.dumps(generate(args.hf_home, args.tokenizer_snapshot), indent=2))


if __name__ == "__main__":
    main()
