#!/usr/bin/env python3
"""Freeze the exact CPU-only prompt-token bundle for the P4 B0 screen."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
LOADER_PATH = REPO_ROOT / "research/88_regime_eval/scripts/regime_datasets.py"
MANIFEST_PATH = PHASE_DIR / "data/p4/p4_b0_prompt_manifest.json"
BUNDLE_PATH = PHASE_DIR / "data/p4/p4_b0_prompt_tokens.jsonl.gz"

MODEL_ID = "Qwen/Qwen3-8B"
TOKENIZER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
LOADER_SHA256 = "d072fc7e8865c0d8406ad01bb6f67b91b6c8ca8426678bc68cd1fbf77ace94e8"
REGIME_ORDER = ("R4", "R5", "R5cot", "R8", "R1", "R6")
CONTENT_SEEDS = (0, 1)
PROMPTS_PER_GROUP = 32
GENERATION_SEED = 0
MAX_MODEL_LEN = 20480
EXPECTED_SOFTWARE = {
    "datasets": "5.0.0",
    "tokenizers": "0.22.2",
    "transformers": "5.5.4",
}
REGIMES = {
    "R4": {
        "batch": 8,
        "max_output_tokens": 512,
        "temperature": 0.0,
        "context_target_tokens": 8000,
    },
    "R5": {
        "batch": 8,
        "max_output_tokens": 512,
        "temperature": 0.0,
        "context_target_tokens": 14000,
    },
    "R5cot": {
        "batch": 8,
        "max_output_tokens": 3072,
        "temperature": 0.0,
        "context_target_tokens": 14000,
    },
    "R8": {
        "batch": 16,
        "max_output_tokens": 2048,
        "temperature": 1.0,
        "context_target_tokens": None,
    },
    "R1": {
        "batch": 1,
        "max_output_tokens": 1024,
        "temperature": 0.0,
        "context_target_tokens": None,
    },
    "R6": {
        "batch": 32,
        "max_output_tokens": 256,
        "temperature": 0.0,
        "context_target_tokens": None,
    },
}
TOKENIZER_FILES = {
    "merges.txt": {
        "bytes": 1671853,
        "sha256": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    },
    "tokenizer.json": {
        "bytes": 11422654,
        "sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    },
    "tokenizer_config.json": {
        "bytes": 9732,
        "sha256": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    },
    "vocab.json": {
        "bytes": 2776833,
        "sha256": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    },
}
DATASETS = (
    {
        "dataset_id": "aime_1983_2024",
        "repo_id": "di-zhang-fdu/AIME_1983_2024",
        "repo_cache": "datasets--di-zhang-fdu--AIME_1983_2024",
        "revision": "3e2cc86390666c5c756622afc0eeb9e6194496bc",
        "config": "default",
        "split": "train",
        "source_format": "datasets-arrow-cache",
        "fingerprint": "e3d8dec87297b9de",
        "used_by_regimes": ["R5cot", "R8", "R1"],
        "files": [
            {
                "path_from_hf_home": (
                    "datasets/di-zhang-fdu___aime_1983_2024/default/0.0.0/"
                    "3e2cc86390666c5c756622afc0eeb9e6194496bc/"
                    "aime_1983_2024-train.arrow"
                ),
                "bytes": 359840,
                "sha256": (
                    "0994f7ec89c8d5e1145bafe475a1a35cc516387d8dd2a90a81b2bf4ef97069c7"
                ),
            }
        ],
    },
    {
        "dataset_id": "c4",
        "repo_id": "allenai/c4",
        "repo_cache": "datasets--allenai--c4",
        "revision": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
        "config": "en",
        "split": "train-shard-00000-of-01024",
        "source_format": "raw-gzip-snapshot",
        "fingerprint": None,
        "used_by_regimes": ["R5", "R5cot"],
        "files": [
            {
                "path_from_hf_home": (
                    "hub/datasets--allenai--c4/snapshots/"
                    "1588ec454efa1a09f29cd18ddd04fe05fc8653a2/en/"
                    "c4-train.00000-of-01024.json.gz"
                ),
                "bytes": 319308785,
                "sha256": (
                    "8ef8d75b0e045dec4aa5123a671b4564466b0707086a7ed1ba8721626dfffbc9"
                ),
            }
        ],
    },
    {
        "dataset_id": "cnn_dailymail",
        "repo_id": "abisee/cnn_dailymail",
        "repo_cache": "datasets--abisee--cnn_dailymail",
        "revision": "96df5e686bee6baa90b8bee7c28b81fa3fa6223d",
        "config": "3.0.0",
        "split": "test[:2000]",
        "source_format": "datasets-arrow-cache",
        "fingerprint": "561ad8a9f5f90b8d",
        "used_by_regimes": ["R4"],
        "files": [
            {
                "path_from_hf_home": (
                    "datasets/abisee___cnn_dailymail/3.0.0/0.0.0/"
                    "96df5e686bee6baa90b8bee7c28b81fa3fa6223d/"
                    "cnn_dailymail-test.arrow"
                ),
                "bytes": 49929984,
                "sha256": (
                    "e62a19168b7aa96e07d4ac6531ec222c985cf2335f54cabf1751b27147e7c266"
                ),
            }
        ],
    },
    {
        "dataset_id": "gsm8k",
        "repo_id": "openai/gsm8k",
        "repo_cache": "datasets--openai--gsm8k",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "config": "main",
        "split": "test",
        "source_format": "datasets-arrow-cache",
        "fingerprint": "59ec1b7f9357c7a2",
        "used_by_regimes": ["R1", "R6"],
        "files": [
            {
                "path_from_hf_home": (
                    "datasets/openai___gsm8k/main/0.0.0/"
                    "740312add88f781978c0658806c59bc2815b9866/gsm8k-test.arrow"
                ),
                "bytes": 714584,
                "sha256": (
                    "45965b000311d1550e5619b60b5bf31cf76edebfd8b8eddc62a876fbf8c9be95"
                ),
            }
        ],
    },
    {
        "dataset_id": "nq_open",
        "repo_id": "google-research-datasets/nq_open",
        "repo_cache": "datasets--google-research-datasets--nq_open",
        "revision": "5dd9790a83002ad084ddeb7c420dc716852c6f28",
        "config": "nq_open",
        "split": "validation",
        "source_format": "datasets-arrow-cache",
        "fingerprint": "0d657f6528371f13",
        "used_by_regimes": ["R5"],
        "files": [
            {
                "path_from_hf_home": (
                    "datasets/google-research-datasets___nq_open/nq_open/0.0.0/"
                    "5dd9790a83002ad084ddeb7c420dc716852c6f28/"
                    "nq_open-validation.arrow"
                ),
                "bytes": 315480,
                "sha256": (
                    "0b9b28b5989c0f5a1ea58db80db6e25ad515518d57aed67631315a1fb0114dcc"
                ),
            }
        ],
    },
)


class PromptGenerationError(RuntimeError):
    """Raised when a supposedly frozen prompt input has drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PromptGenerationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path: Path, expected: Mapping[str, Any]) -> None:
    _require(path.is_file(), f"missing frozen input: {path}")
    _require(path.stat().st_size == expected["bytes"], f"size drift for {path}")
    actual = _sha256(path)
    _require(actual == expected["sha256"], f"hash drift for {path}: {actual}")


def _verify_software() -> dict[str, str]:
    _require(platform.python_version().startswith("3.12."), "Python 3.12 is required")
    observed = {name: importlib.metadata.version(name) for name in EXPECTED_SOFTWARE}
    _require(observed == EXPECTED_SOFTWARE, f"software version drift: {observed}")
    return {"python": platform.python_version(), **observed}


def _verify_inputs(hf_home: Path, tokenizer_snapshot: Path) -> None:
    _require(_sha256(LOADER_PATH) == LOADER_SHA256, "canonical regime loader drifted")
    _require(
        tokenizer_snapshot.name == TOKENIZER_REVISION, "tokenizer revision drifted"
    )
    for name, expected in TOKENIZER_FILES.items():
        _verify_file(tokenizer_snapshot / name, expected)

    expected_c4_files: set[Path] = set()
    for dataset in DATASETS:
        ref_path = hf_home / "hub" / dataset["repo_cache"] / "refs" / "main"
        _require(ref_path.is_file(), f"missing dataset ref: {ref_path}")
        _require(
            ref_path.read_text().strip() == dataset["revision"],
            f"dataset revision drift for {dataset['repo_id']}",
        )
        for expected in dataset["files"]:
            path = hf_home / expected["path_from_hf_home"]
            _verify_file(path, expected)
            if dataset["dataset_id"] == "c4":
                expected_c4_files.add(path)

    observed_c4_files = set(
        (hf_home / "hub" / "datasets--allenai--c4").glob("**/*.json.gz")
    )
    _require(
        observed_c4_files == expected_c4_files,
        "the canonical C4 glob no longer resolves to exactly the frozen shard",
    )


def _load_regime_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("p4_regime_datasets", LOADER_PATH)
    _require(
        spec is not None and spec.loader is not None, "cannot import regime loader"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pin_dataset_loader(hf_home: Path) -> None:
    import datasets

    original = datasets.load_dataset
    by_repo = {dataset["repo_id"]: dataset for dataset in DATASETS}

    def pinned_load_dataset(path: str, *args: Any, **kwargs: Any) -> Any:
        _require(path in by_repo, f"unregistered dataset requested: {path}")
        expected = by_repo[path]
        kwargs["revision"] = expected["revision"]
        value = original(path, *args, **kwargs)
        _require(
            value._fingerprint == expected["fingerprint"],
            f"dataset fingerprint drift for {path}: {value._fingerprint}",
        )
        expected_files = {
            (hf_home / item["path_from_hf_home"]).resolve()
            for item in expected["files"]
        }
        actual_files = {Path(item["filename"]).resolve() for item in value.cache_files}
        _require(actual_files == expected_files, f"dataset cache drift for {path}")
        return value

    datasets.load_dataset = pinned_load_dataset


def _token_bytes(token_ids: Sequence[int]) -> bytes:
    _require(bool(token_ids), "empty prompt token sequence")
    _require(
        all(type(token_id) is int and 0 <= token_id < 2**32 for token_id in token_ids),
        "token ids must be uint32 values",
    )
    return struct.pack(f"<{len(token_ids)}I", *token_ids)


def _canonical_record(record: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )


def _gzip(payload: bytes) -> bytes:
    target = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=9,
        fileobj=target,
        mtime=0,
    ) as stream:
        stream.write(payload)
    return target.getvalue()


def _dataset_manifest() -> list[dict[str, Any]]:
    return [
        {key: value for key, value in dataset.items() if key != "repo_cache"}
        for dataset in DATASETS
    ]


def generate(hf_home: Path, tokenizer_snapshot: Path) -> dict[str, Any]:
    """Generate and write the exact manifest and compressed token bundle.

    Args:
        hf_home: Hugging Face cache root containing the frozen inputs.
        tokenizer_snapshot: Exact Qwen3-8B tokenizer snapshot directory.

    Returns:
        Compact generation summary.

    Raises:
        PromptGenerationError: If an input, prompt contract, or token id drifts.
    """
    software = _verify_software()
    _verify_inputs(hf_home, tokenizer_snapshot)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_snapshot,
        local_files_only=True,
    )
    _require(type(tokenizer).__name__ == "Qwen2Tokenizer", "tokenizer class drifted")
    _require(tokenizer.vocab_size == 151643, "tokenizer vocabulary drifted")
    _pin_dataset_loader(hf_home)
    regime_module = _load_regime_module()

    records: list[dict[str, Any]] = []
    prompt_manifest: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for regime_id in REGIME_ORDER:
        expected_spec = REGIMES[regime_id]
        for content_seed in CONTENT_SEEDS:
            prompts, observed_spec = regime_module.load_regime(
                regime_id,
                tokenizer,
                n=PROMPTS_PER_GROUP,
                seed=content_seed,
            )
            _require(
                len(prompts) == PROMPTS_PER_GROUP,
                f"{regime_id}/seed-{content_seed} returned {len(prompts)} prompts",
            )
            _require(
                observed_spec
                == {
                    "batch": expected_spec["batch"],
                    "max_tokens": expected_spec["max_output_tokens"],
                    "temperature": expected_spec["temperature"],
                },
                f"generation contract drift for {regime_id}: {observed_spec}",
            )

            group_digest = hashlib.sha256()
            lengths: list[int] = []
            for prompt_index, prompt in enumerate(prompts):
                token_ids = tokenizer.encode(prompt)
                token_payload = _token_bytes(token_ids)
                token_digest = hashlib.sha256(token_payload).hexdigest()
                record_id = f"{regime_id}-s{content_seed}-p{prompt_index:03d}"
                _require(
                    len(token_ids) + expected_spec["max_output_tokens"]
                    <= MAX_MODEL_LEN,
                    f"{record_id} exceeds the registered model length",
                )
                record = {
                    "batch": expected_spec["batch"],
                    "content_seed": content_seed,
                    "generation_seed": GENERATION_SEED,
                    "ignore_eos": True,
                    "max_output_tokens": expected_spec["max_output_tokens"],
                    "prompt_index": prompt_index,
                    "record_id": record_id,
                    "regime_id": regime_id,
                    "temperature": expected_spec["temperature"],
                    "token_ids": token_ids,
                }
                records.append(record)
                prompt_manifest.append(
                    {
                        "record_id": record_id,
                        "regime_id": regime_id,
                        "content_seed": content_seed,
                        "prompt_index": prompt_index,
                        "token_count": len(token_ids),
                        "token_ids_sha256": token_digest,
                    }
                )
                lengths.append(len(token_ids))
                group_digest.update(struct.pack("<I", len(token_ids)))
                group_digest.update(token_payload)

            groups.append(
                {
                    "regime_id": regime_id,
                    "content_seed": content_seed,
                    "prompt_count": PROMPTS_PER_GROUP,
                    "minimum_prompt_tokens": min(lengths),
                    "maximum_prompt_tokens": max(lengths),
                    "total_prompt_tokens": sum(lengths),
                    "token_ids_sha256": group_digest.hexdigest(),
                }
            )

    payload = b"".join(_canonical_record(record) for record in records)
    compressed = _gzip(payload)
    _require(compressed[4:8] == b"\x00\x00\x00\x00", "gzip mtime is not zero")
    BUNDLE_PATH.write_bytes(compressed)

    manifest = {
        "schema_version": 1,
        "manifest_id": "p4-b0-six-regime-prompts-v1",
        "status": "frozen",
        "frozen_date": "2026-08-09",
        "source_artifacts": {
            "regime_loader": {
                "path": "research/88_regime_eval/scripts/regime_datasets.py",
                "sha256": LOADER_SHA256,
            },
            "prompt_generator": {
                "path": (
                    "research/97_composition_runtime/scripts/"
                    "generate_p4_prompt_manifest.py"
                ),
                "sha256": _sha256(Path(__file__)),
            },
        },
        "software": software,
        "tokenizer": {
            "model_id": MODEL_ID,
            "revision": TOKENIZER_REVISION,
            "class_name": type(tokenizer).__name__,
            "vocab_size": tokenizer.vocab_size,
            "tokenization_call": "tokenizer.encode(prompt)",
            "files": [
                {"name": name, **values}
                for name, values in sorted(TOKENIZER_FILES.items())
            ],
        },
        "datasets": _dataset_manifest(),
        "prompt_plan": {
            "regime_order": list(REGIME_ORDER),
            "content_seeds": list(CONTENT_SEEDS),
            "prompts_per_seed_and_regime": PROMPTS_PER_GROUP,
            "total_prompt_records": len(records),
            "generation_seed": GENERATION_SEED,
            "generation_seed_policy": "same_zero_seed_per_prompt_across_actions",
            "ignore_eos": True,
            "max_model_len": MAX_MODEL_LEN,
            "regimes": [
                {"regime_id": regime_id, **REGIMES[regime_id]}
                for regime_id in REGIME_ORDER
            ],
        },
        "hashing": {
            "algorithm": "sha256",
            "token_id_encoding": "little-endian-uint32",
            "group_encoding": ("little-endian-uint32-count-then-token-ids-per-prompt"),
            "json_encoding": "utf-8",
            "jsonl_canonicalization": (
                "sort_keys,separators=(comma,colon),ensure_ascii=false,trailing_newline"
            ),
        },
        "bundle": {
            "path": (
                "research/97_composition_runtime/data/p4/p4_b0_prompt_tokens.jsonl.gz"
            ),
            "format": "canonical-jsonl",
            "compression": "gzip",
            "compression_level": 9,
            "gzip_mtime": 0,
            "record_count": len(records),
            "uncompressed_bytes": len(payload),
            "uncompressed_sha256": hashlib.sha256(payload).hexdigest(),
            "compressed_bytes": len(compressed),
            "sha256": hashlib.sha256(compressed).hexdigest(),
        },
        "groups": groups,
        "prompts": prompt_manifest,
        "authorizations": {
            "exact_prompt_manifest_frozen": True,
            "gpu_measurement": False,
            "p4a_engineering": False,
            "action_admission": False,
            "production_value_claim": False,
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "generated",
        "manifest": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "bundle": str(BUNDLE_PATH.relative_to(REPO_ROOT)),
        "records": len(records),
        "prompt_tokens": sum(item["token_count"] for item in prompt_manifest),
        "bundle_sha256": manifest["bundle"]["sha256"],
    }


def main() -> None:
    """Run the offline prompt freeze."""
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
            / TOKENIZER_REVISION
        ),
    )
    args = parser.parse_args()
    print(json.dumps(generate(args.hf_home, args.tokenizer_snapshot), indent=2))


if __name__ == "__main__":
    main()
