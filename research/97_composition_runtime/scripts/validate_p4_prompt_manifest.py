#!/usr/bin/env python3
"""Validate the exact Phase 97 P4 B0 prompt-token bundle."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from generate_p4_prompt_manifest import (
    CONTENT_SEEDS,
    DATASETS,
    EXPECTED_SOFTWARE,
    GENERATION_SEED,
    LOADER_PATH,
    LOADER_SHA256,
    MAX_MODEL_LEN,
    PROMPTS_PER_GROUP,
    REGIME_ORDER,
    REGIMES,
    TOKENIZER_FILES,
    TOKENIZER_REVISION,
)
from jsonschema import Draft202012Validator

PHASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PHASE_DIR.parents[1]
SCHEMA_PATH = PHASE_DIR / "schemas/p4_prompt_manifest.schema.json"
MANIFEST_PATH = PHASE_DIR / "data/p4/p4_b0_prompt_manifest.json"
EXPECTED_SOURCE_PATHS = {
    "regime_loader": "research/88_regime_eval/scripts/regime_datasets.py",
    "prompt_generator": (
        "research/97_composition_runtime/scripts/generate_p4_prompt_manifest.py"
    ),
}
RECORD_FIELDS = {
    "batch",
    "content_seed",
    "generation_seed",
    "ignore_eos",
    "max_output_tokens",
    "prompt_index",
    "record_id",
    "regime_id",
    "temperature",
    "token_ids",
}


class PromptManifestError(ValueError):
    """Raised when the exact prompt package is incomplete or has drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PromptManifestError(message)


def _format_json_path(parts: Sequence[Any]) -> str:
    path = "$"
    for part in parts:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PromptManifestError(f"cannot load JSON artifact {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON artifact must be an object: {path}")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_path(relative_path: str) -> Path:
    path = (REPO_ROOT / relative_path).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise PromptManifestError(f"artifact escapes repository: {path}") from exc
    return path


def _validate_schema(manifest: Mapping[str, Any]) -> None:
    schema = _load_json(SCHEMA_PATH)
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise PromptManifestError(f"invalid prompt-manifest schema: {exc}") from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(manifest),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    if errors:
        first = errors[0]
        path = _format_json_path(list(first.absolute_path))
        raise PromptManifestError(
            f"prompt-manifest schema rejected {path}: {first.message}"
        )


def _validate_sources(manifest: Mapping[str, Any]) -> None:
    sources = manifest["source_artifacts"]
    _require(
        set(sources) == set(EXPECTED_SOURCE_PATHS),
        "prompt manifest must bind exactly the loader and generator",
    )
    for role, expected_path in EXPECTED_SOURCE_PATHS.items():
        reference = sources[role]
        _require(reference["path"] == expected_path, f"source path drift for {role}")
        path = _repository_path(expected_path)
        _require(path.is_file(), f"missing source artifact: {path}")
        actual = _sha256_file(path)
        _require(
            actual == reference["sha256"],
            f"source hash mismatch for {expected_path}: {actual}",
        )
    _require(_sha256_file(LOADER_PATH) == LOADER_SHA256, "regime loader drifted")


def _expected_datasets() -> list[dict[str, Any]]:
    return [
        {key: value for key, value in dataset.items() if key != "repo_cache"}
        for dataset in DATASETS
    ]


def _validate_provenance(manifest: Mapping[str, Any]) -> None:
    software = manifest["software"]
    _require(
        software["python"].startswith("3.12."),
        "prompt generation must use Python 3.12",
    )
    _require(
        {name: software[name] for name in EXPECTED_SOFTWARE} == EXPECTED_SOFTWARE,
        "prompt-generation software versions drifted",
    )
    tokenizer = manifest["tokenizer"]
    _require(tokenizer["revision"] == TOKENIZER_REVISION, "tokenizer revision drifted")
    _require(tokenizer["class_name"] == "Qwen2Tokenizer", "tokenizer class drifted")
    _require(tokenizer["vocab_size"] == 151643, "tokenizer vocabulary drifted")
    expected_files = [
        {"name": name, **values} for name, values in sorted(TOKENIZER_FILES.items())
    ]
    _require(tokenizer["files"] == expected_files, "tokenizer file set drifted")
    _require(manifest["datasets"] == _expected_datasets(), "dataset provenance drifted")


def _validate_plan(manifest: Mapping[str, Any]) -> None:
    plan = manifest["prompt_plan"]
    _require(plan["regime_order"] == list(REGIME_ORDER), "regime order drifted")
    _require(plan["content_seeds"] == list(CONTENT_SEEDS), "content seeds drifted")
    _require(
        plan["prompts_per_seed_and_regime"] == PROMPTS_PER_GROUP,
        "prompt count per group drifted",
    )
    _require(plan["generation_seed"] == GENERATION_SEED, "generation seed drifted")
    _require(plan["ignore_eos"], "the fixed-work prompt plan requires ignore_eos")
    _require(plan["max_model_len"] == MAX_MODEL_LEN, "model-length bound drifted")
    expected_regimes = [
        {"regime_id": regime_id, **REGIMES[regime_id]} for regime_id in REGIME_ORDER
    ]
    _require(plan["regimes"] == expected_regimes, "regime generation specs drifted")


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


def _load_bundle(manifest: Mapping[str, Any]) -> tuple[bytes, list[bytes]]:
    bundle = manifest["bundle"]
    path = _repository_path(bundle["path"])
    _require(path.is_file(), f"missing prompt-token bundle: {path}")
    compressed = path.read_bytes()
    _require(len(compressed) == bundle["compressed_bytes"], "bundle size drifted")
    _require(_sha256_bytes(compressed) == bundle["sha256"], "bundle hash mismatch")
    _require(compressed[:3] == b"\x1f\x8b\x08", "bundle is not gzip data")
    _require(compressed[3] == 0, "gzip bundle contains optional header fields")
    _require(compressed[4:8] == b"\x00\x00\x00\x00", "gzip mtime is not zero")
    try:
        payload = gzip.decompress(compressed)
    except (OSError, EOFError) as exc:
        raise PromptManifestError(f"cannot decompress token bundle: {exc}") from exc
    _require(len(payload) == bundle["uncompressed_bytes"], "payload size drifted")
    _require(
        _sha256_bytes(payload) == bundle["uncompressed_sha256"],
        "payload hash mismatch",
    )
    _require(payload.endswith(b"\n"), "canonical JSONL requires a trailing newline")
    lines = payload.splitlines(keepends=True)
    _require(len(lines) == bundle["record_count"], "bundle record count drifted")
    return payload, lines


def _validate_records(
    manifest: Mapping[str, Any], lines: Sequence[bytes]
) -> tuple[int, list[dict[str, Any]]]:
    prompt_rows = manifest["prompts"]
    _require(len(prompt_rows) == len(lines), "prompt index and bundle length differ")
    calculated_groups: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    total_tokens = 0
    line_index = 0
    for regime_id in REGIME_ORDER:
        spec = REGIMES[regime_id]
        for content_seed in CONTENT_SEEDS:
            group_digest = hashlib.sha256()
            lengths: list[int] = []
            for prompt_index in range(PROMPTS_PER_GROUP):
                raw_line = lines[line_index]
                try:
                    record = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise PromptManifestError(
                        f"invalid bundle record at line {line_index + 1}: {exc}"
                    ) from exc
                _require(isinstance(record, dict), "bundle record must be an object")
                _require(set(record) == RECORD_FIELDS, "bundle record fields drifted")
                _require(
                    raw_line == _canonical_record(record),
                    f"bundle line {line_index + 1} is not canonical JSON",
                )
                record_id = f"{regime_id}-s{content_seed}-p{prompt_index:03d}"
                expected_fields = {
                    "batch": spec["batch"],
                    "content_seed": content_seed,
                    "generation_seed": GENERATION_SEED,
                    "ignore_eos": True,
                    "max_output_tokens": spec["max_output_tokens"],
                    "prompt_index": prompt_index,
                    "record_id": record_id,
                    "regime_id": regime_id,
                    "temperature": spec["temperature"],
                }
                for key, expected in expected_fields.items():
                    _require(
                        record[key] == expected,
                        f"{record_id} has unexpected {key}",
                    )
                _require(record_id not in seen_ids, f"duplicate record id: {record_id}")
                seen_ids.add(record_id)
                token_ids = record["token_ids"]
                token_payload = _token_bytes(token_ids)
                token_count = len(token_ids)
                _require(
                    token_count + spec["max_output_tokens"] <= MAX_MODEL_LEN,
                    f"{record_id} exceeds the registered model length",
                )
                calculated_prompt = {
                    "record_id": record_id,
                    "regime_id": regime_id,
                    "content_seed": content_seed,
                    "prompt_index": prompt_index,
                    "token_count": token_count,
                    "token_ids_sha256": _sha256_bytes(token_payload),
                }
                _require(
                    prompt_rows[line_index] == calculated_prompt,
                    f"per-prompt manifest mismatch for {record_id}",
                )
                group_digest.update(struct.pack("<I", token_count))
                group_digest.update(token_payload)
                lengths.append(token_count)
                total_tokens += token_count
                line_index += 1
            calculated_groups.append(
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
    _require(line_index == len(lines), "unconsumed bundle records remain")
    return total_tokens, calculated_groups


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a parsed exact prompt manifest and its repository bundle.

    Args:
        manifest: Parsed P4 prompt-manifest object.

    Returns:
        Machine-readable prompt-freeze summary.

    Raises:
        PromptManifestError: If provenance, bytes, ordering, or authority drifted.
    """
    _validate_schema(manifest)
    _validate_sources(manifest)
    _validate_provenance(manifest)
    _validate_plan(manifest)
    _, lines = _load_bundle(manifest)
    total_tokens, calculated_groups = _validate_records(manifest, lines)
    _require(manifest["groups"] == calculated_groups, "group summaries drifted")
    for offset in range(0, len(calculated_groups), len(CONTENT_SEEDS)):
        seed_groups = calculated_groups[offset : offset + len(CONTENT_SEEDS)]
        _require(
            len({group["token_ids_sha256"] for group in seed_groups})
            == len(CONTENT_SEEDS),
            f"content seeds are not distinct for {seed_groups[0]['regime_id']}",
        )
    authorizations = manifest["authorizations"]
    _require(
        authorizations["exact_prompt_manifest_frozen"],
        "exact prompt freeze must be satisfied",
    )
    _require(
        not any(
            authorizations[key]
            for key in (
                "gpu_measurement",
                "p4a_engineering",
                "action_admission",
                "production_value_claim",
            )
        ),
        "prompt freezing cannot grant downstream authority",
    )
    return {
        "status": "pass",
        "manifest_id": manifest["manifest_id"],
        "exact_prompt_manifest_frozen": True,
        "tokenizer_revision": manifest["tokenizer"]["revision"],
        "regimes": list(REGIME_ORDER),
        "content_seeds": list(CONTENT_SEEDS),
        "record_count": len(lines),
        "total_prompt_tokens": total_tokens,
        "bundle_sha256": manifest["bundle"]["sha256"],
        "gpu_measurement_authorized": False,
        "p4a_engineering_authorized": False,
    }


def main() -> None:
    """Validate a manifest and optionally refresh its checked result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_manifest(_load_json(args.manifest))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
