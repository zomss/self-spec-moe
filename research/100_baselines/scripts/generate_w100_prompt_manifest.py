#!/usr/bin/env python3
"""Freeze the W100 final-grid prompt sets (registration item).

Materializes every cell's maximal prompt set (seed 0) through the pinned
loaders, tokenizes with the registered target tokenizer, and writes:

* `data/registration/w100_prompts.jsonl.gz` -- one row per prompt:
  {cell, prompt_index, sha256 (of prompt text), n_tokens, token_ids}.
  Deterministic bytes (fixed gzip mtime, fixed row order), so its file
  hash is stable and the barrier can pin it.
* `data/registration/w100_prompt_manifest.json` -- dataset revision pins,
  tokenizer identity (snapshot path + tokenizer.json sha256), per-cell
  counts/caps/batch clamps/n-rule, software versions, and the prompt
  file's sha256.

Scored campaigns MUST read prompts from the frozen file, never from live
HF loads -- streaming order is stable at a pinned revision, but the
frozen file removes the dependency entirely (registration checklist,
`final_eval_design.md`).

Per-(cell, batch) request count: n = min(cell_max, max(16, 2b)), nested
(smaller-n sets are prefixes), paired across arms. LO's cell_max is 60:
AIME 2024 + 2025 hold 60 unique problems and the loader wraps modulo
beyond that; duplicated prompts would add no content.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE = SCRIPT_DIR.parent
REG = PHASE / "data" / "registration"
sys.path.insert(0, str(SCRIPT_DIR))

TARGET_SNAPSHOT = (
    "/h/v-sukmincho/.cache/huggingface/hub/models--Qwen--Qwen3-8B/"
    "snapshots/b968826d9c46dd6066d109eabc6255188de91218"
)
CONTENT_SEED = 0

# cell -> (max_n, batches). n per batch = min(max_n, max(16, 2b)), nested.
CELL_PLAN = {
    "LI": (32, [1, 8, 16]),
    "LO": (60, [1, 8, 16, 32]),
    "LIO": (32, [1, 8, 16]),
    "SS": (128, [1, 8, 32, 64]),
}
LO_T1 = {"batch": 8, "n": 16, "temperature": 1.0, "seed": 1234}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    from transformers import AutoTokenizer

    import w100_eval_datasets as wed

    tok = AutoTokenizer.from_pretrained(TARGET_SNAPSHOT)
    REG.mkdir(parents=True, exist_ok=True)

    rows, cells_meta = [], {}
    for cell, (max_n, batches) in CELL_PLAN.items():
        prompts, spec = wed.load_cell(cell, tok, n=max_n, seed=CONTENT_SEED)
        assert len(prompts) == max_n, (cell, len(prompts))
        for i, p in enumerate(prompts):
            ids = tok.encode(p)
            rows.append({"cell": cell, "prompt_index": i,
                         "sha256": _sha(p.encode()),
                         "n_tokens": len(ids), "token_ids": ids})
        cells_meta[cell] = {
            "max_n": max_n, "batches": batches,
            "n_rule": "min(max_n, max(16, 2*batch)), nested prefixes",
            "cap_max_tokens": spec["max_tokens"],
            "temperature": spec["temperature"],
            "thinking": spec["thinking"],
            "prompt_tokens_min": min(
                r["n_tokens"] for r in rows if r["cell"] == cell),
            "prompt_tokens_max": max(
                r["n_tokens"] for r in rows if r["cell"] == cell),
        }
        print(f"[freeze] {cell}: {max_n} prompts, "
              f"tok {cells_meta[cell]['prompt_tokens_min']}"
              f"-{cells_meta[cell]['prompt_tokens_max']}", flush=True)

    payload = "".join(
        json.dumps(r, separators=(",", ":")) + "\n" for r in rows
    ).encode()
    gz_path = REG / "w100_prompts.jsonl.gz"
    with open(gz_path, "wb") as fh:
        with gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:
            gz.write(payload)

    import datasets as ds_lib
    import transformers
    import vllm

    manifest = {
        "schema_version": 1,
        "phase": "100_baselines",
        "manifest_id": "w100-final-grid-prompts-v1",
        "frozen_date": "2026-08-16",
        "content_seed": CONTENT_SEED,
        "dataset_pins": wed.PINS,
        "bands": {"LI": wed.LI_BAND, "LIO": wed.LIO_BAND},
        "tokenizer": {
            "snapshot": TARGET_SNAPSHOT,
            "tokenizer_json_sha256": _sha(
                (Path(TARGET_SNAPSHOT) / "tokenizer.json").read_bytes()),
        },
        "cells": cells_meta,
        "lo_t1_arm": LO_T1,
        "prompt_file": {
            "path": "data/registration/w100_prompts.jsonl.gz",
            "sha256": _sha(gz_path.read_bytes()),
            "rows": len(rows),
            "payload_sha256": _sha(payload),
        },
        "software": {
            "transformers": transformers.__version__,
            "datasets": ds_lib.__version__,
            "vllm": vllm.__version__,
        },
    }
    man_path = REG / "w100_prompt_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(f"[freeze] {len(rows)} rows -> {gz_path.name}; "
          f"manifest -> {man_path.name}", flush=True)


if __name__ == "__main__":
    main()
