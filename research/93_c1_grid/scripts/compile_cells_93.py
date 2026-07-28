#!/usr/bin/env python3
"""Phase 93 wrapper around the 82 compile protocol.

Fixes the prompt-loading bottleneck for the batch-128 grid: phase-82's
load_ctx_prompts retokenizes a growing buffer per line (quadratic) and
was called per boot for up to 128 unique docs per ctx. Here: 16 unique
C4 docs per (model, ctx), packed once with incremental token counting,
DISK-CACHED, and tiled to the requested n. Identical prompts across a
batch are fine for COST cells (prefix caching off; decode work and
greedy accept unaffected).

Usage: compile_cells_93.py --measure {off,k2,...}   (same envs as
compile_policy: COMPILE_MODEL/DRAFT/TP/BATCHES/CTXS/KV_LIMIT/CELLS)
"""
import argparse
import glob
import gzip
import json
import re
import sys
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
P82 = PHASE.parent / "82_runtime_switching" / "scripts"
sys.path.insert(0, str(P82))
import compile_policy as cp  # noqa: E402  (reads envs at import)

N_UNIQUE = 16


def _pack_docs(tok, n, target_tok):
    files = glob.glob(
        "/data/smcho/huggingface/hub/datasets--allenai--c4/**/*.json.gz",
        recursive=True)
    docs, ids = [], []
    with gzip.open(files[0], "rt") as f:
        for line in f:
            ids += tok(json.loads(line)["text"] + "\n\n").input_ids
            if len(ids) >= target_tok:
                docs.append(tok.decode(ids[:target_tok]))
                ids = []
                if len(docs) >= n:
                    return docs
    return docs


def cached_load_ctx_prompts(tok, n, target_tok):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", cp.MODEL)
    cache = PHASE / "data" / f"prompts_cache_{slug}_{target_tok}.json"
    if cache.exists():
        docs = json.loads(cache.read_text())
    else:
        docs = _pack_docs(tok, N_UNIQUE, target_tok)
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps(docs))
        print(f"[93-compile] cached {len(docs)} docs -> {cache}",
              flush=True)
    return [docs[i % len(docs)] for i in range(n)]


cp.load_ctx_prompts = cached_load_ctx_prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", required=True, choices=list(cp.ARM_K))
    a = ap.parse_args()
    cp.measure(a.measure)


if __name__ == "__main__":
    main()
