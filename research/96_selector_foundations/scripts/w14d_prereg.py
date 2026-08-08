#!/usr/bin/env python3
"""W14/D0 — materialize and freeze the D registration bundle.

Writes data/w14/w14d_prereg.json with EXACT prompt token IDs and their
SHA-256 hashes, train/held-out labels, the frozen boot-block and cell
orders, registered graph strata, and software/model revisions.

A context label such as ctx=8192 is a design target; the registered
artifact carries the exact token IDs actually used, so `Q` is never
inferred from a tokenizer median.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE.parent / "88_regime_eval/scripts"))

MODEL = "Qwen/Qwen3-8B"
DRAFT = "/data/smcho/ckpts/Qwen3-8B-W4A8-gptq"
TRAIN_CTX = [2048, 8192, 14336]
HELD_CTX = [5120, 11264]
SEEDS = {  # (ctx) -> (R5 seed, R5cot seed), frozen per the plan
    2048: (2, 5), 8192: (3, 6), 14336: (4, 7),
    5120: (8, 10), 11264: (9, 11),
}
BLOCK_ORDER = [
    ["w512-K2", "w-off-K4", "w2048-K2", "w512-K4", "AR", "w2048-K4", "w-off-K2"],
    ["w2048-K4", "w-off-K4", "w512-K4", "w-off-K2", "w2048-K2", "w512-K2", "AR"],
    ["w-off-K4", "w512-K4", "w-off-K2", "AR", "w2048-K4", "w512-K2", "w2048-K2"],
]
CTX_ORDER = [
    [2048, 5120, 8192, 11264, 14336],
    [8192, 14336, 5120, 2048, 11264],
    [11264, 8192, 2048, 14336, 5120],
]
STRATA = [  # action -> query tokens/request, and target tokens at b1/b8
    {"action": "AR", "q_per_req": 1, "b1": 1, "b8": 8},
    {"action": "K2", "q_per_req": 3, "b1": 3, "b8": 24},
    {"action": "K4", "q_per_req": 5, "b1": 5, "b8": 40},
]


def git_rev():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PHASE.parents[1],
            text=True).strip()
    except Exception:
        return "unknown"


def main():
    from transformers import AutoTokenizer

    from regime_datasets import load_regime  # noqa: E402
    tok = AutoTokenizer.from_pretrained(MODEL)

    prompts = {}
    for ctx in TRAIN_CTX + HELD_CTX:
        split = "train" if ctx in TRAIN_CTX else "held_out"
        for rid, seed in zip(("R5", "R5cot"), SEEDS[ctx]):
            # draw content, then truncate/verify to the EXACT token count
            ps, _ = load_regime(rid, tok, n=8, seed=seed)
            ids = tok.encode(ps[0])
            if len(ids) < ctx:
                # extend deterministically by re-drawing more of the same
                # source until the exact length is reachable
                j = 1
                while len(ids) < ctx and j < len(ps):
                    ids = ids + tok.encode(ps[j]); j += 1
            ids = ids[:ctx]
            if len(ids) != ctx:
                print(f"  WARN {rid}@{ctx}: only {len(ids)} tokens available")
            h = hashlib.sha256(
                json.dumps(ids).encode()).hexdigest()
            prompts[f"{rid}_{ctx}"] = {
                "rid": rid, "ctx_target": ctx, "n_tokens": len(ids),
                "split": split, "seed": seed, "sha256": h,
                "token_ids": ids,
            }
            print(f"  {rid:<6} ctx={ctx:<6} split={split:<9} "
                  f"n={len(ids):<6} sha={h[:16]}")

    bundle = {
        "registered_utc": "2026-08-08",
        "git_rev": git_rev(),
        "model": MODEL, "draft": DRAFT,
        "configurations": ["w512", "w2048", "w-off"],
        "K": [2, 4], "batches": [1, 8],
        "gen_interval_tokens": 256, "ignore_eos": True,
        "iters": 4, "temperature": 0.0, "tune": False,
        "train_ctx": TRAIN_CTX, "held_out_ctx": HELD_CTX,
        "block_order": BLOCK_ORDER, "ctx_order": CTX_ORDER,
        "strata": STRATA,
        "watchdog_s": 400,
        "n_valid_scored_boots": 21,
        "eps_arm": 0.015, "eps_sel": 0.015,
        "prompts": prompts,
    }
    d = PHASE / "data" / "w14"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "w14d_prereg.json"
    p.write_text(json.dumps(bundle, indent=1))
    body = {k: v for k, v in bundle.items() if k != "prompts"}
    body["prompt_hashes"] = {k: v["sha256"] for k, v in prompts.items()}
    print(f"\nbundle sha256 (excl. token ids): "
          f"{hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:32]}")
    print("saved ->", p, f"({p.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
