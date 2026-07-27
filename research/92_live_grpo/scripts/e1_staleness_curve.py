#!/usr/bin/env python3
"""Phase 92 E1: REAL staleness curve — accept vs GRPO training step.

Stage-2 measurement on OUR stack. Targets are the per-step HF policy
dumps from the stage-1 veRL run (EfficientRollout stack); the drafter
is a fixed W4 RTN quantization of the step-0 policy (base Qwen2.5-7B).

  accept(drafter@0, policy@k)  vs k   -> the real staleness curve
  accept(drafter@k, policy@k)         -> the refresh ceiling (pass a
                                         matching E92_DRAFT)

Prompts = the actual training distribution (simplerl-8k-hard-qwen
train.parquet), T=1.0 — on-policy measurement per the phase-90 theorem.

Env: E92_TARGET (HF dir or Qwen/Qwen2.5-7B), E92_DRAFT, E92_K,
     E92_TAG, E92_NPROMPTS, E92_MAXTOK, E92_SEED, E92_OFFSET.
Appends one JSON line per run -> data/e1_curve.jsonl.
"""
import json
import os
import time
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
TARGET = os.environ.get("E92_TARGET", "Qwen/Qwen2.5-7B")
DRAFT = os.path.expanduser(os.environ.get(
    "E92_DRAFT", "~/ckpts/Qwen2.5-7B-W4A16-INT4-sym"))
K = int(os.environ.get("E92_K", "4"))
TAG = os.environ.get("E92_TAG", "step?")
N_PROMPTS = int(os.environ.get("E92_NPROMPTS", "16"))
MAX_TOK = int(os.environ.get("E92_MAXTOK", "1024"))
OFFSET = int(os.environ.get("E92_OFFSET", "0"))  # prompt window start


def main():
    import pandas as pd

    from vllm import LLM, SamplingParams

    df = pd.read_parquet(
        "/data/smcho/efficientrollout/data/simplerl-8k-hard-qwen/"
        "train.parquet")
    # base-model prompt style: raw user content, no chat template
    prompts = [r[0]["content"] for r in
               df["prompt"].iloc[OFFSET:OFFSET + N_PROMPTS]]

    spec = None
    if os.environ.get("E92_SPEC", "1") == "1":
        spec = {"method": "draft_model", "model": DRAFT,
                "num_speculative_tokens": K,
                "draft_tensor_parallel_size": 1}
    # veRL dumps are fp32; serve bf16 exactly like the veRL rollout does
    llm = LLM(model=TARGET, speculative_config=spec, dtype="bfloat16",
              tensor_parallel_size=1, max_model_len=8192,
              gpu_memory_utilization=0.90, max_num_seqs=16,
              enable_prefix_caching=False, disable_log_stats=False,
              async_scheduling=True, max_num_batched_tokens=8192)
    sp = SamplingParams(max_tokens=MAX_TOK, temperature=1.0,
                        seed=int(os.environ.get("E92_SEED", "0")))

    def counters():
        acc = drafts = tok = 0
        for m in llm.get_metrics():
            if m.name == "vllm:spec_decode_num_accepted_tokens":
                acc = m.value
            elif m.name == "vllm:spec_decode_num_drafts":
                drafts = m.value
        return acc, drafts

    llm.generate(prompts[:4], sp, use_tqdm=False)  # warmup
    a0, d0 = counters()
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    a1, d1 = counters()
    acc = 1 + (a1 - a0) / max(d1 - d0, 1)
    row = {"tag": TAG, "target": TARGET, "draft": DRAFT if spec else None,
           "k": K, "toks": round(ntok / dt, 1),
           "accept": round(acc, 3), "tokens": ntok,
           "n_drafts": d1 - d0, "seed": sp.seed, "offset": OFFSET}
    print(f"[E92] {TAG}: accept={acc:.3f} tok/s={ntok/dt:.1f}", flush=True)
    out = PHASE / "data" / "e1_curve.jsonl"
    out.parent.mkdir(exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
