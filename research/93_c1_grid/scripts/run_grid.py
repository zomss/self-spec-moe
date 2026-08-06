#!/usr/bin/env python3
"""Phase 93: dataset x batch-sweep serving runs, uncapped generation.

Regime v2 semantics:
  - dataset determines input/output character; generation is UNCAPPED
    (natural EOS under G93_CEILING, clip ratio recorded per run --
    the audit that the ceiling never shaped the data)
  - batch is swept (G93_BATCHES); over-capacity points run as waves
    under the engine scheduler and are recorded with preemption stats
    (capacity behavior is DATA, not failure)

Arms (composable):
  G93_DRAFT   off | self | <ckpt path>     (self = target ckpt as draft)
  G93_K       draft depth (spec arms)
  G93_WINDOW  draft KV window (0 = full; sinks default 16)
  G93_SKIP    comma layer list (search-derived sets only)

env: G93_MODEL, G93_TP, G93_DATASETS, G93_BATCHES, G93_ITERS,
     G93_CEILING (16384), G93_MAXLEN (32768), G93_NLOAD_LONG (32),
     G93_OUT, G93_TAG
"""
import json
import os
import sys
import time
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE.parents[0] / "88_regime_eval" / "scripts"))
from regime_datasets import REGIMES, load_regime  # noqa: E402

MODEL = os.environ.get("G93_MODEL", "Qwen/Qwen3-8B")
TP = int(os.environ.get("G93_TP", "1"))
DRAFT = os.environ.get("G93_DRAFT", "off")
K = int(os.environ.get("G93_K", "4"))
WINDOW = int(os.environ.get("G93_WINDOW", "0"))
SKIP = os.environ.get("G93_SKIP", "")
CEILING = int(os.environ.get("G93_CEILING", "16384"))
MAXLEN = int(os.environ.get("G93_MAXLEN", "32768"))
ITERS = int(os.environ.get("G93_ITERS", "3"))
BATCHES = [int(b) for b in os.environ.get(
    "G93_BATCHES", "1,4,8,16,32,64,128").split(",")]
LONG_CTX = {"R4", "R5", "R5cot"}   # packed-doc datasets: capacity-bound
NLOAD_LONG = int(os.environ.get("G93_NLOAD_LONG", "32"))
TAG = os.environ.get("G93_TAG", "")
# G93_TUNE=0 disables flashinfer autotune (the 40% boot lottery, 96/W1).
# Default 1 preserves the historical grid's semantics.
TUNE = os.environ.get("G93_TUNE", "1") == "1"


def counters(llm):
    vals = {"acc": 0, "drafts": 0, "preempt": 0}
    for m in llm.get_metrics():
        if m.name == "vllm:spec_decode_num_accepted_tokens":
            vals["acc"] = m.value
        elif m.name == "vllm:spec_decode_num_drafts":
            vals["drafts"] = m.value
        elif m.name == "vllm:num_preemptions":
            vals["preempt"] = m.value
    return vals


def main():
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams

    if WINDOW:
        os.environ["VLLM_SELF_SPEC_DRAFT_KV_WINDOW"] = str(WINDOW)
    if SKIP:
        os.environ["VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS"] = SKIP

    spec = None
    if DRAFT != "off":
        spec = {"method": "draft_model",
                "model": MODEL if DRAFT == "self"
                else os.path.expanduser(DRAFT),
                "num_speculative_tokens": K,
                "draft_tensor_parallel_size":
                    int(os.environ.get("G93_DRAFT_TP", str(TP)))}
    extra = {} if TUNE else {
        "kernel_config": {"enable_flashinfer_autotune": False}}
    llm = LLM(model=MODEL, speculative_config=spec,
              tensor_parallel_size=TP, max_model_len=MAXLEN,
              gpu_memory_utilization=0.90, max_num_seqs=max(BATCHES),
              enable_prefix_caching=False, disable_log_stats=False,
              async_scheduling=True, max_num_batched_tokens=8192,
              enforce_eager=os.environ.get("G93_ENFORCE_EAGER") == "1",
              trust_remote_code=True, **extra)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    want = os.environ.get("G93_DATASETS")
    rids = want.split(",") if want else REGIMES
    out = {"model": MODEL, "tag": TAG, "draft": DRAFT, "K": K if spec else 0,
           "tune": TUNE,
           "window": WINDOW, "skip": SKIP, "ceiling": CEILING,
           "draft_kv_dtype": os.environ.get(
               "VLLM_SELF_SPEC_DRAFT_KV_DTYPE", ""),
           "cells": []}
    for rid in rids:
        n_load = NLOAD_LONG if rid in LONG_CTX else max(BATCHES)
        prompts, gs = load_regime(rid, tok, n=n_load)
        plens = [len(tok.encode(p)) for p in prompts]
        # regime v2: dataset sets input/output character, NOT length --
        # natural EOS under the ceiling; temperature stays canonical
        sp = SamplingParams(max_tokens=CEILING,
                            temperature=gs["temperature"],
                            seed=0 if gs["temperature"] else None)
        llm.generate(prompts[:2], SamplingParams(
            max_tokens=64, temperature=gs["temperature"], seed=0),
            use_tqdm=False)  # warmup
        for b in BATCHES:
            if b > len(prompts):
                out["cells"].append({"rid": rid, "batch": b,
                                     "status": "not-enough-prompts",
                                     "n_prompts": len(prompts)})
                continue
            rates, accepts, clips, olens, preempts = [], [], [], [], []
            for it in range(ITERS):
                c0 = counters(llm)
                t0 = time.perf_counter()
                if b == 1:
                    outs = []
                    for p in prompts[:min(4, len(prompts))]:
                        outs += llm.generate([p], sp, use_tqdm=False)
                else:
                    outs = llm.generate(prompts[:b], sp, use_tqdm=False)
                dt = time.perf_counter() - t0
                c1 = counters(llm)
                toks = [len(o.outputs[0].token_ids) for o in outs]
                clip = sum(1 for o in outs
                           if o.outputs[0].finish_reason == "length")
                rates.append(sum(toks) / dt)
                olens += toks
                clips.append(clip / len(outs))
                preempts.append(c1["preempt"] - c0["preempt"])
                if c1["drafts"] > c0["drafts"]:
                    accepts.append(1 + (c1["acc"] - c0["acc"]) /
                                   (c1["drafts"] - c0["drafts"]))
            rates.sort()
            olens.sort()
            cell = {"rid": rid, "batch": b,
                    "toks": round(rates[len(rates) // 2], 1),
                    "all": [round(r, 1) for r in rates],
                    "accept": round(sum(accepts) / len(accepts), 3)
                    if accepts else None,
                    "clip_ratio": round(sum(clips) / len(clips), 3),
                    "out_tok_p50": olens[len(olens) // 2],
                    "out_tok_p95": olens[int(len(olens) * 0.95) - 1],
                    "prompt_tok_p50": sorted(plens)[len(plens) // 2],
                    "preemptions": sum(preempts),
                    "temp": gs["temperature"]}
            out["cells"].append(cell)
            print(f"[G93] {rid} b={b} toks={cell['toks']} "
                  f"accept={cell['accept']} clip={cell['clip_ratio']} "
                  f"out_p50={cell['out_tok_p50']} "
                  f"preempt={cell['preemptions']}", flush=True)
        # incremental save: a timeout mid-sweep must not destroy the
        # datasets already measured (lost 25 cells to a looping-model
        # 6h-timeout on MLA before this)
        _p = os.environ.get("G93_OUT", str(
            PHASE / "data" / f"grid_{TAG or 'run'}.json"))
        Path(_p).parent.mkdir(exist_ok=True)
        Path(_p).write_text(json.dumps(out, indent=1))
    out["complete"] = True
    path = os.environ.get("G93_OUT", str(
        PHASE / "data" / f"grid_{TAG or 'run'}.json"))
    Path(path).parent.mkdir(exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=1))
    print("[G93] saved ->", path, flush=True)


if __name__ == "__main__":
    main()
