#!/usr/bin/env python3
"""E0: live-engine toggle-cost anatomy (window, K) on Qwen3-8B + W4 draft.

One engine, steady b8 decode load. Toggles applied between generate()
calls via collective_rpc worker-side attribute mutation:
  window: drafter._kv_window (runtime metadata; scratchpad sized at the
          INIT cap -> only toggles <= init value are legal)
  K:      drafter.num_speculative_tokens (propose loop count; scheduler
          keeps allocating K_max lookahead slots -> the PADDED regime)
OFF <-> on is a scheduler-level per-step primitive in stock vLLM
(disable_by_batch_size; this fork also has
num_speculative_tokens_per_batch_size) -- free by construction, no
worker state to touch; cited, not re-measured.

Per phase: tok/s (3 timed rounds after 1 warm), accept len from metric
deltas, toggle RPC latency, first-round blip, worker memory.
Chain: piecewise-CG (no draft FULLCG) -- window/K ride dynamic shapes;
the FULLCG capture-set cost is a separate line in the results doc.
"""
import json
import os
import time
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
OUT = PHASE / "data/e0_toggle.json"
CKPT = os.path.expanduser("/data/smcho/ckpts/Qwen3-8B-W4A16-INT4")
PROMPTS = PHASE.parent / "57_large_ep_spec_strategy/data/prompts_ondist.txt"


def set_window(worker, w):
    d = worker.model_runner.drafter
    d._kv_window = w
    return (type(worker).__name__, type(d).__name__, d._kv_window)


def set_k(worker, k):
    d = worker.model_runner.drafter
    d.num_speculative_tokens = k
    return (type(d).__name__, d.num_speculative_tokens)


def mem_gb(worker):
    import torch
    return round(torch.cuda.memory_allocated() / 2**30, 3)


def spec_counters(llm):
    acc = drafts = 0
    for m in llm.get_metrics():
        if m.name == "vllm:spec_decode_num_accepted_tokens":
            acc = m.value
        elif m.name == "vllm:spec_decode_num_drafts":
            drafts = m.value
    return acc, drafts


def main():
    from vllm import LLM, SamplingParams
    prompts = [ln.strip() for ln in PROMPTS.read_text().splitlines()
               if ln.strip()][:8]
    prompts = [p[:6000] for p in prompts]
    llm = LLM(model="Qwen/Qwen3-8B",
              speculative_config={"method": "draft_model", "model": CKPT,
                                  "num_speculative_tokens": 6,
                                  "draft_tensor_parallel_size": 1},
              max_model_len=12288, gpu_memory_utilization=0.90,
              max_num_seqs=16, enable_prefix_caching=True,
              disable_log_stats=False)
    sp = SamplingParams(max_tokens=96, ignore_eos=True, temperature=0)

    def one_round():
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        return ntok / dt, dt

    results = []

    def phase(name, toggle=None, arg=None):
        t_rpc = 0.0
        if toggle is not None:
            t0 = time.perf_counter()
            ret = llm.collective_rpc(toggle, args=(arg,))
            t_rpc = time.perf_counter() - t0
            print(f"[E0-probe] {name}: rpc -> {ret}", flush=True)
        a0, d0 = spec_counters(llm)
        first_tps, first_dt = one_round()     # the blip round
        rates = []
        for _ in range(3):
            tps, _ = one_round()
            rates.append(tps)
        a1, d1 = spec_counters(llm)
        acc = 1 + (a1 - a0) / max(1, d1 - d0)
        mem = llm.collective_rpc(mem_gb)[0]
        steady = sum(rates) / len(rates)
        row = dict(phase=name, toggle_rpc_ms=round(t_rpc * 1e3, 2),
                   first_round_toks=round(first_tps, 1),
                   steady_toks=round(steady, 1),
                   blip_pct=round(100 * (1 - first_tps / steady), 1),
                   accept_len=round(acc, 3), mem_gb=mem)
        print("[E0]", json.dumps(row), flush=True)
        results.append(row)

    phase("base_win512_K6")
    phase("win64_control", set_window, 64)
    phase("win512_restore", set_window, 512)
    phase("K4", set_k, 4)
    phase("K2", set_k, 2)
    phase("K6_restore", set_k, 6)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
