#!/usr/bin/env python3
"""Phase 91 E6: GRPO-shaped rollout benchmark (the EfficientRollout
comparison on our stack).

Shape = their per-GPU rollout: N_PROMPTS math prompts x GROUP samples
(SamplingParams(n=GROUP)), T=1.0, MAX_TOK budget, natural EOS -> the
batch drains from ~max_num_seqs down to the long tail. One rollout =
one generate call; we report wall time, tokens, tok/s, accept.

Arms (E6_ARM):
  off          AR baseline
  policy       full system, FRESH drafter (their Table-2 comparison:
               they re-quantize per step = always fresh)
  rl_refresh   3 "RL steps": drift injected before each rollout
               (eps 0, .25, .35), background detector fires the DRAM
               refresh mid-rollout -- the drift-robust variant
  rl_stale     same drift schedule, NO refresh (staleness cost at the
               realistic shape)
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, "research/89_dram_lever_swap/scripts")
from e3_rl_demo import SNAP, rpc_dump, rpc_swap_noisy  # noqa: E402

ARM = os.environ.get("E6_ARM", "off")
N_PROMPTS = int(os.environ.get("E6_PROMPTS", "16"))
GROUP = int(os.environ.get("E6_GROUP", "8"))
MAX_TOK = int(os.environ.get("E6_MAXTOK", "8192"))
GATE = float(os.environ.get("E6_GATE", "2.45"))
DRIFTS = [float(x) for x in os.environ.get(
    "E6_DRIFTS", "0,0.25,0.35").split(",")]


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams

    spec = None
    if ARM != "off":
        spec = {"method": "draft_model",
                "model": os.path.expanduser("~/ckpts/Qwen3-8B-W4A8-gptq"),
                "num_speculative_tokens": int(os.environ.get("E6_K", "8")),
                "draft_tensor_parallel_size": 1}
    llm = LLM(model="Qwen/Qwen3-8B", speculative_config=spec,
              tensor_parallel_size=1, max_model_len=MAX_TOK + 512,
              gpu_memory_utilization=0.90, max_num_seqs=int(os.environ.get("E6_MAXSEQS", "64")),
              enable_prefix_caching=False, disable_log_stats=False,
              async_scheduling=True, max_num_batched_tokens=8192)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    ds = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": ds[i]["Question"]
          + "\nPlease reason step by step."}],
        tokenize=False, add_generation_prompt=True,
        enable_thinking=os.environ.get("E6_THINK", "0") == "1")
        for i in range(N_PROMPTS)]
    sp = SamplingParams(n=GROUP, max_tokens=MAX_TOK, temperature=1.0,
                        seed=0)

    def counters():
        acc = drafts = 0
        for m in llm.get_metrics():
            if m.name == "vllm:spec_decode_num_accepted_tokens":
                acc = m.value
            elif m.name == "vllm:spec_decode_num_drafts":
                drafts = m.value
        return acc, drafts

    def rollout(tag):
        a0, d0 = counters()
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        ntok = sum(len(o.token_ids) for out in outs
                   for o in out.outputs)
        a1, d1 = counters()
        acc = 1 + (a1 - a0) / max(d1 - d0, 1)
        print(f"[E6] {tag}: rollout_s={dt:.1f} tokens={ntok} "
              f"tok/s={ntok/dt:.1f} accept={acc:.2f}", flush=True)
        return {"seconds": round(dt, 1), "tokens": ntok,
                "toks": round(ntok / dt, 1), "accept": round(acc, 2)}

    # small warmup (not timed)
    llm.generate(prompts[:2],
                 SamplingParams(n=2, max_tokens=128, temperature=1.0,
                                seed=0), use_tqdm=False)
    if ARM != "off":
        llm.collective_rpc(rpc_dump, args=(SNAP,))

    report = {"arm": ARM, "rollouts": []}
    if ARM in ("off", "policy"):
        report["rollouts"].append(rollout(ARM))
    else:
        # rl_refresh / rl_stale: drift injected before each rollout
        stop = threading.Event()

        def detector():
            prev_a, prev_d = counters()
            cooldown = 0.0
            while not stop.is_set():
                time.sleep(10.0)
                a, dd = counters()
                da, dn = a - prev_a, dd - prev_d
                prev_a, prev_d = a, dd
                now = time.monotonic()
                if dn < 64 or now < cooldown:
                    continue
                w = 1 + da / dn
                if w < GATE:
                    r = llm.collective_rpc(
                        rpc_swap_noisy, args=(SNAP, 0.0, 0))[0]
                    cooldown = now + 6.0
                    print(f"[E6] detector: accept {w:.2f} < {GATE} -> "
                          f"refresh ({r['ms']:.0f} ms)", flush=True)

        if ARM == "rl_refresh":
            threading.Thread(target=detector, daemon=True).start()
        for si, eps in enumerate(DRIFTS):
            llm.collective_rpc(rpc_swap_noisy, args=(SNAP, eps, 50 + si))
            report["rollouts"].append(rollout(f"{ARM} step{si} eps={eps}"))
        stop.set()

    out = Path(f"research/91_bandit_stage3/data/e6_{ARM}.json")
    out.write_text(json.dumps(report, indent=1))
    print("[E6] saved ->", out, flush=True)


if __name__ == "__main__":
    main()
