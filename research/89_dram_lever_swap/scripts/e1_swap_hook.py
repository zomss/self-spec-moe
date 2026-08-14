#!/usr/bin/env python3
"""Phase 89 E1: in-engine draft weight swap via collective_rpc.

Boot the fixed 8B stack (W4+win K4, wholechain FULLCG), then:
  1. baseline generation (accept A0)
  2. dump the draft's live (post-processed) params to a snapshot
  3. identity swap-in (measure in-engine latency; accept must == A0)
  4. corrupted swap-in (scales zeroed): accept must COLLAPSE -- proves
     the captured graphs execute the swapped weights end-to-end
  5. swap back: accept must recover to A0
"""
import json
import os
import time
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
SNAP = str(PHASE / "data" / "e1_draft_snapshot.pt")
SNAP_BAD = str(PHASE / "data" / "e1_draft_snapshot_bad.pt")


def _drafter_model(worker):
    return worker.model_runner.drafter.model


def rpc_dump(worker, path):
    import torch
    m = _drafter_model(worker)
    sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    torch.save(sd, path)
    return len(sd)


def rpc_swap(worker, path):
    import torch
    m = _drafter_model(worker)
    sd = torch.load(path, map_location="cpu")
    live = dict(m.state_dict())
    t0 = time.perf_counter()
    n = 0
    for k, v in live.items():
        src = sd.get(k)
        if src is None or src.shape != v.shape or src.dtype != v.dtype:
            continue
        v.copy_(src, non_blocking=True)
        n += 1
    torch.cuda.synchronize()
    return {"n_tensors": n, "swap_ms": (time.perf_counter() - t0) * 1e3}


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams

    llm = LLM(model="Qwen/Qwen3-8B",
              speculative_config={
                  "method": "draft_model",
                  "model": os.path.expanduser("/data/smcho/ckpts/Qwen3-8B-W4A16-INT4"),
                  "num_speculative_tokens": 4,
                  "draft_tensor_parallel_size": 1},
              tensor_parallel_size=1, max_model_len=8192,
              gpu_memory_utilization=0.90, max_num_seqs=32,
              enable_prefix_caching=False, disable_log_stats=False,
              async_scheduling=True, max_num_batched_tokens=8192)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": gsm[i]["question"]
          + "\nPlease reason step by step."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for i in range(8)]
    sp = SamplingParams(max_tokens=256, temperature=0.0)

    def accept():
        acc = drafts = 0
        for m in llm.get_metrics():
            if m.name == "vllm:spec_decode_num_accepted_tokens":
                acc = m.value
            elif m.name == "vllm:spec_decode_num_drafts":
                drafts = m.value
        return acc, drafts

    def gen_phase(tag):
        a0, d0 = accept()
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        a1, d1 = accept()
        acc = 1 + (a1 - a0) / max(d1 - d0, 1)
        print(f"[E1] {tag}: {ntok/dt:.1f} tok/s accept={acc:.2f}",
              flush=True)
        return {"toks": round(ntok / dt, 1), "accept": round(acc, 3)}

    report = {}
    llm.generate(prompts[:2], sp, use_tqdm=False)  # warmup
    report["baseline"] = gen_phase("baseline")

    n = llm.collective_rpc(rpc_dump, args=(SNAP,))[0]
    print(f"[E1] dumped {n} tensors -> {SNAP}", flush=True)

    r = llm.collective_rpc(rpc_swap, args=(SNAP,))[0]
    print(f"[E1] identity swap: {r['n_tensors']} tensors in "
          f"{r['swap_ms']:.0f} ms", flush=True)
    report["identity_swap"] = r
    report["after_identity"] = gen_phase("after-identity-swap")

    # corrupted snapshot: zero every *scale* tensor
    import torch
    sd = torch.load(SNAP, map_location="cpu")
    nbad = 0
    for k in sd:
        if "scale" in k:
            sd[k] = torch.zeros_like(sd[k])
            nbad += 1
    torch.save(sd, SNAP_BAD)
    print(f"[E1] corrupted snapshot: zeroed {nbad} scale tensors",
          flush=True)
    r = llm.collective_rpc(rpc_swap, args=(SNAP_BAD,))[0]
    report["corrupt_swap"] = r
    report["after_corrupt"] = gen_phase("after-corrupt-swap")

    r = llm.collective_rpc(rpc_swap, args=(SNAP,))[0]
    report["restore_swap"] = r
    report["after_restore"] = gen_phase("after-restore")

    ok = (abs(report["after_identity"]["accept"]
              - report["baseline"]["accept"]) < 0.15
          and report["after_corrupt"]["accept"]
          < report["baseline"]["accept"] - 0.8
          and abs(report["after_restore"]["accept"]
                  - report["baseline"]["accept"]) < 0.15)
    report["VERDICT"] = "PASS" if ok else "FAIL"
    print(f"[E1] VERDICT: {report['VERDICT']}", flush=True)
    out = PHASE / "data" / "e1_swap_hook.json"
    out.write_text(json.dumps(report, indent=1))
    print("[E1] saved ->", out, flush=True)


if __name__ == "__main__":
    main()
