#!/usr/bin/env python3
"""Phase 89 E3: RL-rollout staleness demo — drift-triggered draft refresh.

Emulation: the target policy "drifts" during training; a draft that is
never re-quantized goes stale. With a fixed target, staleness is
emulated by perturbing the DRAFT (multiplicative noise on scale/norm
tensors) — weight distance grows with phase, accept decays, exactly
the RL staleness signature. The DRAM-cached "fresh" snapshot is the
unperturbed draft (in real RL: the re-quantized current policy).

Modes (E3_MODE):
  cal   noise-level -> accept calibration sweep
  run   the demo: phases x arms
        arms: off (AR) | stale (drift, never refresh) |
              refresh (drift + accept-EMA detector fires the 113ms
              DRAM swap back to fresh; amortization-gated)

Trace per phase: b16 x MATH prompts, T=1.0, 1024 new tok (the R8
rollout shape). Drift schedule: phase noise eps in E3_EPS.
"""
import json
import os
import time
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
SNAP = str(PHASE / "data" / "e3_fresh.pt")
MODE = os.environ.get("E3_MODE", "run")
ARM = os.environ.get("E3_ARM", "refresh")
EPS = [float(x) for x in os.environ.get(
    "E3_EPS", "0,0.04,0.08,0.14,0.22").split(",")]
ACCEPT_GATE = float(os.environ.get("E3_GATE", "3.6"))
SWAP_COST_S = 0.113


def _drafter_model(worker):
    return worker.model_runner.drafter.model


def rpc_dump(worker, path):
    import torch
    m = _drafter_model(worker)
    torch.save({k: v.detach().cpu() for k, v in m.state_dict().items()},
               path)
    return len(dict(m.state_dict()))


def rpc_swap_noisy(worker, path, eps, seed):
    """Load the fresh snapshot, apply multiplicative noise to scale/norm
    tensors (staleness emulation), copy in place. eps=0 == fresh."""
    import torch
    m = _drafter_model(worker)
    sd = torch.load(path, map_location="cpu")
    g = torch.Generator().manual_seed(seed)
    live = dict(m.state_dict())
    t0 = time.perf_counter()
    n = 0
    for k, v in live.items():
        src = sd.get(k)
        if src is None or src.shape != v.shape or src.dtype != v.dtype:
            continue
        if eps > 0 and src.is_floating_point() and (
                "scale" in k or "norm" in k):
            noise = 1.0 + eps * torch.randn(
                src.shape, generator=g, dtype=torch.float32)
            src = (src.to(torch.float32) * noise).to(src.dtype)
            n += 1
        elif (eps > 0 and src.dtype == torch.int32
              and "weight" in k):
            # re-quantization drift on the packed int4 weights: flip a
            # random nibble in an eps/10 fraction of int32 words.
            p = eps / 10.0
            flip = torch.rand(src.shape, generator=g) < p
            nib = torch.randint(1, 16, src.shape, generator=g,
                                dtype=torch.int32)
            slot = torch.randint(0, 8, src.shape, generator=g,
                                 dtype=torch.int32)
            src = src.bitwise_xor(
                torch.where(flip, nib << (slot * 4),
                            torch.zeros_like(nib)))
            n += 1
        v.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    return {"noised": n, "ms": (time.perf_counter() - t0) * 1e3}


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams

    spec = None
    if os.environ.get("E3_SPEC", "1") == "1":
        spec = {"method": "draft_model",
                "model": os.path.expanduser("~/ckpts/Qwen3-8B-W4A8-gptq"),
                "num_speculative_tokens": int(os.environ.get("E3_K", "4")),
                "draft_tensor_parallel_size": 1}
    extra = {}
    if os.environ.get("R88_NO_AUTOTUNE"):
        extra["kernel_config"] = {"enable_flashinfer_autotune": False}
    llm = LLM(model="Qwen/Qwen3-8B", speculative_config=spec,
              tensor_parallel_size=1, max_model_len=8192, **extra,
              gpu_memory_utilization=0.90, max_num_seqs=32,
              enable_prefix_caching=False, disable_log_stats=False,
              async_scheduling=True, max_num_batched_tokens=8192)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    ds = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": ds[i]["Question"]
          + "\nPlease reason step by step."}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for i in range(16)]
    sp = SamplingParams(max_tokens=1024, temperature=1.0, seed=0,
                        ignore_eos=True)

    def counters():
        acc = drafts = 0
        for m in llm.get_metrics():
            if m.name == "vllm:spec_decode_num_accepted_tokens":
                acc = m.value
            elif m.name == "vllm:spec_decode_num_drafts":
                drafts = m.value
        return acc, drafts

    def gen(tag, n_prompts=16):
        a0, d0 = counters()
        t0 = time.perf_counter()
        outs = llm.generate(prompts[:n_prompts], sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        a1, d1 = counters()
        acc = 1 + (a1 - a0) / max(d1 - d0, 1)
        print(f"[E3] {tag}: {ntok/dt:.1f} tok/s accept={acc:.2f}",
              flush=True)
        return ntok / dt, acc, dt

    llm.generate(prompts[:4], sp, use_tqdm=False)  # warmup
    if os.environ.get("E3_SPEC", "1") == "1":
        llm.collective_rpc(rpc_dump, args=(SNAP,))

    if MODE == "cal":
        report = {}
        for eps in EPS:
            llm.collective_rpc(rpc_swap_noisy, args=(SNAP, eps, 7))
            toks, acc, _ = gen(f"cal eps={eps}", 8)
            report[str(eps)] = {"toks": round(toks, 1),
                                "accept": round(acc, 2)}
        (PHASE / "data" / "e3_cal.json").write_text(
            json.dumps(report, indent=1))
        return

    # ---- run mode: one arm over the drift schedule ----
    spec_on = os.environ.get("E3_SPEC", "1") == "1"
    report = {"arm": ARM, "phases": []}
    for pi, eps in enumerate(EPS):
        if spec_on:
            # the world drifts: stale draft distance grows
            r = llm.collective_rpc(
                rpc_swap_noisy, args=(SNAP, eps, 100 + pi))[0]
        toks, acc, dt = gen(f"{ARM} phase{pi} eps={eps}")
        entry = {"phase": pi, "eps": eps, "toks": round(toks, 1),
                 "accept": round(acc, 2), "refreshed": False}
        if spec_on and ARM.startswith("refresh") and acc < ACCEPT_GATE:
            # amortization gate: remaining phases * phase_time * dS
            # >> swap cost (log the math; fire the refresh)
            print(f"[E3] detector: accept {acc:.2f} < {ACCEPT_GATE} "
                  f"-> refresh (gate: phase {dt:.0f}s >> "
                  f"{SWAP_COST_S/0.05:.1f}s amortization)", flush=True)
            r = llm.collective_rpc(rpc_swap_noisy, args=(SNAP, 0.0, 0))[0]
            toks2, acc2, _ = gen(f"{ARM} phase{pi} POST-REFRESH")
            entry.update({"refreshed": True, "swap_ms": r["ms"],
                          "post_toks": round(toks2, 1),
                          "post_accept": round(acc2, 2)})
        report["phases"].append(entry)
    out = PHASE / "data" / f"e3_{ARM}.json"
    out.write_text(json.dumps(report, indent=1))
    print("[E3] saved ->", out, flush=True)


if __name__ == "__main__":
    main()
