#!/usr/bin/env python3
"""W2: regime runner that records BOTH currencies per run (I1/G7).

Same boot path and protocol as 95/run_e0_envelope.py, but each timed round
additionally snapshots the engine's per-request phase timers, so every run
carries:

  end-to-end currency   gen_toks / wall            (what phase 95 scored)
  decode-only currency  from request_decode_time   (what C2's map predicts)
  the bridge            phi = prefill / (prefill + decode) request-time share

The decomposition uses histogram/counter DELTAS across the timed window:
  vllm:request_prefill_time_seconds   (sum over finished requests)
  vllm:request_decode_time_seconds
  vllm:time_to_first_token_seconds
  vllm:e2e_request_latency_seconds
  vllm:prompt_tokens / vllm:generation_tokens
  spec accepted / drafts

Raw sums are recorded; the scorer derives rates. Decode tokens are
generation_tokens - finished_requests (the first output token belongs to
prefill by the TTFT convention).

env: W2_ARCH dense|llama   W2_WINDOW off|512|2048   W2_TUNE 1|0
     W2_REGIMES (default R4,R5,R5cot,R8,R1,R6)      W2_SEED (default 0)
     W2_ITERS (default 3)                            W2_OUT
"""
import json
import os
import sys
import time
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
P95 = PHASE.parent / "95_c3_deploy"
sys.path.insert(0, str(PHASE.parent / "88_regime_eval/scripts"))
from regime_datasets import load_regime  # noqa: E402

ARCH = os.environ.get("W2_ARCH", "dense")
WINDOW = os.environ.get("W2_WINDOW", "off")
TUNE = os.environ.get("W2_TUNE", "1") == "1"
ITERS = int(os.environ.get("W2_ITERS", "3"))
SEED = int(os.environ.get("W2_SEED", "0"))
REGIMES = os.environ.get("W2_REGIMES", "R4,R5,R5cot,R8,R1,R6").split(",")

ARCHS = {   # identical to 95/run_e0_envelope.py
    "dense": {
        "model": "Qwen/Qwen3-8B",
        "draft": os.path.expanduser("/data/smcho/ckpts/Qwen3-8B-W4A8-gptq"),
        "config": "q-hum_s-b2",
        "skip": "2,8",
    },
    "llama": {
        "model": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "draft": os.path.expanduser(
            "/data/smcho/ckpts/Llama31-8B-Instruct-W4A16-INT4-sym"),
        "config": "q-w4a16_s-b2",
        "skip": "3,8",
    },
}
KMAX = 4

HISTS = ["vllm:request_prefill_time_seconds",
         "vllm:request_decode_time_seconds",
         "vllm:time_to_first_token_seconds",
         "vllm:e2e_request_latency_seconds"]
CTRS = ["vllm:prompt_tokens", "vllm:generation_tokens",
        "vllm:spec_decode_num_accepted_tokens", "vllm:spec_decode_num_drafts"]


def snap(llm):
    s = {}
    for m in llm.get_metrics():
        if m.name in HISTS:
            s[m.name] = (m.sum, m.count)
        elif m.name in CTRS:
            s[m.name] = (m.value, 0)
    return s


def delta(s0, s1):
    return {k: (round(s1[k][0] - s0.get(k, (0, 0))[0], 4),
                s1[k][1] - s0.get(k, (0, 0))[1])
            for k in s1}


def main():
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams

    a = ARCHS[ARCH]
    spec = None
    if WINDOW != "off":
        tbl = P95 / "data" / f"policy_{ARCH}_{a['config']}_w{WINDOW}.json"
        if not tbl.exists():
            raise SystemExit(f"missing policy table {tbl}")
        os.environ["VLLM_SELF_SPEC_POLICY_FILE"] = str(tbl)
        os.environ["VLLM_SELF_SPEC_DRAFT_KV_WINDOW"] = WINDOW
        os.environ["VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS"] = a["skip"]
        spec = {"method": "draft_model", "model": a["draft"],
                "num_speculative_tokens": KMAX,
                "draft_tensor_parallel_size": 1}

    extra = {}
    if not TUNE:
        extra["kernel_config"] = {"enable_flashinfer_autotune": False}
    llm = LLM(model=a["model"], speculative_config=spec,
              tensor_parallel_size=1, max_model_len=20480,
              gpu_memory_utilization=0.90, max_num_seqs=32,
              enable_prefix_caching=False, disable_log_stats=False,
              async_scheduling=True, max_num_batched_tokens=8192,
              **extra)
    tok = AutoTokenizer.from_pretrained(a["model"])

    out = {"arch": ARCH, "window": WINDOW, "autotune": TUNE, "seed": SEED,
           "config": a["config"], "skip_set": a["skip"] if spec else None,
           "kmax": KMAX if spec else 0, "model": a["model"],
           "draft": a["draft"] if spec else None, "regimes": {}}
    for rid in REGIMES:
        n_load = 32 if rid == "R6" else 16
        prompts, gs = load_regime(rid, tok, n=n_load, seed=SEED)
        b = gs["batch"]
        sp = SamplingParams(max_tokens=gs["max_tokens"],
                            temperature=gs["temperature"],
                            seed=SEED if gs["temperature"] else None,
                            ignore_eos=(rid in ("R5cot", "R8")))
        llm.generate(prompts[:b], sp, use_tqdm=False)      # warmup
        rounds = []
        for _ in range(ITERS):
            s0 = snap(llm)
            t0 = time.perf_counter()
            ntok = 0
            if b == 1:
                for p in prompts[:4]:
                    o = llm.generate([p], sp, use_tqdm=False)
                    ntok += sum(len(x.outputs[0].token_ids) for x in o)
            else:
                o = llm.generate(prompts[:b], sp, use_tqdm=False)
                ntok = sum(len(x.outputs[0].token_ids) for x in o)
            dt = time.perf_counter() - t0
            d = delta(s0, snap(llm))
            pf_s, n_req = d["vllm:request_prefill_time_seconds"]
            dc_s, _ = d["vllm:request_decode_time_seconds"]
            gen = d["vllm:generation_tokens"][0]
            acc = d.get("vllm:spec_decode_num_accepted_tokens", (0, 0))[0]
            drf = d.get("vllm:spec_decode_num_drafts", (0, 0))[0]
            rounds.append({
                "wall_s": round(dt, 3), "ntok": ntok,
                "e2e_rate": round(ntok / dt, 1),
                "n_req": n_req,
                "prompt_toks": d["vllm:prompt_tokens"][0],
                "gen_toks": gen,
                "prefill_time_s": pf_s, "decode_time_s": dc_s,
                "ttft_sum_s": d["vllm:time_to_first_token_seconds"][0],
                "e2e_lat_sum_s": d["vllm:e2e_request_latency_seconds"][0],
                # decode currency: decode tokens over per-request decode time
                "decode_rate_req": round((gen - n_req) / dc_s, 1)
                if dc_s > 0 else None,
                "phi_req": round(pf_s / (pf_s + dc_s), 4)
                if pf_s + dc_s > 0 else None,
                "accept": round(1 + acc / drf, 3) if drf > 0 else None,
            })
        rates = sorted(r["e2e_rate"] for r in rounds)
        med = rates[len(rates) // 2]
        out["regimes"][rid] = {"toks": med, "batch": b,
                               "max_tokens": gs["max_tokens"],
                               "temp": gs["temperature"], "rounds": rounds}
        r0 = rounds[len(rounds) // 2]
        print(f"[W2] {ARCH} w={WINDOW} tune={int(TUNE)} s{SEED} {rid} "
              f"b={b} e2e={med:.1f} dec_req={r0['decode_rate_req']} "
              f"phi={r0['phi_req']} accept={r0['accept']}", flush=True)

    path = os.environ.get("W2_OUT", str(
        PHASE / "data" / "w2" /
        f"w2_{ARCH}_w{WINDOW}_{'tune' if TUNE else 'notune'}_s{SEED}.json"))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=1))
    print("[W2] saved ->", path, flush=True)


if __name__ == "__main__":
    main()
