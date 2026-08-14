#!/usr/bin/env python3
"""W2b: regime-ORDER effect, one boot. R5 -> R4 -> R5-again, llama w2048.

W1 (fresh boot, R5 only) measured 194-196 tok/s; W2 (R4 then R5, same
boot config, notune) measured 148-157 with IDENTICAL accept (3.12) -- a
-22% steady state created by having run R4 first. Two mechanisms fit:

  H-duty     the compiled policy parks part of R5's steps after R4 lowers
             its EMA/state (parked steps draft nothing -> accept untouched,
             rate slides toward AR's 142)
  H-scatter  R4's alloc/free permutes the KV block pool; R5's pages land
             scattered and the 2176-column scratchpad gather pays a
             bandwidth penalty (numerics identical, cost only)

Discriminators, pre-registered:
  P-W2b1  R5-first reproduces W1 (~195): the W1/W2 gap is order, not
          harness drift.
  P-W2b2  R5-again drops to ~150 AND [kpick] shows armed duty comparable
          to R5-first -> H-scatter (allocator layout).
  P-W2b3  R5-again drops AND duty drops (K=0 fraction rises) -> H-duty
          (policy state pollution across regimes).
  P-W2b4  R5-again stays ~195 -> no order effect in THIS boot; the W2 gap
          came from something else (re-open).

GATE_DEBUG=1; parse with 95/scripts/parse_kpick.py on the boot log.
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

MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
DRAFT = os.path.expanduser("/data/smcho/ckpts/Llama31-8B-Instruct-W4A16-INT4-sym")
ITERS = int(os.environ.get("W2B_ITERS", "3"))


def spec_counters(llm):
    acc = drafts = 0
    for m in llm.get_metrics():
        if m.name == "vllm:spec_decode_num_accepted_tokens":
            acc = m.value
        elif m.name == "vllm:spec_decode_num_drafts":
            drafts = m.value
    return acc, drafts


def main():
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams

    tbl = P95 / "data" / "policy_llama_q-w4a16_s-b2_w2048.json"
    os.environ["VLLM_SELF_SPEC_POLICY_FILE"] = str(tbl)
    os.environ["VLLM_SELF_SPEC_DRAFT_KV_WINDOW"] = "2048"
    os.environ["VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS"] = "3,8"
    os.environ["VLLM_SELF_SPEC_GATE_DEBUG"] = "1"

    llm = LLM(model=MODEL,
              speculative_config={"method": "draft_model", "model": DRAFT,
                                  "num_speculative_tokens": 4,
                                  "draft_tensor_parallel_size": 1},
              tensor_parallel_size=1, max_model_len=20480,
              gpu_memory_utilization=0.90, max_num_seqs=32,
              enable_prefix_caching=False, disable_log_stats=False,
              async_scheduling=True, max_num_batched_tokens=8192,
              kernel_config={"enable_flashinfer_autotune": False})
    tok = AutoTokenizer.from_pretrained(MODEL)

    out = {"order": [], "phases": {}}
    for tag, rid in (("R5_first", "R5"), ("R4_mid", "R4"),
                     ("R5_again", "R5")):
        prompts, gs = load_regime(rid, tok, n=16, seed=0)
        b = gs["batch"]
        sp = SamplingParams(max_tokens=gs["max_tokens"], temperature=0.0)
        print(f"[W2b] === phase {tag} ===", flush=True)
        llm.generate(prompts[:b], sp, use_tqdm=False)      # warmup
        rates, accepts = [], []
        for _ in range(ITERS):
            a0, d0 = spec_counters(llm)
            t0 = time.perf_counter()
            o = llm.generate(prompts[:b], sp, use_tqdm=False)
            dt = time.perf_counter() - t0
            a1, d1 = spec_counters(llm)
            ntok = sum(len(x.outputs[0].token_ids) for x in o)
            rates.append(round(ntok / dt, 1))
            if d1 > d0:
                accepts.append(round(1 + (a1 - a0) / (d1 - d0), 3))
        out["order"].append(tag)
        out["phases"][tag] = {"rates": rates, "accepts": accepts}
        print(f"[W2b] {tag} rates={rates} accepts={accepts}", flush=True)

    path = os.environ.get("W2B_OUT",
                          str(PHASE / "data" / "w2" / "w2b_order.json"))
    Path(path).write_text(json.dumps(out, indent=1))
    print("[W2b] saved ->", path, flush=True)


if __name__ == "__main__":
    main()
