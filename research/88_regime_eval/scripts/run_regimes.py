#!/usr/bin/env python3
"""Phase 88 E1/E2: AR baseline + best-lever arm on every canonical
regime (real datasets, serving driver, wall clock).

env: R88_ARM   off | w4win | w4a8   (draft ckpt + kernel per arm)
     R88_K     draft depth for spec arms (default 6)
     R88_REGIMES comma filter (default: all)
     R88_ITERS timed rounds per regime (default 3)
     R88_OUT   output json path
Boot once, run each regime at its canonical (batch, gen spec).
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from regime_datasets import REGIMES, load_regime  # noqa: E402

PHASE = Path(__file__).resolve().parents[1]
MODEL = os.environ.get("R88_MODEL", "Qwen/Qwen3-8B")
ARM = os.environ.get("R88_ARM", "off")
K = int(os.environ.get("R88_K", "6"))
ITERS = int(os.environ.get("R88_ITERS", "3"))
DRAFTS = {"w4win": "~/ckpts/Qwen3-8B-W4A16-INT4",
          "w4a8": "~/ckpts/Qwen3-8B-W4A8-gptq"}


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

    spec = None
    if ARM != "off":
        spec = {"method": "draft_model",
                "model": os.path.expanduser(DRAFTS[ARM]),
                "num_speculative_tokens": K,
                "draft_tensor_parallel_size": 1}
    extra = {}
    if os.environ.get("R88_NO_AUTOTUNE"):
        extra["kernel_config"] = {"enable_flashinfer_autotune": False}
    llm = LLM(model=MODEL, speculative_config=spec,
              tensor_parallel_size=1, max_model_len=20480,
              gpu_memory_utilization=0.90, max_num_seqs=32,
              enable_prefix_caching=False, disable_log_stats=False,
              async_scheduling=True, max_num_batched_tokens=8192,
              **extra)
    tok = AutoTokenizer.from_pretrained(MODEL)

    want = os.environ.get("R88_REGIMES")
    rids = want.split(",") if want else REGIMES
    out = {"arm": ARM, "K": K if spec else 0, "model": MODEL}
    for rid in rids:
        n_load = 32 if rid == "R6" else 16
        prompts, gs = load_regime(rid, tok, n=n_load)
        b = gs["batch"]
        sp = SamplingParams(max_tokens=gs["max_tokens"],
                            temperature=gs["temperature"],
                            seed=0 if gs["temperature"] else None,
                            ignore_eos=(rid in ("R5cot", "R8")))
        # warmup on the first batch shape
        llm.generate(prompts[:b], sp, use_tqdm=False)
        rates, accepts = [], []
        for it in range(ITERS):
            a0, d0 = spec_counters(llm)
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
            a1, d1 = spec_counters(llm)
            rates.append(ntok / dt)
            if d1 > d0:
                accepts.append(1 + (a1 - a0) / (d1 - d0))
        rates.sort()
        med = rates[len(rates) // 2]
        acc = round(sum(accepts) / len(accepts), 2) if accepts else None
        out[rid] = {"toks": round(med, 1), "all": [round(r, 1) for r in rates],
                    "accept": acc, "batch": b,
                    "max_tokens": gs["max_tokens"], "temp": gs["temperature"]}
        print(f"[R88] {rid} arm={ARM} b={b} toks={med:.1f} "
              f"accept={acc}", flush=True)
    path = os.environ.get(
        "R88_OUT", str(PHASE / "data" / f"regimes_{ARM}.json"))
    Path(path).write_text(json.dumps(out, indent=1))
    print("[R88] saved ->", path, flush=True)


if __name__ == "__main__":
    main()
