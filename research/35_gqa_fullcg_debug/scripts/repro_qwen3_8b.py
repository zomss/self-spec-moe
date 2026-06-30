"""W7-gqa Step1: reproduce FULL-CG accept_len collapse on Qwen3-8B (single GPU).

Runs draft_model self-spec (draft = Qwen3-8B), greedy, K=4, CUDA graphs ON, in
ONE process so PIECEWISE (FULL_CG off) and FULL-CG (FULL_CG on) are compared on
the same machine/code. Each mode is a fresh LLM in a subprocess so envs apply at
init.

Usage:
  MODE=piecewise|fullcg python repro_qwen3_8b.py
Prints: accept_len for the mode.
"""

import os
import sys

MODEL = os.environ.get("MB_MODEL", "Qwen/Qwen3-8B")
K = int(os.environ.get("MB_K", "4"))
BATCH = int(os.environ.get("MB_BATCH", "16"))
OUTLEN = int(os.environ.get("MB_OUTLEN", "128"))
MAX_MODEL_LEN = int(os.environ.get("MB_MAX_MODEL_LEN", "2048"))
GPU_MEM = float(os.environ.get("MB_GPU_MEM", "0.85"))


def _metric_value(metrics, name):
    for m in metrics:
        if m.name == name:
            v = getattr(m, "value", None)
            if v is not None:
                return v
            if hasattr(m, "values"):
                return sum(m.values)
    return None


def main():
    mode = os.environ.get("MODE", "piecewise")
    assert mode in ("piecewise", "fullcg")
    if mode == "fullcg":
        os.environ["VLLM_SELF_SPEC_DRAFT_FULL_CG"] = "1"
    else:
        os.environ["VLLM_SELF_SPEC_DRAFT_FULL_CG"] = "0"

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM,
        enforce_eager=False,
        disable_log_stats=False,
        speculative_config={
            "method": "draft_model",
            "model": MODEL,
            "num_speculative_tokens": K,
            "draft_tensor_parallel_size": 1,
        },
    )

    base = (
        "The history of artificial intelligence began in antiquity, with myths, "
        "stories and rumors of artificial beings endowed with intelligence by "
        "master craftsmen. In modern times, the field was founded in"
    )
    prompts = [f"{base} the year {1900 + i}." for i in range(BATCH)]
    sp = SamplingParams(temperature=0.0, max_tokens=OUTLEN, ignore_eos=True, seed=0)

    def snap():
        m = llm.get_metrics()
        return (
            _metric_value(m, "vllm:spec_decode_num_accepted_tokens") or 0.0,
            _metric_value(m, "vllm:spec_decode_num_drafts") or 0.0,
        )

    # warm up graphs
    llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=8, seed=0),
                 use_tqdm=False)
    s0 = snap()
    llm.generate(prompts, sp, use_tqdm=False)
    s1 = snap()
    accepted = s1[0] - s0[0]
    ndrafts = s1[1] - s0[1]
    al = (1 + accepted / ndrafts) if ndrafts else None
    print(f"RESULT mode={mode} accept_len={al} accepted={accepted} "
          f"ndrafts={ndrafts}", flush=True)


if __name__ == "__main__":
    main()
