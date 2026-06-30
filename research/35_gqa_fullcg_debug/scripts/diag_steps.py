"""Diag: dump per-step draft tokens for the first propose() call, batch=1.

Run with MODE=piecewise and MODE=fullcg; the first propose() call sees the
identical prefix in both, so comparing the [W7DBG call=0 ...] lines isolates
whether the captured FULL-CG attention produces different draft tokens than
eager. This was the decisive root-cause diagnostic (PIECEWISE step0 -> token
576, FULL-CG step0 -> 21718 for IDENTICAL inputs).

NOTE: requires the temporary [W7DBG] logging block (gated by W7_GQA_DEBUG_STEPS)
in propose(); that instrumentation was removed from llm_base_proposer.py once the
fix landed. Kept here as a record of the diagnostic; re-add the logging to rerun.
"""

import os

MODEL = os.environ.get("MB_MODEL", "Qwen/Qwen3-8B")
K = int(os.environ.get("MB_K", "4"))


def main():
    mode = os.environ.get("MODE", "piecewise")
    os.environ["VLLM_SELF_SPEC_DRAFT_FULL_CG"] = "1" if mode == "fullcg" else "0"
    os.environ["W7_GQA_DEBUG_STEPS"] = os.environ.get("W7_GQA_DEBUG_STEPS", "2")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL, tensor_parallel_size=1, trust_remote_code=True,
        max_model_len=2048, gpu_memory_utilization=0.85, enforce_eager=False,
        disable_log_stats=True,
        speculative_config={
            "method": "draft_model", "model": MODEL,
            "num_speculative_tokens": K, "draft_tensor_parallel_size": 1,
        },
    )
    prompt = ("The capital of France is Paris. The capital of Germany is Berlin. "
              "The capital of Italy is")
    # batch=1 so call=0 prefix is deterministic and identical across modes.
    sp = SamplingParams(temperature=0.0, max_tokens=16, seed=0)
    out = llm.generate([prompt], sp, use_tqdm=False)
    print(f"GENTOK mode={mode} {list(out[0].outputs[0].token_ids)}", flush=True)


if __name__ == "__main__":
    main()
