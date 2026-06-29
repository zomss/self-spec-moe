"""W2c GATE: verify vLLM loads + RUNS the NVFP4 checkpoint on H100 (sm_90).

Loads the exported NVFP4 DeepSeek-V2-Lite checkpoint standalone in vLLM and
generates a few tokens. On sm_90 (no FP4 tensor cores), the NVFP4 MoE oracle
should fall through to the MARLIN backend (FP4 weights dequantized -> bf16 GEMM,
the W4A16 path). If vLLM cannot load/run NVFP4 here, this prints the exact error.

Run:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/smcho/ssm-w7fp4 \
    NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1 \
    VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 \
    /data/smcho/self-spec-moe/.venv/bin/python \
    research/34_worldA_system/scripts/fp4_gate.py
"""
import os
import traceback

CKPT = os.environ.get(
    "FP4_CKPT",
    "/data/smcho/ssm-w7fp4/research/34_worldA_system/ckpts/dsv2lite-nvfp4",
)


def main():
    from vllm import LLM, SamplingParams

    try:
        llm = LLM(
            model=CKPT,
            tensor_parallel_size=1,
            enable_expert_parallel=False,
            trust_remote_code=False,
            max_model_len=2048,
            gpu_memory_utilization=0.85,
            enforce_eager=False,  # exercise CUDA graphs too
        )
    except Exception:
        print("=== GATE FAIL: vLLM could not LOAD the NVFP4 checkpoint ===")
        traceback.print_exc()
        raise

    prompts = [
        "The capital of France is",
        "Q: What is 2 + 2? A:",
        "The history of artificial intelligence began",
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=24, seed=0)
    try:
        outs = llm.generate(prompts, sp)
    except Exception:
        print("=== GATE FAIL: vLLM loaded but could not RUN/generate ===")
        traceback.print_exc()
        raise

    print("=== GATE PASS: vLLM loaded + ran NVFP4 on H100 (sm_90) ===")
    for o in outs:
        print(f"  PROMPT: {o.prompt!r}")
        print(f"  OUTPUT: {o.outputs[0].text!r}")
    print("=== GATE PASS confirmed ===")


if __name__ == "__main__":
    main()
