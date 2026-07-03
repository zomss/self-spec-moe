"""Torch-profiler trace of the spec verify forward (Phase 52 follow-up).

Single-node DP8 external-launcher spawn (same pattern as w7_2node.py). Rank 0
gets profiler_config(torch) and traces a short generate window (~4 cycles);
other ranks run lockstep untraced. Modes:
  spec   : Phase-47 stack, K=2, FP8 full-replica draft, W7_BATCH (default 64)
  nospec : plain decode at W7_BATCH (use 192 = the verify-equivalent load)

Goal: read the verify forward's execution mode from the trace —
FULL cudagraph (1 cudaGraphLaunch per forward) vs PIECEWISE (~n_layers graph
launches + eager attention kernels) vs eager (thousands of kernel launches).
"""
import os
import sys
from multiprocessing import Process

MODEL = os.environ.get("W7_MODEL", "Qwen/Qwen3-30B-A3B")
LOCAL_WORLD = int(os.environ.get("W7_LOCAL_WORLD", "8"))
MASTER_PORT = int(os.environ.get("W7_MASTER_PORT", "13390"))
BATCH = int(os.environ.get("W7_BATCH", "64"))
K = int(os.environ.get("W7_K", "2"))
TRACE_DIR = os.environ.get(
    "W7_TRACE_DIR",
    "/h/v-sukmincho/self-spec-moe/research/52_two_node_e2e/data/trace",
)
GPU_MEM = float(os.environ.get("W7_GPU_MEM", "0.90"))

BASE_PROMPT = (
    "The history of artificial intelligence began in antiquity with myths and "
    "stories, and the modern field was founded in"
)


def worker(rank, mode):
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(LOCAL_WORLD)
    os.environ["VLLM_DP_MASTER_IP"] = "127.0.0.1"
    os.environ["VLLM_DP_MASTER_PORT"] = str(MASTER_PORT)

    spec = mode == "spec"
    if spec:
        os.environ["VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE"] = os.environ.get(
            "W7_DRAFT_LOCAL_ROUTE", "1"
        )
        os.environ["VLLM_SELF_SPEC_DRAFT_NODE_LOCAL"] = os.environ.get(
            "W7_DRAFT_NODE_LOCAL", "0"
        )
        os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = "0"
        os.environ["VLLM_SELF_SPEC_DRAFT_FULL_REPLICA"] = os.environ.get(
            "W7_DRAFT_FULL_REPLICA", "1"
        )

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=MODEL,
        tensor_parallel_size=1,
        enable_expert_parallel=True,
        trust_remote_code=os.environ.get("W7_TRC", "0") == "1",
        max_model_len=2048,
        gpu_memory_utilization=GPU_MEM,
        enforce_eager=False,
        disable_log_stats=False,
    )
    if spec:
        kwargs["speculative_config"] = {
            "method": "draft_model",
            "model": MODEL,
            "num_speculative_tokens": K,
            "draft_tensor_parallel_size": 1,
            "quantization": "fp8",
        }
    if rank == 0:
        kwargs["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": TRACE_DIR,
            "torch_profiler_with_stack": bool(
                int(os.environ.get("W7_TRACE_STACK", "0"))
            ),
        }
    llm = LLM(**kwargs)

    prompts = [f"{BASE_PROMPT} the year {1900 + i}." for i in range(BATCH)]

    def gen(n):
        sp = SamplingParams(
            temperature=0.0, max_tokens=n, ignore_eos=True, seed=0
        )
        llm.generate(prompts, sp, use_tqdm=False)

    gen(24)
    gen(24)
    if rank == 0:
        llm.start_profile()
    gen(9)
    if rank == 0:
        llm.stop_profile()
        print(f"[TRACE] {mode} batch={BATCH} trace written to {TRACE_DIR}",
              flush=True)


def main():
    mode = sys.argv[1]
    assert mode in ("nospec", "spec")
    os.makedirs(TRACE_DIR, exist_ok=True)
    procs = [Process(target=worker, args=(r, mode)) for r in range(LOCAL_WORLD)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
