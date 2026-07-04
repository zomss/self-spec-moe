"""Phase 55 repro: mixed-step drafter crash (AssertionError: 1 != N).

1-node DP8 V2-Lite, FP8 draft, node-local dispatch. Forces non-uniform
per-DP-rank token counts (rank 0 gets one extra request, plus a small
prefill budget so decode/prefill mix within steps) so the draft step-0
dispatch sees non-uniform DP chunk sizes -- reproducing the 236B b>=32
mixed prefill/decode crash without the 2-node 236B setup.

Root cause this reproduces: the FP8 draft's per-TENSOR activation scale
(shape (1,)) rode the token-sized all-gatherv in naive_dp_ep.prepare;
uniform per-rank sizes collapse to a plain all_gather and mask it, the
first non-uniform step asserts `1 != num_tokens` on every rank. Run with
W7_A2A_DEBUG=1 to log every non-uniform dispatch (branch, sizes, shapes,
forward-context keys).

Usage: source research/52_two_node_e2e/scripts/env_2node.sh, then run;
spawns 8 ranks like w7_2node.py. Exits 0 iff all ranks complete.
"""
import os
import sys
import time
from multiprocessing import Process, Queue

MODEL = os.environ.get("W7_MODEL", "deepseek-ai/DeepSeek-V2-Lite")
LOCAL_WORLD = int(os.environ.get("W7_LOCAL_WORLD", "8"))
DP = LOCAL_WORLD
TP = 1
MASTER_PORT = int(os.environ.get("W7_MASTER_PORT", "13460"))
BATCH = int(os.environ.get("W7_BATCH", "8"))
OUTLEN = int(os.environ.get("W7_REPRO_OUTLEN", "64"))
K = int(os.environ.get("W7_K", "1"))
SKEW = int(os.environ.get("W7_SKEW", "1"))  # extra requests on rank 0
# Small prefill budget + long prompts -> chunked, staggered prefill so
# decode and prefill MIX within single steps (the 236B b>=32 regime).
MAX_BATCHED = int(os.environ.get("W7_REPRO_MAX_BATCHED", "256"))

BASE_PROMPT = (
    "The history of artificial intelligence began in antiquity with myths and "
    "stories, and the modern field was founded in"
) * 5


def worker(rank, q):
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(DP)
    os.environ["VLLM_DP_MASTER_IP"] = "127.0.0.1"
    os.environ["VLLM_DP_MASTER_PORT"] = str(MASTER_PORT)
    os.environ["VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE"] = os.environ.get(
        "W7_DRAFT_LOCAL_ROUTE", "0"
    )
    os.environ["VLLM_SELF_SPEC_DRAFT_NODE_LOCAL"] = os.environ.get(
        "W7_DRAFT_NODE_LOCAL", "1"
    )
    os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = "0"
    os.environ["VLLM_SELF_SPEC_DRAFT_FULL_REPLICA"] = "0"

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=TP,
        enable_expert_parallel=True,
        trust_remote_code=True,
        max_model_len=2048,
        max_num_batched_tokens=MAX_BATCHED,
        max_num_seqs=16,
        gpu_memory_utilization=0.90,
        disable_log_stats=False,
        speculative_config={
            "method": "draft_model",
            "model": MODEL,
            "num_speculative_tokens": K,
            "draft_tensor_parallel_size": TP,
            "quantization": "fp8",
        },
    )

    # Rank 0 gets SKEW extra requests -> per-rank decode token counts differ
    # (18 vs 16 at BATCH=8, SKEW=1) -> non-uniform DP chunk sizes every step.
    batch = BATCH + (SKEW if rank == 0 else 0)
    prompts = [f"{BASE_PROMPT} the year {1900 + i}." for i in range(batch)]
    sp = SamplingParams(temperature=0.0, max_tokens=OUTLEN, ignore_eos=True, seed=0)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    q.put((rank, dt, ntok))


if __name__ == "__main__":
    q = Queue()
    procs = [Process(target=worker, args=(r, q)) for r in range(DP)]
    for p in procs:
        p.start()
    ok = True
    for p in procs:
        p.join()
        if p.exitcode != 0:
            ok = False
    results = []
    while not q.empty():
        results.append(q.get())
    print(f"[repro] ok={ok} results={sorted(results)}")
    sys.exit(0 if ok and len(results) == DP else 1)
