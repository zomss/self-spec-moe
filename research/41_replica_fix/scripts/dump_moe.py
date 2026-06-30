"""Phase 41 Gate 1: dump per-token MoE output for the GENUINE full-replica path
(config A2) vs plain MoE (config D, ground truth) at DP=8, on the same prompt.

Identical to phase 40 dump_moe.py, but config A here sets
VLLM_SELF_SPEC_DRAFT_FULL_REPLICA=1 so the main-model MoE takes the post-fix
genuine-replica branch in FusedMoEParallelConfig.make (tp_size=ep_size=1, full
unsharded experts, no all-reduce needed) instead of the Bug-B flatten-TP path.

Expectation post-fix: A2 (full replica) MoE output at DP=8 == plain MoE (D) at
DP=1 -> rel-err ~0.0 (was 0.70x / rel 0.539 with Bug B).

Env:
  CFG     A | C | D   (required)
  E_DP    DP size              default 8
  E_LAYER dump layer index     default 0
  E_OUT   dump dir             default research/41_replica_fix/data/dump_<cfg>
"""
import os
from multiprocessing import Process

MODEL = os.environ.get(
    "E_MODEL",
    "/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B"
    "/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9",
)
CFG = os.environ["CFG"].strip().upper()
DP = int(os.environ.get("E_DP", "8"))
LAYER = int(os.environ.get("E_LAYER", "0"))
OUT = os.environ.get(
    "E_OUT",
    f"/data/smcho/ssm-num/research/41_replica_fix/data/dump_{CFG}_dp{DP}",
)

# A2 = GENUINE full replica (post-fix): EP off, LOCAL_ROUTE skip (combine
#      passes), AND DRAFT_FULL_REPLICA=1 to force tp=ep=1 (no flatten-TP).
# C  = EP all-to-all (the verify).
# D  = plain MoE baseline (EP off, no local-route skip); at DP=1 it is the
#      single-GPU ground truth.
CFG_MAP = {
    "A": ("0", "1", "1"),  # (enable_ep, local_route, full_replica)
    "C": ("1", "0", "0"),
    "D": ("0", "0", "0"),
}
assert CFG in CFG_MAP, f"CFG must be A/C/D, got {CFG}"
ENABLE_EP, LOCAL_ROUTE, FULL_REPLICA = CFG_MAP[CFG]


def worker(rank, master_ip, master_port):
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(DP)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

    os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = LOCAL_ROUTE
    os.environ["VLLM_SELF_SPEC_DRAFT_FULL_REPLICA"] = FULL_REPLICA
    os.environ["VLLM_SELF_SPEC_MOE_FP32_ACCUM"] = "0"
    os.environ["VLLM_SELF_SPEC_MOE_NUM_DUMP"] = OUT
    os.environ["VLLM_SELF_SPEC_MOE_DUMP_LAYER"] = str(LAYER)

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=MODEL,
        tensor_parallel_size=1,
        enable_expert_parallel=(ENABLE_EP == "1"),
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        enable_prefix_caching=False,
        disable_log_stats=True,
    )
    llm = LLM(**kwargs)

    prompt = (
        "The history of artificial intelligence began in antiquity, with "
        "myths and stories of artificial beings endowed with intelligence by "
        "master craftsmen. In modern times, the field of AI research was "
        "founded at a workshop held on the campus of Dartmouth College"
    )
    sp = SamplingParams(temperature=0.0, max_tokens=1, seed=0)

    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "ARM"), "w").close()
    llm.generate([prompt], sp, use_tqdm=False)
    if rank == 0:
        print(f"[DUMP] cfg={CFG} fr={FULL_REPLICA} layer={LAYER} -> {OUT}",
              flush=True)


def main():
    from vllm.utils.network_utils import get_open_port
    master_ip, master_port = "127.0.0.1", get_open_port()
    procs = [Process(target=worker, args=(r, master_ip, master_port))
             for r in range(DP)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=900)
    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join(timeout=30)


if __name__ == "__main__":
    main()
