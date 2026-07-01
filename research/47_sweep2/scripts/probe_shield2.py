"""Phase 47 draft-shielding probe v2 (robust): capture the AgRs manager's
total/active/real collective counts via the destroy() log line.

In V1 the model executes in separate EngineCore worker processes, so reading the
manager counters from the launcher process is unreliable. Instead we let the
engine shut down GRACEFULLY (del llm) so AgRsAll2AllManager.destroy() fires and
logs 'Self-spec AgRs all2all count: total=.. active=.. real=..' from the worker
process (VLLM_SELF_SPEC_LOG_A2A_COUNTS=1). We do NOT grep-filter stdout, so the
worker log line is preserved.

Runs the fully-optimized comm-free spec stack (full replica + local route + full
CG + compile-consistent + piecewise), K=2, at the given A2A_US. A single short
generate is enough: the ratio total/real is structural (per-forward), independent
of step count.

Env: A2A_US, KVAL(=2), STEPS_BATCH(=4), OUTLEN(=16)
"""
import os
from multiprocessing import Process

MODEL = os.environ.get(
    "W7_MODEL",
    "/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B"
    "/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
)
DP = int(os.environ.get("W7_DP", "8"))
A2A_US = float(os.environ.get("A2A_US", "500"))
KVAL = int(os.environ.get("KVAL", "2"))
BATCH = int(os.environ.get("STEPS_BATCH", "4"))
OUTLEN = int(os.environ.get("OUTLEN", "16"))


def worker(rank, master_ip, master_port):
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(DP)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)
    os.environ["VLLM_SELF_SPEC_LOG_A2A_COUNTS"] = "1"
    if A2A_US > 0:
        os.environ["VLLM_SELF_SPEC_EMULATE_A2A_DELAY_US"] = str(A2A_US)
    os.environ["VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE"] = "1"
    os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = "0"
    os.environ["VLLM_SELF_SPEC_DRAFT_FULL_REPLICA"] = "1"
    os.environ["VLLM_SELF_SPEC_DRAFT_FULL_CG"] = "1"
    os.environ["VLLM_SELF_SPEC_COMPILE_CONSISTENT"] = "1"
    os.environ["VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE"] = "1"

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL, tensor_parallel_size=1, enable_expert_parallel=True,
        trust_remote_code=False, max_model_len=2048,
        gpu_memory_utilization=0.90, enforce_eager=False, disable_log_stats=False,
        speculative_config={
            "method": "draft_model", "model": MODEL,
            "num_speculative_tokens": KVAL, "draft_tensor_parallel_size": 1,
            "quantization": "fp8",
        },
    )
    prompts = [f"The year {1900 + i}. Tell a long story about" for i in range(BATCH)]
    sp = SamplingParams(temperature=0.0, max_tokens=OUTLEN, ignore_eos=True, seed=0)
    llm.generate(prompts, sp, use_tqdm=False)
    # Graceful teardown -> AgRsAll2AllManager.destroy() logs the counts.
    del llm


def main():
    from vllm.utils.network_utils import get_open_port
    master_ip = "127.0.0.1"
    master_port = get_open_port()
    procs = []
    for rank in range(DP):
        p = Process(target=worker, args=(rank, master_ip, master_port))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=1200)
        if p.is_alive():
            p.terminate()
    print("PROBE2_DONE")


if __name__ == "__main__":
    main()
