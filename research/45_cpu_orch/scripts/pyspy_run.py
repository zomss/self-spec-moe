"""Launch a DP=8 comm-free self-spec engine and drive a LONG steady decode so
py-spy can sample the EngineCore/Worker wall-clock CPU during decode.

Runs one rank per process (like the recon harness). Rank 0 prints its own PID
and the PIDs are discoverable via pgrep. The long generate keeps the decode
loop busy for ~PYSPY_SECS so an external `py-spy record/dump` can attach.
"""
import os
import sys
import time
from multiprocessing import Process

MODEL = os.environ.get("CP_MODEL", "Qwen/Qwen1.5-MoE-A2.7B")
DP = int(os.environ.get("CP_DP", "8"))
BATCH = int(os.environ.get("CP_BATCH", "64"))
K = int(os.environ.get("CP_K", "2"))
DRAFT_QUANT = os.environ.get("CP_DRAFT_QUANT", "fp8").strip()
GPU_MEM = float(os.environ.get("CP_GPU_MEM", "0.90"))
MAX_MODEL_LEN = int(os.environ.get("CP_MAX_MODEL_LEN", "2048"))
OUTLEN = int(os.environ.get("PYSPY_OUTLEN", "1200"))
KNOBS = [kv for kv in os.environ.get("CP_KNOBS", "").split(",") if kv.strip()]

BASE_PROMPT = (
    "The history of artificial intelligence began in antiquity with myths and "
    "stories, and the modern field was founded in"
)


def worker(rank, dp, master_ip, master_port):
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(dp)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)
    os.environ["VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE"] = "1"
    os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = "0"
    os.environ["VLLM_SELF_SPEC_DRAFT_FULL_REPLICA"] = "1"
    os.environ["VLLM_SELF_SPEC_DRAFT_FULL_CG"] = "1"
    os.environ["VLLM_SELF_SPEC_COMPILE_CONSISTENT"] = "1"
    for kv in KNOBS:
        k, _, v = kv.partition("=")
        os.environ[k.strip()] = v.strip()

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=MODEL, tensor_parallel_size=1, enable_expert_parallel=True,
        trust_remote_code=False, max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM, enforce_eager=False,
        disable_log_stats=False,
        speculative_config={
            "method": "draft_model", "model": MODEL,
            "num_speculative_tokens": K, "draft_tensor_parallel_size": 1,
            **({"quantization": DRAFT_QUANT} if DRAFT_QUANT else {}),
        },
    )
    llm = LLM(**kwargs)
    prompts = [f"{BASE_PROMPT} the year {1900 + i}." for i in range(BATCH)]

    def gen(n):
        sp = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True, seed=0)
        return llm.generate(prompts, sp, use_tqdm=False)

    gen(32)  # warm
    if rank == 0:
        print(f"[PYSPY] rank0 worker pid={os.getpid()} STEADY-DECODE-START",
              flush=True)
    t0 = time.perf_counter()
    gen(OUTLEN)
    if rank == 0:
        print(f"[PYSPY] rank0 decode done in {time.perf_counter()-t0:.1f}s",
              flush=True)


def main():
    from vllm.utils.network_utils import get_open_port
    master_ip = "127.0.0.1"
    master_port = get_open_port()
    procs = []
    for rank in range(DP):
        p = Process(target=worker, args=(rank, DP, master_ip, master_port))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
