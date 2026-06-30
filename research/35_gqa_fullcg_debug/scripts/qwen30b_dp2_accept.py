"""W7-gqa Step3: Qwen3-30B-A3B DP=2 accept_len, FULL-CG-fixed vs PIECEWISE.

Full World-A self-spec stack at DP=2 + EP=2, forced-PCIe (driver sets NCCL_*),
FP8 full-replica draft, target bf16. Reports accept_len so we can confirm the
GQA FULL-CG fix lifts accept from ~1.9 -> ~4-5 at DP=2.

  MODE=fullcg|piecewise python qwen30b_dp2_accept.py
"""

import os
import sys
import time
from multiprocessing import Process, Queue

MODEL = os.environ.get(
    "W7_MODEL",
    "/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B"
    "/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
)
DP = int(os.environ.get("W7_DP", "2"))
TP = int(os.environ.get("W7_TP", "1"))
K = int(os.environ.get("MB_K", "4"))
BATCH = int(os.environ.get("MB_BATCH", "16"))
OUTLEN = int(os.environ.get("MB_OUTLEN", "96"))
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


def worker(rank, dp, master_ip, master_port, mode, q):
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(dp)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)
    # Full World-A self-spec stack.
    os.environ["VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE"] = "1"
    os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = "0"
    os.environ["VLLM_SELF_SPEC_DRAFT_FULL_REPLICA"] = "1"
    os.environ["VLLM_SELF_SPEC_DRAFT_FULL_CG"] = "1" if mode == "fullcg" else "0"

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=TP,
        enable_expert_parallel=True,
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM,
        enforce_eager=False,
        disable_log_stats=False,
        speculative_config={
            "method": "draft_model",
            "model": MODEL,
            "num_speculative_tokens": K,
            "draft_tensor_parallel_size": TP,
            "quantization": "fp8",
        },
    )

    base = ("The history of artificial intelligence began in antiquity, with "
            "myths and stories of artificial beings. In modern times, the field "
            "was founded in")
    prompts = [f"{base} the year {1900 + i}." for i in range(BATCH)]
    sp = SamplingParams(temperature=0.0, max_tokens=OUTLEN, ignore_eos=True, seed=0)

    def snap():
        m = llm.get_metrics()
        return (
            _metric_value(m, "vllm:spec_decode_num_accepted_tokens") or 0.0,
            _metric_value(m, "vllm:spec_decode_num_drafts") or 0.0,
        )

    llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=8, seed=0),
                 use_tqdm=False)
    s0 = snap()
    t0 = time.perf_counter()
    llm.generate(prompts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    s1 = snap()
    if rank == 0:
        accepted = s1[0] - s0[0]
        ndrafts = s1[1] - s0[1]
        al = (1 + accepted / ndrafts) if ndrafts else None
        q.put({"accept_len": al, "accepted": accepted, "ndrafts": ndrafts,
               "long_s": dt})


def main():
    mode = os.environ.get("MODE", "fullcg")
    from vllm.utils.network_utils import get_open_port
    master_ip, master_port = "127.0.0.1", get_open_port()
    q = Queue()
    procs = [Process(target=worker, args=(r, DP, master_ip, master_port, mode, q))
             for r in range(DP)]
    for p in procs:
        p.start()
    res = None
    try:
        res = q.get(timeout=2400)
    except Exception:
        res = {"error": "timeout"}
    for p in procs:
        p.join(timeout=60)
    for p in procs:
        if p.is_alive():
            p.terminate()
    print(f"RESULT30B mode={mode} {res}", flush=True)


if __name__ == "__main__":
    main()
