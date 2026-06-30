"""Phase 37 headline: Qwen3-30B-A3B DP=8 + EP=8, FP8 full-replica comm-free
self-spec draft, target bf16, forced-PCIe, FULL-CG, with the compile-consistency
fix (VLLM_SELF_SPEC_COMPILE_CONSISTENT=1).

Measures, on the SAME workload:
  - spec accept_len (rank 0 aggregated spec metrics)
  - spec gen wall time
  - no-spec gen wall time (a separate run, MODE=nospec)
  - lossless check: spec vs no-spec produce identical greedy output tokens

Teardown: each worker is its own process; main joins/terminates on timeout. On
crash the OS reaps the children (we also terminate()).

  MODE=spec|nospec  W7_DP=8  python qwen30b_dp8.py
"""
import json
import os
import time
from multiprocessing import Process, Queue

MODEL = os.environ.get(
    "W7_MODEL",
    "/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B"
    "/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
)
DP = int(os.environ.get("W7_DP", "8"))
TP = int(os.environ.get("W7_TP", "1"))
K = int(os.environ.get("W7_K", "4"))
BATCH = int(os.environ.get("W7_BATCH", "16"))
OUTLEN = int(os.environ.get("W7_OUTLEN", "96"))
MAXLEN = int(os.environ.get("W7_MAXLEN", "2048"))
GPU_MEM = float(os.environ.get("W7_GPU_MEM", "0.88"))
MODE = os.environ.get("MODE", "spec")
OUT = os.environ.get(
    "W7_OUT", "/data/smcho/ssm-cc/research/37_compile_consistency/data"
)


def _mv(metrics, name):
    for m in metrics:
        if m.name == name:
            v = getattr(m, "value", None)
            if v is not None:
                return v
            if hasattr(m, "values"):
                return sum(m.values)
    return None


def worker(rank, master_ip, master_port, q):
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(DP)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)
    if MODE == "spec":
        os.environ["VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE"] = "1"
        os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = "0"
        os.environ["VLLM_SELF_SPEC_DRAFT_FULL_REPLICA"] = "1"
        os.environ["VLLM_SELF_SPEC_DRAFT_FULL_CG"] = "1"
        os.environ["VLLM_SELF_SPEC_COMPILE_CONSISTENT"] = "1"

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=MODEL,
        tensor_parallel_size=TP,
        enable_expert_parallel=True,
        trust_remote_code=True,
        max_model_len=MAXLEN,
        gpu_memory_utilization=GPU_MEM,
        enforce_eager=False,
        disable_log_stats=False,
    )
    if MODE == "spec":
        kwargs["speculative_config"] = {
            "method": "draft_model",
            "model": MODEL,
            "num_speculative_tokens": K,
            "draft_tensor_parallel_size": TP,
            "quantization": "fp8",
        }
    try:
        llm = LLM(**kwargs)
    except Exception as e:
        if rank == 0:
            q.put({"error": f"init: {repr(e)[:400]}"})
        return

    base = ("The history of artificial intelligence began in antiquity, with "
            "myths and stories of artificial beings. In modern times, the field "
            "was founded in")
    prompts = [f"{base} the year {1900 + i}." for i in range(BATCH)]
    sp = SamplingParams(temperature=0.0, max_tokens=OUTLEN, ignore_eos=True, seed=0)

    def snap():
        m = llm.get_metrics()
        return (
            _mv(m, "vllm:spec_decode_num_accepted_tokens") or 0.0,
            _mv(m, "vllm:spec_decode_num_drafts") or 0.0,
        )

    try:
        # warmup
        llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=8, seed=0),
                     use_tqdm=False)
        s0 = snap()
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        s1 = snap()
        if rank == 0:
            tokens = [list(o.outputs[0].token_ids) for o in outs]
            ntok = sum(len(t) for t in tokens)
            accepted = s1[0] - s0[0]
            ndrafts = s1[1] - s0[1]
            al = (1 + accepted / ndrafts) if ndrafts else None
            q.put({
                "mode": MODE, "dp": DP, "K": K, "accept_len": al,
                "accepted": accepted, "ndrafts": ndrafts, "gen_s": dt,
                "ntok": ntok, "tokens": tokens,
            })
    except Exception as e:
        if rank == 0:
            q.put({"error": f"gen: {repr(e)[:400]}", "mode": MODE})


def main():
    from vllm.utils.network_utils import get_open_port
    master_ip, master_port = "127.0.0.1", get_open_port()
    q = Queue()
    procs = [Process(target=worker, args=(r, master_ip, master_port, q))
             for r in range(DP)]
    for p in procs:
        p.start()
    res = None
    try:
        res = q.get(timeout=3000)
    except Exception:
        res = {"error": "no result (timeout)"}
    for p in procs:
        p.join(timeout=90)
    for p in procs:
        if p.is_alive():
            p.terminate()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"qwen30b_dp{DP}_{MODE}.json")
    with open(path, "w") as f:
        json.dump(res, f)
    short = {k: v for k, v in (res or {}).items() if k != "tokens"}
    print(f"RESULT30B mode={MODE} {json.dumps(short)[:400]}", flush=True)


if __name__ == "__main__":
    main()
