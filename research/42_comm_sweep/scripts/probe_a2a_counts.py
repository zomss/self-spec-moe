"""Fairness probe: does the comm-free full-replica DRAFT pay the emulated A2A
delay, and does no-spec pay it?

Builds a DP=8/EP=8 engine (spec or nospec), runs a fixed number of decode steps,
then GRACEFULLY shuts down so AgRsAll2AllManager.destroy() runs and logs
total/active/real counts. Compares:
  - nospec: real == total (every collective is a real all_gatherv/reduce_scatterv).
  - spec  : if the draft routes through AgRs comm-free, total >> real (the gap is
            the draft dispatch/combine calls; each still ran _emulate_exposed_a2a
            but issued NO real collective, so real counts only the verify).
            if the draft never touches AgRs, total == real (verify only).

We ALSO write per-rank counts to files so the parent can aggregate even if the
worker log is truncated by teardown.

Env: MODE=spec|nospec  A2A_US=<float>  KVAL=<int>  STEPS_BATCH=<int>
"""
import json
import os
import sys
import time
from multiprocessing import Process, Queue

MODEL = os.environ.get(
    "W7_MODEL",
    "/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B"
    "/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
)
DP = int(os.environ.get("W7_DP", "8"))
TP = 1
A2A_US = float(os.environ.get("A2A_US", "500"))
KVAL = int(os.environ.get("KVAL", "4"))
BATCH = int(os.environ.get("STEPS_BATCH", "8"))
OUTLEN = int(os.environ.get("OUTLEN", "48"))
MODE = os.environ.get("MODE", "spec")
COUNTD = os.environ.get(
    "COUNTD", "/data/smcho/self-spec-moe/research/42_comm_sweep/data/probe_counts"
)


def worker(rank, master_ip, master_port, q):
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(DP)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)
    os.environ["VLLM_SELF_SPEC_LOG_A2A_COUNTS"] = "1"
    if A2A_US > 0:
        os.environ["VLLM_SELF_SPEC_EMULATE_A2A_DELAY_US"] = str(A2A_US)
    spec = MODE == "spec"
    if spec:
        os.environ["VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE"] = "1"
        os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = "0"
        os.environ["VLLM_SELF_SPEC_DRAFT_FULL_REPLICA"] = "1"

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=MODEL, tensor_parallel_size=TP, enable_expert_parallel=True,
        trust_remote_code=False, max_model_len=2048,
        gpu_memory_utilization=0.90, enforce_eager=False,
        disable_log_stats=False,
    )
    if spec:
        kwargs["speculative_config"] = {
            "method": "draft_model", "model": MODEL,
            "num_speculative_tokens": KVAL, "draft_tensor_parallel_size": TP,
            "quantization": "fp8",
        }
    llm = LLM(**kwargs)

    prompts = [f"The year {1900 + i}. Tell a long story about" for i in range(BATCH)]
    sp = SamplingParams(temperature=0.0, max_tokens=OUTLEN, ignore_eos=True, seed=0)
    llm.generate(prompts, sp, use_tqdm=False)  # warm
    # Reset counters by reading the shared AgRs manager and zeroing, then a
    # measured window of exactly OUTLEN decode steps per seq.
    from vllm.distributed.parallel_state import get_ep_group
    mgr = get_ep_group().device_communicator.all2all_manager
    t_before = getattr(mgr, "_emulated_a2a_count", None)
    r_before = getattr(mgr, "_real_collective_count", None)
    t0 = time.perf_counter()
    llm.generate(prompts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    t_after = getattr(mgr, "_emulated_a2a_count", None)
    r_after = getattr(mgr, "_real_collective_count", None)

    total = (t_after - t_before) if (t_after is not None and t_before is not None) else None
    real = (r_after - r_before) if (r_after is not None and r_before is not None) else None
    rec = {"rank": rank, "mode": MODE, "a2a_us": A2A_US, "k": KVAL,
           "batch": BATCH, "outlen": OUTLEN, "window_s": dt,
           "delta_total_collectives": total, "delta_real_collectives": real,
           "abs_total": t_after, "abs_real": r_after}
    os.makedirs(COUNTD, exist_ok=True)
    with open(os.path.join(COUNTD, f"probe_{MODE}_a2a{int(A2A_US)}_r{rank}.json"), "w") as f:
        json.dump(rec, f, indent=2)
    if rank == 0:
        q.put(rec)
    # Let manager.destroy() log fire too.
    del llm


def main():
    from vllm.utils.network_utils import get_open_port
    master_ip = "127.0.0.1"
    master_port = get_open_port()
    q = Queue()
    procs = []
    for rank in range(DP):
        p = Process(target=worker, args=(rank, master_ip, master_port, q))
        p.start()
        procs.append(p)
    rec = None
    try:
        rec = q.get(timeout=1800)
    except Exception:
        rec = {"error": "timeout"}
    for p in procs:
        p.join(timeout=60)
    print("PROBE_RESULT", json.dumps(rec))


if __name__ == "__main__":
    main()
