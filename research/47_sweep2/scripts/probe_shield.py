"""Phase 47 draft-shielding probe (piecewise K=2, fully-optimized comm-free spec).

Question (Report item 4): with the genuine full-replica draft (use_ep=False) +
PIECEWISE chain, does the draft still eat the injected A2A sleep, or is it now
truly comm-free (real=0 collectives AND not charged the emulated sleep)?

Builds the DP=8/EP=8 spec engine with the SAME env as the sweep (full replica +
local route + full CG + compile-consistent + PIECEWISE), reads the AgRs manager's
_emulated_a2a_count (total, incl. draft's local-route dispatch/combine that STILL
calls torch.cuda._sleep) and _real_collective_count (true all_gatherv/reduce_scatterv,
0 for a genuine comm-free draft's MoE) directly in-process over a measured decode
window, writes per-rank JSONs.

Interpretation:
  real == 0                 -> draft issues NO cross-rank collective (comm-free). GOOD.
  total  > real (== verify) -> the draft STILL runs _emulate_exposed_a2a_delay on its
                               local-route dispatch/combine (over-charge; the sleep is
                               NOT gated by self_spec_local_route_enabled in all2all.py).
  verify collectives/step   = real / decode_steps  (what a shielded multi-node draft
                               would pay).

Env: A2A_US=<float>  KVAL=<int default 2>  STEPS_BATCH  OUTLEN
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
TP = 1
A2A_US = float(os.environ.get("A2A_US", "500"))
KVAL = int(os.environ.get("KVAL", "2"))
BATCH = int(os.environ.get("STEPS_BATCH", "8"))
OUTLEN = int(os.environ.get("OUTLEN", "48"))
COUNTD = os.environ.get(
    "COUNTD", "/data/smcho/self-spec-moe/research/47_sweep2/data/probe_counts"
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
    # Full fully-optimized comm-free spec stack (matches the sweep).
    os.environ["VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE"] = "1"
    os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = "0"
    os.environ["VLLM_SELF_SPEC_DRAFT_FULL_REPLICA"] = "1"
    os.environ["VLLM_SELF_SPEC_DRAFT_FULL_CG"] = "1"
    os.environ["VLLM_SELF_SPEC_COMPILE_CONSISTENT"] = "1"
    os.environ["VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE"] = "1"

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=MODEL, tensor_parallel_size=TP, enable_expert_parallel=True,
        trust_remote_code=False, max_model_len=2048,
        gpu_memory_utilization=0.90, enforce_eager=False,
        disable_log_stats=False,
    )
    kwargs["speculative_config"] = {
        "method": "draft_model", "model": MODEL,
        "num_speculative_tokens": KVAL, "draft_tensor_parallel_size": TP,
        "quantization": "fp8",
    }
    llm = LLM(**kwargs)

    prompts = [f"The year {1900 + i}. Tell a long story about" for i in range(BATCH)]
    sp = SamplingParams(temperature=0.0, max_tokens=OUTLEN, ignore_eos=True, seed=0)
    llm.generate(prompts, sp, use_tqdm=False)  # warm

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
    # decode steps in the measured window: BATCH seqs * OUTLEN tokens, but with spec
    # each *step* emits accept_len tokens; #target-forward steps ~ OUTLEN/accept_len.
    rec = {"rank": rank, "mode": "spec_piecewise", "a2a_us": A2A_US, "k": KVAL,
           "batch": BATCH, "outlen": OUTLEN, "window_s": dt,
           "delta_total_collectives": total, "delta_real_collectives": real,
           "abs_total": t_after, "abs_real": r_after}
    os.makedirs(COUNTD, exist_ok=True)
    with open(os.path.join(COUNTD, f"probe_spec_pw_a2a{int(A2A_US)}_r{rank}.json"), "w") as f:
        json.dump(rec, f, indent=2)
    if rank == 0:
        q.put(rec)
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
        if p.is_alive():
            p.terminate()
    print("PROBE_RESULT", json.dumps(rec))


if __name__ == "__main__":
    main()
