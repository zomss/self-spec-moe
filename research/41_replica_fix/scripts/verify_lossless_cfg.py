"""Phase 41: lossless probe parametrized by config (A full-replica vs C EP-full
draft) at DP=8, against the SAME harness, so we can tell whether any spec-vs-nospec
token divergence is specific to the full-replica fix (A) or a pre-existing
batched-verify property shared by the EP-full draft (C).

argv[1] = A | C | nospec
Writes data/ll_<cfg>.json with rank-0 output token_ids.
"""
import json
import os
import sys
from multiprocessing import Process, Queue

MODEL = os.environ.get(
    "E_MODEL",
    "/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B"
    "/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9",
)
DP = int(os.environ.get("E_DP", "8"))
K = int(os.environ.get("E_K", "4"))
BATCH = int(os.environ.get("E_BATCH", "16"))
OUTLEN = int(os.environ.get("E_OUTLEN", "64"))
OUT = os.environ.get("E_OUT", "/data/smcho/ssm-num/research/41_replica_fix/data")
CFG = sys.argv[1] if len(sys.argv) > 1 else "A"

# CFG -> (spec?, FULL_REPLICA, LOCAL_ROUTE)
CMAP = {"A": (True, "1", "1"), "C": (True, "0", "0"), "nospec": (False, "0", "0")}


def worker(rank, master_ip, master_port, q):
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(DP)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)
    spec, fr, lr = CMAP[CFG]
    if spec:
        os.environ["VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE"] = lr
        os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = "0"
        os.environ["VLLM_SELF_SPEC_DRAFT_FULL_REPLICA"] = fr
        os.environ["VLLM_SELF_SPEC_DRAFT_FULL_CG"] = "1"
        os.environ["VLLM_SELF_SPEC_COMPILE_CONSISTENT"] = "1"

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=MODEL, tensor_parallel_size=1, enable_expert_parallel=True,
        trust_remote_code=True, max_model_len=2048, gpu_memory_utilization=0.85,
        enforce_eager=False, disable_log_stats=False,
    )
    if spec:
        kwargs["speculative_config"] = {
            "method": "draft_model", "model": MODEL,
            "num_speculative_tokens": K, "draft_tensor_parallel_size": 1,
        }
    llm = LLM(**kwargs)
    base = ("The history of artificial intelligence began in antiquity, with "
            "myths and stories of artificial beings. In modern times, the field "
            "was founded in")
    prompts = [f"{base} the year {1900 + i}." for i in range(BATCH)]
    sp = SamplingParams(temperature=0.0, max_tokens=OUTLEN, ignore_eos=True, seed=0)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    if rank == 0:
        q.put({"cfg": CFG, "tokens": [list(o.outputs[0].token_ids) for o in outs]})


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
        res = q.get(timeout=2000)
    except Exception:
        res = {"error": "timeout"}
    for p in procs:
        p.join(timeout=120)
    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join(timeout=30)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"ll_{CFG}.json"), "w") as f:
        json.dump(res, f)
    print(f"[LL] wrote ll_{CFG}.json", flush=True)


if __name__ == "__main__":
    main()
