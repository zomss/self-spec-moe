"""W7 DP-accept isolation: measure mean accept_len for World A full-replica
self-spec draft across (DP, draft_quant) on a small GQA-MoE that fits at DP=1
and DP=2 (Qwen1.5-MoE-A2.7B by default).

Self-spec: draft_model method, draft == target, full-replica comm-free draft
(VLLM_SELF_SPEC_DRAFT_FULL_REPLICA=1), greedy. PIECEWISE (CUDA graphs on,
no draft FULL-CG) to avoid the cudagraph variable.

Each process is one DP rank (manual VLLM_DP_* env). Only rank 0 has the
aggregated spec-decode metrics, so it reports accept_len.

Env knobs:
  AI_MODEL    HF path/id          default Qwen1.5-MoE-A2.7B local snapshot
  AI_DP       data-parallel size  default 1
  AI_TP       tensor-parallel     default 1
  AI_QUANT    draft quantization  ("" -> bf16 draft)  default ""
  AI_K        num_speculative     default 4
  AI_BATCH    batch size          default 16
  AI_OUTLEN   max output tokens   default 128
  AI_EP       1 -> target EP on   default 1
  AI_GPU_MEM  gpu_mem_util        default 0.85
  AI_MAXLEN   max_model_len       default 2048
  AI_TRC      1 -> trust_remote   default 0
  AI_EAGER    1 -> enforce_eager  default 0
  AI_TAG      tag for json file   default isolate
  AI_OUT      output dir
"""
import json
import os
import sys
import time
from multiprocessing import Process, Queue

MODEL = os.environ.get(
    "AI_MODEL",
    "/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B"
    "/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9",
)
DP = int(os.environ.get("AI_DP", "1"))
TP = int(os.environ.get("AI_TP", "1"))
QUANT = os.environ.get("AI_QUANT", "").strip()
K = int(os.environ.get("AI_K", "4"))
BATCH = int(os.environ.get("AI_BATCH", "16"))
OUTLEN = int(os.environ.get("AI_OUTLEN", "128"))
EP = os.environ.get("AI_EP", "1") == "1"
GPU_MEM = float(os.environ.get("AI_GPU_MEM", "0.85"))
MAXLEN = int(os.environ.get("AI_MAXLEN", "2048"))
TRC = os.environ.get("AI_TRC", "0") == "1"
EAGER = os.environ.get("AI_EAGER", "0") == "1"
TAG = os.environ.get("AI_TAG", "isolate")
OUT_DIR = os.environ.get(
    "AI_OUT", "/data/smcho/ssm-cc/research/37_compile_consistency/data"
)

BASE_PROMPT = (
    "The history of artificial intelligence began in antiquity with myths and "
    "stories, and the modern field was founded in"
)


def _metric_value(metrics, name):
    total = 0.0
    found = False
    for m in metrics:
        if m.name == name:
            found = True
            v = getattr(m, "value", None)
            if v is not None:
                total += v
            else:
                vals = getattr(m, "values", None)
                if vals is not None:
                    total += sum(vals)
    return total if found else None


def worker(rank, q):
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(DP)
    os.environ["VLLM_DP_MASTER_IP"] = "127.0.0.1"
    os.environ["VLLM_DP_MASTER_PORT"] = os.environ["AI_MASTER_PORT"]

    fullrep = os.environ.get("AI_FULLREP", "1") == "1"
    # AI_LOCALROUTE defaults to fullrep value, but can be forced independently
    localroute = os.environ.get("AI_LOCALROUTE", "1" if fullrep else "0") == "1"
    os.environ["VLLM_SELF_SPEC_DRAFT_FULL_REPLICA"] = "1" if fullrep else "0"
    os.environ["VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE"] = "1" if localroute else "0"
    os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = "0"

    from vllm import LLM, SamplingParams

    spec_cfg = {
        "method": "draft_model",
        "model": MODEL,
        "num_speculative_tokens": K,
        "draft_tensor_parallel_size": TP,
    }
    if QUANT:
        spec_cfg["quantization"] = QUANT

    kwargs = dict(
        model=MODEL,
        tensor_parallel_size=TP,
        enable_expert_parallel=EP,
        trust_remote_code=TRC,
        max_model_len=MAXLEN,
        gpu_memory_utilization=GPU_MEM,
        enforce_eager=EAGER,
        disable_log_stats=False,
        speculative_config=spec_cfg,
    )
    cc = {}
    cgmode = os.environ.get("AI_CGMODE", "").strip()
    if cgmode:
        cc["cudagraph_mode"] = cgmode
    if os.environ.get("AI_NO_COMBO", "0") == "1":
        cc["inductor_compile_config"] = {"combo_kernels": False,
                                         "benchmark_combo_kernel": False}
    if cc:
        kwargs["compilation_config"] = cc
    try:
        llm = LLM(**kwargs)
    except Exception as e:
        if rank == 0:
            q.put({"error": "init: " + repr(e)[:400]})
        return

    prompts = [f"{BASE_PROMPT} the year {1900 + i}." for i in range(BATCH)]
    sp = SamplingParams(temperature=0.0, max_tokens=OUTLEN, ignore_eos=True, seed=0)

    try:
        # warmup
        llm.generate(prompts, sp, use_tqdm=False)

        def snap():
            if rank != 0:
                return None
            m = llm.get_metrics()
            return (
                _metric_value(m, "vllm:spec_decode_num_accepted_tokens") or 0.0,
                _metric_value(m, "vllm:spec_decode_num_draft_tokens") or 0.0,
                _metric_value(m, "vllm:spec_decode_num_drafts") or 0.0,
            )

        s0 = snap()
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        s1 = snap()
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)

        if rank == 0:
            accepted = s1[0] - s0[0]
            drafted = s1[1] - s0[1]
            ndrafts = s1[2] - s0[2]
            accept_len = (1 + accepted / ndrafts) if ndrafts else None
            per_tok = (accepted / drafted) if drafted else None
            row = {
                "model": MODEL, "dp": DP, "tp": TP,
                "draft_quant": QUANT or "bf16", "K": K, "batch": BATCH,
                "outlen": OUTLEN, "ep": EP, "eager": EAGER,
                "accept_len": accept_len, "per_token_accept": per_tok,
                "num_accepted": accepted, "num_drafted": drafted,
                "num_drafts": ndrafts, "ntok": ntok, "gen_s": dt,
            }
            print(f"[ACCEPT dp={DP} q={QUANT or 'bf16'} K={K}] "
                  f"accept_len={accept_len} per_tok={per_tok} "
                  f"(acc={accepted} drafted={drafted} ndrafts={ndrafts})",
                  flush=True)
            q.put(row)
    except Exception as e:
        if rank == 0:
            q.put({"error": "gen: " + repr(e)[:400], "dp": DP,
                   "draft_quant": QUANT or "bf16"})


def main():
    from vllm.utils.network_utils import get_open_port
    os.environ["AI_MASTER_PORT"] = str(get_open_port())
    os.makedirs(OUT_DIR, exist_ok=True)
    q = Queue()
    procs = []
    for rank in range(DP):
        p = Process(target=worker, args=(rank, q))
        p.start()
        procs.append(p)
    res = None
    try:
        res = q.get(timeout=2400)
    except Exception:
        res = {"error": "no result from rank 0 (timeout)"}
    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
    path = os.path.join(OUT_DIR, f"accept_{TAG}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[ACCEPT] wrote {path}: {json.dumps(res)[:300]}", flush=True)


if __name__ == "__main__":
    main()
