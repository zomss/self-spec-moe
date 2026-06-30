"""Phase 38: decisive EP-isolation MEASUREMENT (no source changes).

Question: World A's comm-free FULL-REPLICA draft accept COLLAPSES at large EP
(Qwen3-30B DP8 accept ~1.0, present even eager -> structural comm-free-replica
local-sum vs the EP all-to-all FP-reduction divergence). Does routing the draft
through the EP path (L1) recover it, before we build it?

Proxy: Qwen/Qwen1.5-MoE-A2.7B (qwen2_moe, 60 experts top-4, 24 layers, non-MLA),
DP=8 -> EP=8, forced-PCIe, greedy, bf16 (NO FP8 confound). Always set
VLLM_SELF_SPEC_DRAFT_FULL_CG=1 AND VLLM_SELF_SPEC_COMPILE_CONSISTENT=1 so the
per-step compile bug is FIXED -> the ONLY remaining divergence is comm-free-vs-EP.

Three draft routings (Test 1 isolates L1 = the EP-path draft):
  A full-replica comm-free  DRAFT_FULL_REPLICA=1 DRAFT_LOCAL_ROUTE=1
  B EP-shard comm-free      DRAFT_FULL_REPLICA=0 DRAFT_LOCAL_ROUTE=1
  C EP-full (upper bound)   DRAFT_FULL_REPLICA=0 DRAFT_LOCAL_ROUTE=0

Test 2 (isolates L2 = verify-context-KV): per-POSITION acceptance vector
(vllm:spec_decode_num_accepted_tokens_per_pos) tells us whether rejection is at
the FIRST draft token (depth-0 / per-step MoE divergence -> L2 won't help) or
accumulates with depth (KV drift -> L2 would help). Also sweep K=1 vs K=4.

Env knobs:
  CFG        A | B | C                (required)  -> sets the two routing flags
  E_DP       data-parallel size (=EP) default 8
  E_TP       tensor-parallel size     default 1
  E_K        num_speculative_tokens   default 4
  E_BATCH    batch size               default 16
  E_OUTLEN   output length            default 128
  E_MAXLEN   max_model_len            default 2048
  E_GPU_MEM  gpu_memory_utilization   default 0.85
  E_EAGER    1 -> enforce_eager       default 0 (CUDA graphs ON + FULL_CG)
  E_TAG      filename tag             default cfg
  E_OUT      output dir
"""
import json
import os
import time
from multiprocessing import Process, Queue

MODEL = os.environ.get(
    "E_MODEL",
    "/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B"
    "/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9",
)
CFG = os.environ["CFG"].strip().upper()
DP = int(os.environ.get("E_DP", "8"))
TP = int(os.environ.get("E_TP", "1"))
K = int(os.environ.get("E_K", "4"))
BATCH = int(os.environ.get("E_BATCH", "16"))
OUTLEN = int(os.environ.get("E_OUTLEN", "128"))
MAXLEN = int(os.environ.get("E_MAXLEN", "2048"))
GPU_MEM = float(os.environ.get("E_GPU_MEM", "0.85"))
EAGER = os.environ.get("E_EAGER", "0") == "1"
TAG = os.environ.get("E_TAG", "cfg")
OUT = os.environ.get(
    "E_OUT", "/data/smcho/self-spec-moe/research/40_num_divergence/data"
)

# CFG -> (FULL_REPLICA, LOCAL_ROUTE)
CFG_MAP = {
    "A": ("1", "1"),  # full-replica comm-free
    "B": ("0", "1"),  # EP-shard comm-free
    "C": ("0", "0"),  # EP-full (real all-to-all = the verify; upper bound)
}
assert CFG in CFG_MAP, f"CFG must be A/B/C, got {CFG}"
FULL_REPLICA, LOCAL_ROUTE = CFG_MAP[CFG]


def _mv(metrics, name):
    """Scalar counter value (summed across labels)."""
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


def _vec(metrics, name):
    """Per-position vector (list of per-position accepted counts)."""
    for m in metrics:
        if m.name == name:
            vals = getattr(m, "values", None)
            if vals is not None:
                return list(vals)
    return None


def worker(rank, master_ip, master_port, q):
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(DP)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

    # The compile bug is FIXED in all configs (the ONLY remaining divergence is
    # comm-free-vs-EP): draft FULL cudagraphs + batch-invariant numerics.
    os.environ["VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE"] = LOCAL_ROUTE
    os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = "0"  # verify stays full-EP
    os.environ["VLLM_SELF_SPEC_DRAFT_FULL_REPLICA"] = FULL_REPLICA
    # Phase 40 Step 2: FP32-accumulation in BOTH MoE reduce paths (the fix).
    os.environ["VLLM_SELF_SPEC_MOE_FP32_ACCUM"] = os.environ.get("E_FP32", "0")
    if not EAGER:
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
        enforce_eager=EAGER,
        disable_log_stats=False,
    )
    kwargs["speculative_config"] = {
        "method": "draft_model",
        "model": MODEL,
        "num_speculative_tokens": K,
        "draft_tensor_parallel_size": TP,
        # bf16 draft (no quantization key) -> NO FP8 confound.
    }
    try:
        llm = LLM(**kwargs)
    except Exception as e:
        if rank == 0:
            q.put({"error": f"init: {repr(e)[:500]}"})
        return

    base = ("The history of artificial intelligence began in antiquity, with "
            "myths and stories of artificial beings. In modern times, the field "
            "was founded in")
    prompts = [f"{base} the year {1900 + i}." for i in range(BATCH)]
    sp = SamplingParams(temperature=0.0, max_tokens=OUTLEN, ignore_eos=True, seed=0)

    POS = "vllm:spec_decode_num_accepted_tokens_per_pos"

    def snap():
        m = llm.get_metrics()
        return {
            "accepted": _mv(m, "vllm:spec_decode_num_accepted_tokens") or 0.0,
            "drafted": _mv(m, "vllm:spec_decode_num_draft_tokens") or 0.0,
            "ndrafts": _mv(m, "vllm:spec_decode_num_drafts") or 0.0,
            "per_pos": _vec(m, POS) or [],
        }

    try:
        # warmup (captures cudagraphs + clears transient)
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
            accepted = s1["accepted"] - s0["accepted"]
            drafted = s1["drafted"] - s0["drafted"]
            ndrafts = s1["ndrafts"] - s0["ndrafts"]
            # per-position delta
            pp0, pp1 = s0["per_pos"], s1["per_pos"]
            n = max(len(pp0), len(pp1))
            pp0 += [0] * (n - len(pp0))
            pp1 += [0] * (n - len(pp1))
            per_pos = [pp1[i] - pp0[i] for i in range(n)]
            al = (1 + accepted / ndrafts) if ndrafts else None
            per_tok = (accepted / drafted) if drafted else None
            # per-position acceptance RATE: at each depth i, fraction of drafts
            # that reached AND accepted position i. pos-0 rate = pp[0]/ndrafts.
            pos_rate = [p / ndrafts if ndrafts else None for p in per_pos]
            q.put({
                "cfg": CFG, "full_replica": FULL_REPLICA,
                "local_route": LOCAL_ROUTE, "dp": DP, "ep": DP, "K": K,
                "fp32_accum": os.environ.get("E_FP32", "0"),
                "batch": BATCH, "outlen": OUTLEN, "eager": EAGER,
                "accept_len": al, "per_tok": per_tok,
                "accepted": accepted, "drafted": drafted, "ndrafts": ndrafts,
                "per_pos_accepted": per_pos, "per_pos_rate": pos_rate,
                "gen_s": dt, "ntok": ntok,
                "tokens": tokens,
            })
    except Exception as e:
        if rank == 0:
            q.put({"error": f"gen: {repr(e)[:500]}", "cfg": CFG})


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
        res = q.get(timeout=2400)
    except Exception:
        res = {"error": "no result (timeout)"}
    for p in procs:
        p.join(timeout=90)
    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join(timeout=30)
    os.makedirs(OUT, exist_ok=True)
    suffix = ("eager" if EAGER else "cg")
    fp = "_fp32" if os.environ.get("E_FP32", "0") == "1" else ""
    path = os.path.join(OUT, f"acc_{TAG}_{CFG}_dp{DP}_K{K}_{suffix}{fp}.json")
    with open(path, "w") as f:
        json.dump(res, f)
    short = {k: v for k, v in (res or {}).items() if k != "tokens"}
    print(f"RESULT cfg={CFG} dp={DP} K={K} {json.dumps(short)[:500]}", flush=True)
    print(f"[EPISO] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
