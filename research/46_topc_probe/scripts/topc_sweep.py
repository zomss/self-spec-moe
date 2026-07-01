"""Phase 46 W7: top-C draft-prune elasticity probe.

For a single (model, C) point, launch one DP engine group with the comm-free
full-replica self-spec draft (piecewise chain on) and VLLM_SELF_SPEC_DRAFT_TOPC=C,
then record:
  - accept_len (from spec_decode metrics: 1 + accepted/num_drafts)
  - draft_forward mean_ms (env profiler, the in-chain single draft forward)
  - verify mean_ms and draft_chain mean_ms (context)
  - sys_tok_s over the timed run
  - optionally token_ids for a byte-identical losslessness check

Mirrors research/44_recon/scripts/recon_microbench.py (proven W7 launch + DP
worker + profiler read). The DRAFT computes only C experts/token; the VERIFY
keeps full top_k (lossless regardless of C).

Env knobs (TC_*):
  TC_MODEL     HF path/id (required)
  TC_C         top-C value (required; C==top_k -> off/full draft when >=top_k)
  TC_DP        default 8
  TC_TP        default 1
  TC_TRC       1 -> trust_remote_code   default 0
  TC_BATCH     global batch (n seqs)    default 64
  TC_K         num_speculative_tokens   default 2
  TC_DRAFT_QUANT  draft quant; "" -> bf16 draft  default fp8
  TC_GPU_MEM   default 0.90
  TC_MAX_MODEL_LEN default 2048
  TC_STEADY    steady decode steps      default 80
  TC_WARMUP    profiler warmup to drop  default 60
  TC_CAPTURE_TOKENS 1 -> save output token_ids (for losslessness)  default 0
  TC_OUT       results dir
  TC_TAG       filename tag
"""
import glob
import json
import os
import sys
import time
from multiprocessing import Process, Queue

MODEL = os.environ["TC_MODEL"]
C = int(os.environ["TC_C"])
DP = int(os.environ.get("TC_DP", "8"))
TP = int(os.environ.get("TC_TP", "1"))
TRC = os.environ.get("TC_TRC", "0") == "1"
BATCH = int(os.environ.get("TC_BATCH", "64"))
K = int(os.environ.get("TC_K", "2"))
DRAFT_QUANT = os.environ.get("TC_DRAFT_QUANT", "fp8").strip()
GPU_MEM = float(os.environ.get("TC_GPU_MEM", "0.90"))
MAX_MODEL_LEN = int(os.environ.get("TC_MAX_MODEL_LEN", "2048"))
STEADY = int(os.environ.get("TC_STEADY", "80"))
WARMUP = int(os.environ.get("TC_WARMUP", "60"))
CAPTURE_TOKENS = os.environ.get("TC_CAPTURE_TOKENS", "0") == "1"
NOSPEC = os.environ.get("TC_NOSPEC", "0") == "1"  # no-spec reference (losslessness)
OUT_DIR = os.environ.get("TC_OUT", os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))) + "/data")
TAG = os.environ.get("TC_TAG", "model")

BASE_PROMPT = (
    "The history of artificial intelligence began in antiquity with myths and "
    "stories, and the modern field was founded in"
)

OUTLEN = WARMUP + STEADY + 80
SHORTLEN = 32

REGIONS = ("draft_forward_first", "draft_forward", "draft_chain", "verify")


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


def worker(rank, dp, tp, master_ip, master_port, profile_dir, q):
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(dp)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

    os.environ["VLLM_SELF_SPEC_PROFILE"] = "1"
    os.environ["VLLM_SELF_SPEC_PROFILE_OUT"] = profile_dir
    os.environ["VLLM_SELF_SPEC_PROFILE_WARMUP"] = str(WARMUP)

    if not NOSPEC:
        # Genuine comm-free full-replica draft (all experts resident, local
        # route -> no collective). Piecewise chain + full CG + consistent
        # compile are set in the launcher env (run_topc.sh).
        os.environ["VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE"] = "1"
        os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = "0"
        os.environ["VLLM_SELF_SPEC_DRAFT_FULL_REPLICA"] = "1"
        # The probe: draft computes only C experts/token (0/off when C>=top_k
        # is handled inside prune, but we set exactly C).
        os.environ["VLLM_SELF_SPEC_DRAFT_TOPC"] = str(C)

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=MODEL,
        tensor_parallel_size=tp,
        enable_expert_parallel=True,
        trust_remote_code=TRC,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM,
        enforce_eager=False,
        disable_log_stats=False,
    )
    if not NOSPEC:
        spec_cfg = {
            "method": "draft_model",
            "model": MODEL,
            "num_speculative_tokens": K,
            "draft_tensor_parallel_size": tp,
        }
        if DRAFT_QUANT:
            spec_cfg["quantization"] = DRAFT_QUANT
        kwargs["speculative_config"] = spec_cfg
    llm = LLM(**kwargs)

    def run_batch(out_len):
        prompts = [f"{BASE_PROMPT} the year {1900 + i}." for i in range(BATCH)]
        sp = SamplingParams(
            temperature=0.0, max_tokens=out_len, ignore_eos=True, seed=0,
        )
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        return dt, ntok, outs

    def _snap():
        if rank != 0:
            return None
        m = llm.get_metrics()
        return (
            _metric_value(m, "vllm:spec_decode_num_accepted_tokens") or 0.0,
            _metric_value(m, "vllm:spec_decode_num_draft_tokens") or 0.0,
            _metric_value(m, "vllm:spec_decode_num_drafts") or 0.0,
        )

    result = {}
    snap0 = _snap() if not NOSPEC else None
    run_batch(SHORTLEN)          # warm CUDA graphs.
    lt, ln, outs = run_batch(OUTLEN)
    snap1 = _snap() if not NOSPEC else None
    if rank == 0:
        al = None
        if not NOSPEC and snap0 is not None and snap1 is not None:
            accepted = snap1[0] - snap0[0]
            drafted = snap1[1] - snap0[1]
            ndrafts = snap1[2] - snap0[2]
            al = (1 + accepted / ndrafts) if ndrafts else None
            result.update(num_accepted=accepted, num_drafted=drafted,
                          num_drafts=ndrafts)
        result.update(
            long_s=lt, long_ntok=ln,
            sys_tok_s=(ln / lt) if lt else None,
            accept_len=al,
        )
        if CAPTURE_TOKENS:
            result["token_ids"] = [list(o.outputs[0].token_ids) for o in outs]
        q.put(result)


def _read_profiles(profile_dir):
    files = sorted(glob.glob(os.path.join(profile_dir, "*.json")))
    parsed = []
    for f in files:
        try:
            with open(f) as fh:
                parsed.append((f, json.load(fh)))
        except Exception:
            pass

    def _nsamp(p):
        c = p.get("counts", {})
        return c.get("draft_chain", 0) or c.get("verify", 0)

    parsed.sort(key=lambda fp: _nsamp(fp[1]), reverse=True)
    best = parsed[0][1] if parsed else None
    return best, [p for _, p in parsed]


def main():
    from vllm.utils.network_utils import get_open_port

    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = f"{TAG}_b{BATCH}_K{K}_C{C}" + ("_nospec" if NOSPEC else "")
    profile_dir = os.path.join(OUT_DIR, f"tc_profiles_{suffix}")
    if os.path.isdir(profile_dir):
        for f in glob.glob(os.path.join(profile_dir, "*.json")):
            os.remove(f)
    os.makedirs(profile_dir, exist_ok=True)

    master_ip = "127.0.0.1"
    master_port = get_open_port()
    q = Queue()
    procs = []
    for rank in range(DP):
        p = Process(target=worker,
                    args=(rank, DP, TP, master_ip, master_port,
                          profile_dir, q))
        p.start()
        procs.append(p)
    res = None
    try:
        res = q.get(timeout=5400)
    except Exception:
        res = {"error": "no result from rank 0 (timeout)"}
    for p in procs:
        p.join()

    rank0_prof, all_prof = _read_profiles(profile_dir)
    out = {
        "model": MODEL, "tag": TAG, "dp": DP, "tp": TP,
        "batch": BATCH, "K": K, "C": C, "nospec": NOSPEC,
        "draft_quant": DRAFT_QUANT or "bf16",
        "gpu_mem": GPU_MEM, "outlen": OUTLEN, "warmup": WARMUP, "steady": STEADY,
        "run_result": {k: v for k, v in (res or {}).items() if k != "token_ids"},
        "rank0_profile": rank0_prof,
        "n_profile_files": len(all_prof),
    }
    if CAPTURE_TOKENS and isinstance(res, dict):
        out["token_ids"] = res.get("token_ids")
    path = os.path.join(OUT_DIR, f"tc_{suffix}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[TC] wrote {path}", flush=True)
    print(f"[TC] model={TAG} C={C} nospec={NOSPEC} batch={BATCH} K={K}",
          flush=True)
    if isinstance(res, dict) and res.get("accept_len") is not None:
        print(f"[TC] accept_len = {res['accept_len']:.4f} "
              f"sys_tok_s = {res.get('sys_tok_s'):.1f}", flush=True)
    if rank0_prof:
        s = rank0_prof.get("summary", {})
        for lbl in REGIONS:
            if lbl in s and s[lbl].get("mean_ms") is not None:
                d = s[lbl]
                print(f"[TC]   {lbl:20s} n={d['n']:4d} "
                      f"mean={d['mean_ms']:.4f}ms std={d['std_ms']:.4f}",
                      flush=True)


if __name__ == "__main__":
    main()
