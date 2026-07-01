"""Phase 45 — fine-grained CPU-orchestration profile of the comm-free self-spec
decode cycle, and A/B validation of the CPU-reduction knobs.

Reuses the recon (phase 44) launch setup (DP=8 + EP, forced-PCIe, FP8
full-replica comm-free draft, K=2, greedy) and the env-gated profiler
(vllm/v1/spec_decode/self_spec_profiler.py) with FINE regions ON so the outer
CPU orchestration (rejection parse / bookkeeping / next-input build) is
attributed alongside draft_chain / verify.

Two jobs:
  profile : run one spec engine, dump the fine per-region CPU breakdown +
            accept_len + sys tok/s. Set CP_KNOBS to toggle the reduction envs.
  accept  : run a spec engine and print per-request output token ids +
            accept_len, so a before/after diff proves losslessness (greedy,
            fixed seed). Writes the token ids to a JSON for exact comparison.

Env knobs (CP_*):
  CP_JOB       profile | accept                       default profile
  CP_MODEL     HF path / id                           default Qwen1.5-MoE-A2.7B
  CP_DP        data-parallel size                     default 8
  CP_TP                                                default 1
  CP_TRC       1 -> trust_remote_code                 default 0
  CP_BATCH     global batch (== n sequences)          default 64
  CP_K         num_speculative_tokens                 default 2
  CP_DRAFT_QUANT  draft quant (fp8) ; "" -> bf16       default fp8
  CP_GPU_MEM                                          default 0.90
  CP_MAX_MODEL_LEN                                     default 2048
  CP_STEADY    steady decode steps after warmup       default 80
  CP_WARMUP    profiler warmup samples to drop        default 40
  CP_OUT       results dir                            default data/
  CP_TAG       filename tag                           default qwen15moe
  CP_KNOBS     comma list of extra env=val to set in each worker
               e.g. "VLLM_SELF_SPEC_FUSE_BOOKKEEP=1"
  CP_FINE      1 -> profiler FINE regions on          default 1
"""
import glob
import json
import os
import sys
import time
from multiprocessing import Process, Queue

MODEL = os.environ.get("CP_MODEL", "Qwen/Qwen1.5-MoE-A2.7B")
DP = int(os.environ.get("CP_DP", "8"))
TP = int(os.environ.get("CP_TP", "1"))
TRC = os.environ.get("CP_TRC", "0") == "1"
BATCH = int(os.environ.get("CP_BATCH", "64"))
K = int(os.environ.get("CP_K", "2"))
DRAFT_QUANT = os.environ.get("CP_DRAFT_QUANT", "fp8").strip()
GPU_MEM = float(os.environ.get("CP_GPU_MEM", "0.90"))
MAX_MODEL_LEN = int(os.environ.get("CP_MAX_MODEL_LEN", "2048"))
STEADY = int(os.environ.get("CP_STEADY", "80"))
WARMUP = int(os.environ.get("CP_WARMUP", "40"))
OUT_DIR = os.environ.get("CP_OUT", os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))) + "/data")
TAG = os.environ.get("CP_TAG", "qwen15moe")
JOB = os.environ.get("CP_JOB", "profile")
KNOBS = [kv for kv in os.environ.get("CP_KNOBS", "").split(",") if kv.strip()]
FINE = os.environ.get("CP_FINE", "1")

BASE_PROMPT = (
    "The history of artificial intelligence began in antiquity with myths and "
    "stories, and the modern field was founded in"
)

OUTLEN = WARMUP + STEADY + 40

# Fine outer-CPU regions we instrument in vllm/ (plus the coarse ones).
REGIONS = (
    "draft_chain", "verify",
    "draft_forward_first", "draft_forward",
    "cpu_rejection_sample", "cpu_reject_parse", "cpu_bookkeep_loop",
    "cpu_next_input_build", "cpu_prepare_inputs_padded", "cpu_exec_prepare_inputs",
    "step_pos_slot_update", "step_build_attn_md", "step_input_buffering",
    "step_sample", "chain_setup",
    "step0_set_inputs", "step0_build_attn_md", "step0_determine_batch",
    "step0_sample",
)


def _mean_std(xs):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, var ** 0.5


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

    # The accept (losslessness) job only needs deterministic token output; the
    # FINE profiler adds many CUDA syncs that make it needlessly slow, so
    # profile only for the profile job.
    if JOB == "profile":
        os.environ["VLLM_SELF_SPEC_PROFILE"] = "1"
        os.environ["VLLM_SELF_SPEC_PROFILE_OUT"] = profile_dir
        os.environ["VLLM_SELF_SPEC_PROFILE_WARMUP"] = str(WARMUP)
        os.environ["VLLM_SELF_SPEC_PROFILE_FINE"] = FINE

    # Genuine comm-free full-replica draft.
    os.environ["VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE"] = "1"
    os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = "0"
    os.environ["VLLM_SELF_SPEC_DRAFT_FULL_REPLICA"] = "1"
    os.environ["VLLM_SELF_SPEC_DRAFT_FULL_CG"] = "1"
    os.environ["VLLM_SELF_SPEC_COMPILE_CONSISTENT"] = "1"

    # A/B reduction knobs (default off in the binary -> baseline unless set).
    for kv in KNOBS:
        k, _, v = kv.partition("=")
        os.environ[k.strip()] = v.strip()

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
        sp = SamplingParams(temperature=0.0, max_tokens=out_len,
                            ignore_eos=True, seed=0)
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        return dt, outs

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
    snap0 = _snap()
    run_batch(32)                   # warm CUDA graphs.
    lt, outs = run_batch(OUTLEN)    # long run -> profiler steady samples.
    snap1 = _snap()
    if rank == 0:
        accepted = snap1[0] - snap0[0]
        ndrafts = snap1[2] - snap0[2]
        al = (1 + accepted / ndrafts) if ndrafts else None
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        result = {
            "long_s": lt, "long_ntok": ntok,
            "sys_tok_s": (ntok / lt) if lt else None,
            "accept_len": al, "num_accepted": accepted, "num_drafts": ndrafts,
        }
        if JOB == "accept":
            # Deterministic token ids per request for a losslessness diff.
            result["token_ids"] = [list(o.outputs[0].token_ids) for o in outs]
        q.put(result)


def _read_profiles(profile_dir):
    files = sorted(glob.glob(os.path.join(profile_dir, "*.json")))
    parsed = []
    for f in files:
        try:
            with open(f) as fh:
                parsed.append(json.load(fh))
        except Exception:
            pass
    parsed.sort(key=lambda p: (p.get("counts", {}).get("draft_chain", 0)
                               or p.get("counts", {}).get("verify", 0)),
                reverse=True)
    return parsed[0] if parsed else None


def main():
    from vllm.utils.network_utils import get_open_port

    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = f"{TAG}_b{BATCH}_K{K}_{JOB}"
    if KNOBS:
        suffix += "_" + "_".join(kv.split("=")[0].replace(
            "VLLM_SELF_SPEC_", "").lower() for kv in KNOBS)
    profile_dir = os.path.join(OUT_DIR, f"cp_profiles_{suffix}")
    if os.path.isdir(profile_dir):
        for f in glob.glob(os.path.join(profile_dir, "*.json")):
            os.remove(f)
    os.makedirs(profile_dir, exist_ok=True)

    master_ip = "127.0.0.1"
    master_port = get_open_port()
    q = Queue()
    procs = []
    for rank in range(DP):
        p = Process(target=worker, args=(rank, DP, TP, master_ip,
                                         master_port, profile_dir, q))
        p.start()
        procs.append(p)
    try:
        res = q.get(timeout=5400)
    except Exception:
        res = {"error": "no result from rank 0 (timeout)"}
    for p in procs:
        p.join()

    prof = _read_profiles(profile_dir)
    out = {
        "job": JOB, "model": MODEL, "tag": TAG, "dp": DP, "batch": BATCH,
        "K": K, "draft_quant": DRAFT_QUANT or "bf16", "knobs": KNOBS,
        "outlen": OUTLEN, "warmup": WARMUP, "steady": STEADY,
        "run_result": {k: v for k, v in (res or {}).items()
                       if k != "token_ids"},
        "rank0_profile": prof,
    }
    if JOB == "accept" and isinstance(res, dict):
        out["token_ids"] = res.get("token_ids")
    path = os.path.join(OUT_DIR, f"cp_{suffix}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[CP] wrote {path}", flush=True)

    if isinstance(res, dict):
        print(f"[CP] job={JOB} batch={BATCH} K={K} knobs={KNOBS}", flush=True)
        if res.get("accept_len") is not None:
            print(f"[CP] accept_len = {res['accept_len']:.4f}  "
                  f"sys_tok_s = {res.get('sys_tok_s'):.1f}", flush=True)
    if prof:
        s = prof.get("summary", {})
        print(f"[CP] --- fine CPU regions (rank0, warmup={prof.get('warmup')}) ---",
              flush=True)
        for lbl in REGIONS:
            if lbl in s and s[lbl].get("mean_ms") is not None:
                d = s[lbl]
                print(f"[CP]   {lbl:26s} n={d['n']:4d} "
                      f"mean={d['mean_ms']:.4f}ms std={d['std_ms']:.4f} "
                      f"min={d['min_ms']:.4f}", flush=True)


if __name__ == "__main__":
    main()
