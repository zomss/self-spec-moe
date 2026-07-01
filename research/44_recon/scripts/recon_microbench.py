"""Phase 44 recon: decompose the comm-free self-spec cycle and isolate draft cost.

Reuses the W7 launch setup (Qwen3-30B-A3B, DP=8 + EP, forced-PCIe, FP8
full-replica comm-free draft) and the env-gated profiler
(vllm/v1/spec_decode/self_spec_profiler.py) to record CUDA-synced per-region
wall-clock timings of the spec decode cycle. Three run modes:

  spec     : FP8 full-replica comm-free draft (DRAFT_FULL_REPLICA=1,
             DRAFT_LOCAL_ROUTE=1). Profiles draft_forward_first / draft_forward
             / draft_chain / verify. THE real cycle (measurement 1) and the real
             full-replica draft forward (measurement 2).
  nospec   : plain full-EP decode, no spec. Profiles the verify-region forward
             over 1 tok/seq -> the no-spec/verify compute reference (~34ms).
  skiptile : spec with an EP-SHARD draft + VLLM_SELF_SPEC_SKIP_A2A=1 and
             DRAFT_LOCAL_ROUTE=0, so the draft's EP MoE hits the local-chunk
             tile branch (garbage output, timing-only) -- the exact comm-free
             *tile* the cost model's T_compute measured. Its draft_forward is
             the T_compute stand-in at the decode shape.

Everything else (two-length no-spec slope, profiler dump/read) mirrors
research/34_worldA_system/scripts/w7_microbench.py. NO vllm/ changes.

Env knobs (RC_*):
  RC_MODE      spec | nospec | skiptile      (also argv[1]); default spec
  RC_MODEL     HF path / id
  RC_DP        data-parallel size            default 8
  RC_TP        default 1
  RC_TRC       1 -> trust_remote_code        default 0
  RC_BATCH     global batch (== n sequences) default 64
  RC_K         num_speculative_tokens        default 2 (spec/skiptile)
  RC_DRAFT_QUANT  draft quant (fp8) ; "" -> bf16 draft
  RC_GPU_MEM   default 0.90
  RC_MAX_MODEL_LEN default 2048
  RC_STEADY    steady decode steps after warmup; default 80
  RC_WARMUP    profiler warmup samples to drop; default 60
  RC_OUT       results dir
  RC_TAG       filename tag
"""
import glob
import json
import os
import sys
import time
from multiprocessing import Process, Queue

MODEL = os.environ["RC_MODEL"]
DP = int(os.environ.get("RC_DP", "8"))
TP = int(os.environ.get("RC_TP", "1"))
TRC = os.environ.get("RC_TRC", "0") == "1"
BATCH = int(os.environ.get("RC_BATCH", "64"))
K = int(os.environ.get("RC_K", "2"))
DRAFT_QUANT = os.environ.get("RC_DRAFT_QUANT", "").strip()
GPU_MEM = float(os.environ.get("RC_GPU_MEM", "0.90"))
MAX_MODEL_LEN = int(os.environ.get("RC_MAX_MODEL_LEN", "2048"))
STEADY = int(os.environ.get("RC_STEADY", "80"))
WARMUP = int(os.environ.get("RC_WARMUP", "60"))
OUT_DIR = os.environ.get("RC_OUT", os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))) + "/data")
TAG = os.environ.get("RC_TAG", "qwen30b")

BASE_PROMPT = (
    "The history of artificial intelligence began in antiquity with myths and "
    "stories, and the modern field was founded in"
)

# Drive enough output tokens to clear WARMUP + STEADY draft_chain samples.
OUTLEN = WARMUP + STEADY + 80
SHORTLEN = 32  # for the no-spec two-length slope only.

REGIONS = ("draft_forward_first", "draft_forward", "draft_chain", "verify")


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


def worker(rank, dp, tp, master_ip, master_port, mode, profile_dir, q):
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(dp)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

    os.environ["VLLM_SELF_SPEC_PROFILE"] = "1"
    os.environ["VLLM_SELF_SPEC_PROFILE_OUT"] = profile_dir
    os.environ["VLLM_SELF_SPEC_PROFILE_WARMUP"] = str(WARMUP)

    spec = mode in ("spec", "skiptile")
    if spec:
        if mode == "spec":
            # Genuine comm-free full-replica draft (all experts resident, local
            # route -> no collective).
            os.environ["VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE"] = "1"
            os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = "0"
            os.environ["VLLM_SELF_SPEC_DRAFT_FULL_REPLICA"] = "1"
        else:  # skiptile: the T_compute tile (EP-shard draft, skip-A2A branch)
            os.environ["VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE"] = "0"
            os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = "0"
            os.environ["VLLM_SELF_SPEC_DRAFT_FULL_REPLICA"] = "0"
            os.environ["VLLM_SELF_SPEC_SKIP_A2A"] = "1"

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=MODEL,
        tensor_parallel_size=tp,
        enable_expert_parallel=True,
        trust_remote_code=TRC,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM,
        enforce_eager=False,  # CUDA graphs ON.
        disable_log_stats=False,
    )
    if spec:
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
        return dt, ntok

    def _snap():
        if not (spec and rank == 0):
            return None
        m = llm.get_metrics()
        return (
            _metric_value(m, "vllm:spec_decode_num_accepted_tokens") or 0.0,
            _metric_value(m, "vllm:spec_decode_num_draft_tokens") or 0.0,
            _metric_value(m, "vllm:spec_decode_num_drafts") or 0.0,
        )

    result = {}
    if spec:
        snap0 = _snap()
        run_batch(SHORTLEN)          # warm CUDA graphs before the timed run.
        lt, ln = run_batch(OUTLEN)   # long run -> profiler steady samples.
        snap1 = _snap()
        if rank == 0:
            accepted = snap1[0] - snap0[0]
            drafted = snap1[1] - snap0[1]
            ndrafts = snap1[2] - snap0[2]
            al = (1 + accepted / ndrafts) if ndrafts else None
            # cycle-derived tok/s over the long run (whole batch): total output
            # tokens / long time -> mean tokens/sec (system).
            result = {
                "long_s": lt, "long_ntok": ln,
                "sys_tok_s": (ln / lt) if lt else None,
                "accept_len": al, "num_accepted": accepted,
                "num_drafted": drafted, "num_drafts": ndrafts,
            }
    else:
        ITERS = 3
        for _ in range(2):
            run_batch(OUTLEN)
            run_batch(SHORTLEN)
        long_times, short_times = [], []
        for _ in range(ITERS):
            lt, _ = run_batch(OUTLEN)
            st, _ = run_batch(SHORTLEN)
            long_times.append(lt)
            short_times.append(st)
        decode_times = [lt - st for lt, st in zip(long_times, short_times)]
        dt_m, dt_s = _mean_std(decode_times)
        steps = OUTLEN - SHORTLEN
        nospec_step_ms = (dt_m / steps) * 1e3
        if rank == 0:
            result = {
                "decode_s_mean": dt_m, "decode_s_std": dt_s,
                "steps_per_seq": steps,
                "nospec_step_ms": nospec_step_ms,
                "nospec_sys_tok_s": (BATCH * steps / dt_m) if dt_m else None,
            }

    if rank == 0:
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
    mode = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RC_MODE", "spec")
    assert mode in ("spec", "nospec", "skiptile")

    from vllm.utils.network_utils import get_open_port

    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = f"{TAG}_b{BATCH}_K{K}_{mode}"
    profile_dir = os.path.join(OUT_DIR, f"rc_profiles_{suffix}")
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
                    args=(rank, DP, TP, master_ip, master_port, mode,
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
        "mode": mode, "model": MODEL, "tag": TAG, "dp": DP, "tp": TP,
        "batch": BATCH, "K": K, "draft_quant": DRAFT_QUANT or "bf16",
        "gpu_mem": GPU_MEM, "outlen": OUTLEN, "warmup": WARMUP, "steady": STEADY,
        "run_result": res, "rank0_profile": rank0_prof,
        "n_profile_files": len(all_prof),
    }
    path = os.path.join(OUT_DIR, f"rc_{suffix}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[RC] wrote {path}", flush=True)

    print(f"[RC] mode={mode} batch={BATCH} K={K} dp={DP}", flush=True)
    if isinstance(res, dict):
        if res.get("nospec_step_ms") is not None:
            print(f"[RC] nospec_step_ms = {res['nospec_step_ms']:.3f} "
                  f"(sys tok/s {res.get('nospec_sys_tok_s'):.1f})", flush=True)
        if res.get("accept_len") is not None:
            print(f"[RC] accept_len = {res['accept_len']:.3f} "
                  f"sys_tok_s = {res.get('sys_tok_s'):.1f}", flush=True)
    if rank0_prof:
        s = rank0_prof.get("summary", {})
        for lbl in REGIONS:
            if lbl in s and s[lbl].get("mean_ms") is not None:
                d = s[lbl]
                print(f"[RC]   {lbl:20s} n={d['n']:4d} "
                      f"mean={d['mean_ms']:.3f}ms std={d['std_ms']:.3f} "
                      f"min={d['min_ms']:.3f}", flush=True)


if __name__ == "__main__":
    main()
