"""W7 micro-bench: decompose the spec decode cycle into FIXABLE vs FUNDAMENTAL.

Reuses the W7 model/launch setup (DeepSeek-V2-Lite, DP=2 + EP, forced-PCIe +
emulated A2A, bf16 full-replica comm-free draft) but instead of throughput it
records CUDA-synchronized wall-clock timings of three regions per decode step
(env-gated by VLLM_SELF_SPEC_PROFILE, implemented in
vllm/v1/spec_decode/self_spec_profiler.py):

  draft_forward_first : the step-0 draft forward inside propose().
  draft_forward       : a single decode-step draft forward inside the K-chain.
  draft_chain         : the whole propose() (the K-step chain + CPU orchestration).
  verify              : the target verify forward over the K+1 proposed tokens.

The engine-core process (where these accumulate) dumps a per-PID JSON summary
to VLLM_SELF_SPEC_PROFILE_OUT on exit; this harness reads them after the engine
shuts down and reports rank-0 (the file with the most draft_chain samples).

For t_nospec_step we run the no-spec engine separately and take the per-step
wall time via the W7 two-length slope (full decode step incl. sample/bookkeep),
plus the no-spec verify-region forward for context.

Env knobs:
  MB_MODE      spec | nospec            (also argv[1]); default spec
  MB_MODEL     HF path / id             default DeepSeek-V2-Lite local snapshot
  MB_DP        data-parallel size       default 2
  MB_TP        tensor-parallel size     default 1
  MB_TRC       1 -> trust_remote_code   default 0
  MB_BATCH     batch size               default 64
  MB_K         num_speculative_tokens   default 2  (spec only)
  MB_A2A_US    emulated exposed A2A (us) default 100
  MB_GPU_MEM   gpu_memory_utilization   default 0.90
  MB_MAX_MODEL_LEN                       default 2048
  MB_STEADY    steady-state decode steps to drive after warmup; default 60
  MB_WARMUP    decode-step warmup to drop in the profiler summary; default 50
  MB_OUT       results dir              default research/34_worldA_system/data
  MB_TAG       filename tag             default v2lite
"""
import glob
import json
import os
import sys
import time
from multiprocessing import Process, Queue

MODEL = os.environ.get(
    "MB_MODEL",
    "/home/smcho/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite"
    "/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0",
)
DP = int(os.environ.get("MB_DP", "2"))
TP = int(os.environ.get("MB_TP", "1"))
TRC = os.environ.get("MB_TRC", "0") == "1"
BATCH = int(os.environ.get("MB_BATCH", "64"))
K = int(os.environ.get("MB_K", "2"))
A2A_US = float(os.environ.get("MB_A2A_US", "100"))
GPU_MEM = float(os.environ.get("MB_GPU_MEM", "0.90"))
MAX_MODEL_LEN = int(os.environ.get("MB_MAX_MODEL_LEN", "2048"))
STEADY = int(os.environ.get("MB_STEADY", "60"))
WARMUP = int(os.environ.get("MB_WARMUP", "50"))
OUT_DIR = os.environ.get(
    "MB_OUT", "/data/smcho/ssm-mb/research/34_worldA_system/data"
)
TAG = os.environ.get("MB_TAG", "v2lite")

BASE_PROMPT = (
    "The history of artificial intelligence began in antiquity with myths and "
    "stories, and the modern field was founded in"
)

# Drive enough output tokens to clear WARMUP draft-chain samples + STEADY more.
# Each accepted-or-rejected decode step yields one draft_chain sample. We
# overshoot to be safe.
OUTLEN = WARMUP + STEADY + 60
SHORTLEN = 32  # for the no-spec two-length slope only.


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

    # Profiler: enabled in all ranks; each engine-core process dumps its own
    # per-PID file into profile_dir on exit.
    os.environ["VLLM_SELF_SPEC_PROFILE"] = "1"
    os.environ["VLLM_SELF_SPEC_PROFILE_OUT"] = profile_dir
    os.environ["VLLM_SELF_SPEC_PROFILE_WARMUP"] = str(WARMUP)

    spec = mode == "spec"
    if A2A_US > 0:
        os.environ["VLLM_SELF_SPEC_EMULATE_A2A_DELAY_US"] = str(A2A_US)
    if spec:
        os.environ["VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE"] = "1"
        os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = "0"
        os.environ["VLLM_SELF_SPEC_DRAFT_FULL_REPLICA"] = os.environ.get(
            "MB_FULL_REPLICA", "1"
        )

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=MODEL,
        tensor_parallel_size=tp,
        enable_expert_parallel=True,
        trust_remote_code=TRC,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM,
        enforce_eager=False,  # CUDA graphs ON (realistic config).
        disable_log_stats=False,
    )
    if spec:
        kwargs["speculative_config"] = {
            "method": "draft_model",
            "model": MODEL,
            "num_speculative_tokens": K,
            "draft_tensor_parallel_size": tp,
        }
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
        # One long run drives WARMUP+STEADY+ decode steps; the profiler captures
        # per-step region timings and dumps them at engine shutdown.
        snap0 = _snap()
        # A short prior run to ensure CUDA graphs are captured before the timed
        # long run begins (graph capture must not leak into steady samples).
        run_batch(SHORTLEN)
        lt, ln = run_batch(OUTLEN)
        snap1 = _snap()
        if rank == 0:
            accepted = snap1[0] - snap0[0]
            ndrafts = snap1[2] - snap0[2]
            al = (1 + accepted / ndrafts) if ndrafts else None
            result = {
                "long_s": lt, "long_ntok": ln,
                "accept_len": al, "num_accepted": accepted,
                "num_drafts": ndrafts,
            }
    else:
        # No-spec: two-length slope -> full per-decode-step wall time.
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
        # Per-step wall time for the whole BATCH (system step), and per-step
        # *system throughput* basis. t_nospec_step (per system decode step):
        nospec_step_ms = (dt_m / steps) * 1e3
        if rank == 0:
            result = {
                "decode_s_mean": dt_m, "decode_s_std": dt_s,
                "steps_per_seq": steps,
                "nospec_step_ms": nospec_step_ms,
            }

    if rank == 0:
        q.put(result)


def _read_profiles(profile_dir):
    """Return (rank0_summary, all_files) from the dumped per-PID JSONs.

    Rank 0 == the file with the most draft_chain (or verify) samples, i.e. the
    process that actually ran the model decode loop for the timed long run.
    """
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
    mode = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MB_MODE", "spec")
    assert mode in ("spec", "nospec")

    from vllm.utils.network_utils import get_open_port

    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = f"{TAG}_b{BATCH}_K{K}_a2a{int(A2A_US)}us"
    profile_dir = os.path.join(OUT_DIR, f"mb_profiles_{mode}_{suffix}")
    # Clean stale dumps so we only read this run's files.
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
        res = q.get(timeout=3600)
    except Exception:
        res = {"error": "no result from rank 0 (timeout)"}
    for p in procs:
        p.join()

    rank0_prof, all_prof = _read_profiles(profile_dir)

    out = {
        "mode": mode, "model": MODEL, "tag": TAG, "dp": DP, "tp": TP,
        "batch": BATCH, "K": K, "a2a_us": A2A_US, "gpu_mem": GPU_MEM,
        "outlen": OUTLEN, "warmup": WARMUP, "steady": STEADY,
        "run_result": res,
        "rank0_profile": rank0_prof,
        "n_profile_files": len(all_prof),
    }
    path = os.path.join(OUT_DIR, f"mb_{mode}_{suffix}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[MB] wrote {path}", flush=True)

    # Concise console summary.
    print(f"[MB] mode={mode} batch={BATCH} K={K} a2a={A2A_US}us", flush=True)
    if mode == "nospec" and isinstance(res, dict):
        print(f"[MB] nospec_step_ms (full step) = "
              f"{res.get('nospec_step_ms')}", flush=True)
    if isinstance(res, dict) and res.get("accept_len") is not None:
        print(f"[MB] accept_len = {res['accept_len']:.3f}", flush=True)
    if rank0_prof:
        s = rank0_prof.get("summary", {})
        for lbl in ("draft_forward_first", "draft_forward", "draft_chain",
                    "verify"):
            if lbl in s and s[lbl].get("mean_ms") is not None:
                d = s[lbl]
                print(f"[MB]   {lbl:20s} n={d['n']:4d} "
                      f"mean={d['mean_ms']:.3f}ms std={d['std_ms']:.3f}",
                      flush=True)


if __name__ == "__main__":
    main()
