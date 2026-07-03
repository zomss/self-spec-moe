"""W7-Qwen30B: headline decode tok/s -- fully-optimized World A self-spec stack
vs no-spec on Qwen3-30B-A3B at large EP, REAL forced-PCIe (no A2A emulation).

This is the headline test for the merged, fully-optimized World A stack:
  VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE=1   comm-free local-routing draft
  VLLM_SELF_SPEC_DRAFT_FULL_REPLICA=1  draft holds ALL experts (full coverage)
  VLLM_SELF_SPEC_DRAFT_FULL_CG=1       draft FULL cudagraphs w/ correct attn capture
  speculative_config["quantization"]="fp8"  on-the-fly FP8 draft (target stays bf16)

Layout: attention-DP + EP, tp=1, EP = DP across all GPUs (DP=N -> EP=N). Forced PCIe
via NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1 (set by the driver).
Steady-state DECODE throughput via the two-length slope (cancels prefill); CUDA
graphs ON, WARMUP>=2, ITERS>=3, same prompts+outlen for spec vs no-spec.

Unlike w7_fp8_timing.py this sets the FULL stack (FULL_CG) and uses REAL inter-GPU
A2A (no VLLM_SELF_SPEC_EMULATE_A2A_DELAY_US). One engine (one DP group) at a time.

Env knobs:
  W7_MODE        nospec | spec            (also argv[1])
  W7_MODEL       HF path / id             default Qwen3-30B-A3B local snapshot
  W7_DP          data-parallel size (=EP) default 8
  W7_TP          tensor-parallel size     default 1
  W7_DRAFT_QUANT draft quantization       default fp8 ("" -> bf16 draft)
  W7_BATCHES     comma list               default 8,32,64,128,256
  W7_KS          comma list (spec only)   default 2,3,4
  W7_OUTLEN      long output length       default 160
  W7_SHORTLEN    short output length      default 32
  W7_ITERS       timed iterations         default 3
  W7_WARMUP      warmup iterations        default 2
  W7_GPU_MEM     gpu_memory_utilization   default 0.90
  W7_EAGER       1 -> enforce_eager       default 0 (CUDA graphs ON)
  W7_MAX_MODEL_LEN  default 2048
  W7_FULL_CG     1 -> set DRAFT_FULL_CG   default 1
  W7_LOG_A2A     1 -> set LOG_A2A_COUNTS  default 0
  W7_OUT         output dir
  W7_TAG         filename tag
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
TP = int(os.environ.get("W7_TP", "1"))
DRAFT_QUANT = os.environ.get("W7_DRAFT_QUANT", "fp8").strip()
OUT_DIR = os.environ.get(
    "W7_OUT", "/data/smcho/self-spec-moe/research/34_worldA_system/data"
)
TAG = os.environ.get("W7_TAG", "qwen30b")
OUTLEN = int(os.environ.get("W7_OUTLEN", "160"))
SHORTLEN = int(os.environ.get("W7_SHORTLEN", "32"))
ITERS = int(os.environ.get("W7_ITERS", "3"))
WARMUP = int(os.environ.get("W7_WARMUP", "2"))
GPU_MEM = float(os.environ.get("W7_GPU_MEM", "0.90"))
MAX_MODEL_LEN = int(os.environ.get("W7_MAX_MODEL_LEN", "2048"))
BATCHES = [int(x) for x in os.environ.get("W7_BATCHES", "8,32,64,128,256").split(",")]
KS = [int(x) for x in os.environ.get("W7_KS", "2,3,4").split(",")]
EAGER = os.environ.get("W7_EAGER", "0") == "1"
FULL_CG = os.environ.get("W7_FULL_CG", "1") == "1"
FULL_REPLICA = os.environ.get("W7_FULL_REPLICA", "1") == "1"
LOG_A2A = os.environ.get("W7_LOG_A2A", "0") == "1"

BASE_PROMPT = (
    "The history of artificial intelligence began in antiquity with myths and "
    "stories, and the modern field was founded in"
)


def _mean_std(xs):
    n = len(xs)
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


def worker(rank, dp, tp, master_ip, master_port, mode, k, q):
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(dp)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)
    if LOG_A2A:
        os.environ["VLLM_SELF_SPEC_LOG_A2A_COUNTS"] = "1"

    spec = mode == "spec"
    if spec:
        os.environ["VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE"] = "1"
        os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = "0"
        if FULL_REPLICA:
            os.environ["VLLM_SELF_SPEC_DRAFT_FULL_REPLICA"] = "1"
        if FULL_CG:
            os.environ["VLLM_SELF_SPEC_DRAFT_FULL_CG"] = "1"

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=MODEL,
        tensor_parallel_size=tp,
        enable_expert_parallel=True,
        trust_remote_code=False,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM,
        enforce_eager=EAGER,
        disable_log_stats=False,
    )
    if spec:
        spec_cfg = {
            "method": "draft_model",
            "model": MODEL,
            "num_speculative_tokens": k,
            "draft_tensor_parallel_size": tp,
        }
        if DRAFT_QUANT:
            spec_cfg["quantization"] = DRAFT_QUANT
        kwargs["speculative_config"] = spec_cfg
    llm = LLM(**kwargs)

    def run_batch(batch, out_len):
        prompts = [f"{BASE_PROMPT} the year {1900 + i}." for i in range(batch)]
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

    results = []
    for batch in BATCHES:
        try:
            for _ in range(WARMUP):
                run_batch(batch, OUTLEN)
                run_batch(batch, SHORTLEN)

            long_times, short_times, full_ntoks = [], [], []
            accepted = drafted = ndrafts = None
            snap0 = _snap()
            for _ in range(ITERS):
                lt, ln = run_batch(batch, OUTLEN)
                st, _ = run_batch(batch, SHORTLEN)
                long_times.append(lt)
                short_times.append(st)
                full_ntoks.append(ln)
            snap1 = _snap()
            if snap0 is not None:
                accepted = snap1[0] - snap0[0]
                drafted = snap1[1] - snap0[1]
                ndrafts = snap1[2] - snap0[2]

            decode_times = [lt - st for lt, st in zip(long_times, short_times)]
            out_decode = batch * (OUTLEN - SHORTLEN)
            toks = [out_decode / dt for dt in decode_times]
            dt_m, dt_s = _mean_std(decode_times)
            tps_m, tps_s = _mean_std(toks)
            lt_m, _ = _mean_std(long_times)
            st_m, _ = _mean_std(short_times)
            suspect = bool(
                (st_m > 0 and dt_m < 0.15 * st_m)
                or (tps_m > 0 and tps_s / tps_m > 0.15)
            )
            row = {
                "batch": batch, "out_len": OUTLEN, "short_len": SHORTLEN,
                "iters": ITERS, "warmup": WARMUP,
                "long_s_mean": lt_m, "short_s_mean": st_m,
                "long_s_all": long_times, "short_s_all": short_times,
                "decode_s_mean": dt_m, "decode_s_std": dt_s,
                "tok_s_mean": tps_m, "tok_s_std": tps_s,
                "out_tokens_decode": out_decode,
                "full_ntoks": full_ntoks[-1],
                "suspect": suspect,
            }
            if spec and rank == 0:
                al = (1 + accepted / ndrafts) if (accepted and ndrafts) else None
                per_tok = (accepted / drafted) if (accepted and drafted) else None
                row["accept_len"] = al
                row["accept_rate_per_tok"] = per_tok
                row["num_accepted"] = accepted
                row["num_drafted"] = drafted
                row["num_drafts"] = ndrafts
            results.append(row)
            if rank == 0:
                tag = f"K={k}" if spec else "nospec"
                al_s = (
                    f" accept_len={row.get('accept_len'):.3f}"
                    if spec and row.get("accept_len") else ""
                )
                print(f"[W7Q dp={dp} {tag}] batch={batch} decode={dt_m:.3f}s "
                      f"tok/s={tps_m:.1f}+-{tps_s:.1f}{al_s}", flush=True)
        except Exception as e:
            if rank == 0:
                results.append({"batch": batch, "error": repr(e)[:300]})
                print(f"[W7Q dp={dp}] batch={batch} FAILED: {repr(e)[:200]}",
                      flush=True)
            break

    if rank == 0:
        q.put(results)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("W7_MODE", "nospec")
    assert mode in ("nospec", "spec")
    ks = KS if mode == "spec" else [0]

    from vllm.utils.network_utils import get_open_port

    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = ("eager" if EAGER else "cg") + (
        "_fullcg" if (FULL_CG and mode == "spec") else ""
    )
    all_out = {"mode": mode, "model": MODEL, "tag": TAG, "dp": DP, "ep": DP, "tp": TP,
               "draft_quant": DRAFT_QUANT or "bf16",
               "full_cg": FULL_CG, "real_pcie": True,
               "out_len": OUTLEN, "short_len": SHORTLEN, "iters": ITERS,
               "warmup": WARMUP, "gpu_mem": GPU_MEM, "eager": EAGER,
               "suffix": suffix, "by_k": {}}

    for k in ks:
        master_ip = "127.0.0.1"
        # retry get_open_port on collision
        master_port = None
        for _ in range(10):
            p = get_open_port()
            if p != master_port:
                master_port = p
                break
        q = Queue()
        procs = []
        for rank in range(DP):
            p = Process(target=worker,
                        args=(rank, DP, TP, master_ip, master_port, mode, k, q))
            p.start()
            procs.append(p)
        res = None
        try:
            res = q.get(timeout=7200)
        except Exception:
            res = [{"error": "no result from rank 0 (timeout)"}]
        for p in procs:
            p.join()
        all_out["by_k"][str(k)] = res
        ktag = f"K{k}" if mode == "spec" else "nospec"
        path = os.path.join(
            OUT_DIR, f"w7q_{TAG}_dp{DP}_{mode}_{suffix}_{ktag}.json")
        with open(path, "w") as f:
            json.dump({**all_out, "this_k": k, "results": res}, f, indent=2)
        print(f"[W7Q] wrote {path}", flush=True)

    path = os.path.join(OUT_DIR, f"w7q_{TAG}_dp{DP}_{mode}_{suffix}.json")
    with open(path, "w") as f:
        json.dump(all_out, f, indent=2)
    print(f"[W7Q] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
