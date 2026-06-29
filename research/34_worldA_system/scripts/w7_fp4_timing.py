"""W7-FP4: wall-clock decode tokens/s -- World A NVFP4 full-replica draft vs no-spec.

Stage-2 (FP4) re-test of results_W7.md / results_W7_fp8.md. Same two-length-slope
steady-state DECODE methodology and the same forced-PCIe DP+EP layout, but the
DRAFT is a pre-quantized NVFP4 checkpoint (W4A16 dequant via Marlin on sm_90)
while the TARGET stays bf16.

The draft quantization is auto-detected from the NVFP4 checkpoint's embedded
hf_quant_config.json (modelopt_fp4): we pass model=<nvfp4 ckpt> in the spec_cfg
and DO NOT set spec_cfg["quantization"], so the draft ModelConfig reads the
checkpoint's quant config -> the merged draft-quant wiring in
draft_model.py::_create_draft_vllm_config derives ModelOptNvFp4Config for the
draft. Target loads the plain bf16 checkpoint.

Two configs (selected by argv[1] or W7_MODE):
  nospec : plain full-EP decode, no speculative config (bf16 target).
  spec   : World A draft_model spec, VLLM_SELF_SPEC_DRAFT_FULL_REPLICA=1
           (full-replica comm-free draft), producer ON, draft = NVFP4 ckpt.

Decode-time isolation (two-length slope): decode_time = t(OUTLEN) - t(SHORTLEN)
cancels prefill; tok/s = batch*(OUTLEN-SHORTLEN)/decode_time (OUTPUT tokens).

Env knobs (same as w7_fp8_timing.py plus W7_DRAFT_MODEL):
  W7_MODE        nospec | spec            (also argv[1])
  W7_MODEL       target HF path / id      default DeepSeek-V2-Lite bf16 snapshot
  W7_DRAFT_MODEL draft HF path (NVFP4)    default dsv2lite-nvfp4 ckpt
  W7_DP / W7_TP / W7_TRC / W7_BATCHES / W7_KS / W7_OUTLEN / W7_SHORTLEN /
  W7_ITERS / W7_WARMUP / W7_GPU_MEM / W7_EAGER / W7_A2A_US / W7_MAX_MODEL_LEN /
  W7_OUT / W7_TAG  -- as in w7_fp8_timing.py.
"""
import json
import os
import sys
import time
from multiprocessing import Process, Queue

MODEL = os.environ.get(
    "W7_MODEL",
    "/home/smcho/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite"
    "/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0",
)
DRAFT_MODEL = os.environ.get(
    "W7_DRAFT_MODEL",
    "/data/smcho/ssm-w7fp4/research/34_worldA_system/ckpts/dsv2lite-nvfp4",
)
DP = int(os.environ.get("W7_DP", "2"))
TP = int(os.environ.get("W7_TP", "1"))
TRC = os.environ.get("W7_TRC", "0") == "1"
OUT_DIR = os.environ.get(
    "W7_OUT", "/data/smcho/ssm-w7fp4/research/34_worldA_system/data"
)
TAG = os.environ.get("W7_TAG", "fp4")
OUTLEN = int(os.environ.get("W7_OUTLEN", "160"))
SHORTLEN = int(os.environ.get("W7_SHORTLEN", "32"))
ITERS = int(os.environ.get("W7_ITERS", "3"))
WARMUP = int(os.environ.get("W7_WARMUP", "2"))
GPU_MEM = float(os.environ.get("W7_GPU_MEM", "0.90"))
MAX_MODEL_LEN = int(os.environ.get("W7_MAX_MODEL_LEN", "2048"))
BATCHES = [int(x) for x in os.environ.get("W7_BATCHES", "8,32,64,128,256").split(",")]
KS = [int(x) for x in os.environ.get("W7_KS", "2,3,4").split(",")]
EAGER = os.environ.get("W7_EAGER", "0") == "1"
A2A_US = float(os.environ.get("W7_A2A_US", "0"))

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

    spec = mode == "spec"
    if A2A_US > 0:
        os.environ["VLLM_SELF_SPEC_EMULATE_A2A_DELAY_US"] = str(A2A_US)
    if spec:
        os.environ["VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE"] = "1"
        os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = "0"
        os.environ["VLLM_SELF_SPEC_DRAFT_FULL_REPLICA"] = "1"

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=MODEL,
        tensor_parallel_size=tp,
        enable_expert_parallel=True,
        trust_remote_code=TRC,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM,
        enforce_eager=EAGER,
        disable_log_stats=False,
    )
    if spec:
        # NVFP4 draft: a separate pre-quantized checkpoint; quantization is
        # auto-detected from its hf_quant_config.json (modelopt_fp4). Do NOT set
        # spec_cfg["quantization"] (that is the on-the-fly path for bf16 ckpts).
        spec_cfg = {
            "method": "draft_model",
            "model": DRAFT_MODEL,
            "num_speculative_tokens": k,
            "draft_tensor_parallel_size": tp,
        }
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
                row["accept_len"] = al
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
                print(f"[W7FP4 {tag}] batch={batch} decode={dt_m:.3f}s "
                      f"tok/s={tps_m:.1f}+-{tps_s:.1f}{al_s}", flush=True)
        except Exception as e:
            if rank == 0:
                results.append({"batch": batch, "error": repr(e)[:300]})
                print(f"[W7FP4] batch={batch} FAILED: {repr(e)[:200]}", flush=True)
            break

    if rank == 0:
        q.put(results)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("W7_MODE", "nospec")
    assert mode in ("nospec", "spec")
    ks = KS if mode == "spec" else [0]

    from vllm.utils.network_utils import get_open_port

    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = ("eager" if EAGER else "cg")
    if A2A_US > 0:
        suffix += f"_a2a{int(A2A_US)}us"
    all_out = {"mode": mode, "model": MODEL, "draft_model": DRAFT_MODEL,
               "tag": TAG, "dp": DP, "tp": TP, "draft_quant": "nvfp4",
               "out_len": OUTLEN, "short_len": SHORTLEN, "iters": ITERS,
               "warmup": WARMUP, "gpu_mem": GPU_MEM, "eager": EAGER,
               "a2a_emulated_us": A2A_US, "suffix": suffix, "by_k": {}}

    for k in ks:
        master_ip = "127.0.0.1"
        master_port = get_open_port()
        q = Queue()
        procs = []
        for rank in range(DP):
            p = Process(target=worker,
                        args=(rank, DP, TP, master_ip, master_port, mode, k, q))
            p.start()
            procs.append(p)
        res = None
        try:
            res = q.get(timeout=5400)
        except Exception:
            res = [{"error": "no result from rank 0 (timeout)"}]
        for p in procs:
            p.join()
        all_out["by_k"][str(k)] = res
        ktag = f"K{k}" if mode == "spec" else "nospec"
        path = os.path.join(OUT_DIR, f"w7fp4_{TAG}_{mode}_{suffix}_{ktag}.json")
        with open(path, "w") as f:
            json.dump({**all_out, "this_k": k, "results": res}, f, indent=2)
        print(f"[W7FP4] wrote {path}", flush=True)

    path = os.path.join(OUT_DIR, f"w7fp4_{TAG}_{mode}_{suffix}.json")
    with open(path, "w") as f:
        json.dump(all_out, f, indent=2)
    print(f"[W7FP4] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
