"""Phase 45 — prefill-cancelled decode cycle-ms + speedup, before/after the CPU
-orch reductions, for the comm-free self-spec config.

Mirrors research/34_worldA_system/scripts/w7_timing.py's two-length slope
method (decode_time = t(OUTLEN) - t(SHORTLEN) cancels the shared prefill), but
parameterizes the model and adds the three modes:

  nospec      : plain full-EP decode (baseline for speedup denominator).
  spec_base   : comm-free self-spec, reduction knobs OFF (current binary).
  spec_orch   : comm-free self-spec, VLLM_SELF_SPEC_CPU_ORCH=1.

Reports, per mode: decode_s, cycle_ms (= decode_s / steps * 1000 for nospec;
for spec = accept_len * batch / sys_tok_s using the slope tok/s), sys tok/s,
and accept_len (spec). speedup = spec_tok_s / nospec_tok_s.

Env (CS_*): CS_MODEL, CS_DP=8, CS_BATCH=64, CS_K=2, CS_DRAFT_QUANT=fp8,
CS_OUTLEN=128, CS_SHORTLEN=32, CS_ITERS=3, CS_WARMUP=1, CS_GPU_MEM=0.90,
CS_MAX_MODEL_LEN=2048, CS_MODES=nospec,spec_base,spec_orch, CS_OUT, CS_TAG.
"""
import json
import os
import sys
import time
from multiprocessing import Process, Queue

MODEL = os.environ.get("CS_MODEL", "Qwen/Qwen1.5-MoE-A2.7B")
DP = int(os.environ.get("CS_DP", "8"))
BATCH = int(os.environ.get("CS_BATCH", "64"))
K = int(os.environ.get("CS_K", "2"))
DRAFT_QUANT = os.environ.get("CS_DRAFT_QUANT", "fp8").strip()
OUTLEN = int(os.environ.get("CS_OUTLEN", "128"))
SHORTLEN = int(os.environ.get("CS_SHORTLEN", "32"))
ITERS = int(os.environ.get("CS_ITERS", "3"))
WARMUP = int(os.environ.get("CS_WARMUP", "1"))
GPU_MEM = float(os.environ.get("CS_GPU_MEM", "0.90"))
MAX_MODEL_LEN = int(os.environ.get("CS_MAX_MODEL_LEN", "2048"))
MODES = os.environ.get("CS_MODES", "nospec,spec_base,spec_orch").split(",")
OUT_DIR = os.environ.get("CS_OUT", os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))) + "/data")
TAG = os.environ.get("CS_TAG", "qwen15moe")

BASE_PROMPT = (
    "The history of artificial intelligence began in antiquity with myths and "
    "stories, and the modern field was founded in"
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
    total, found = 0.0, False
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


def worker(rank, dp, master_ip, master_port, mode, q):
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(dp)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

    spec = mode.startswith("spec")
    if spec:
        os.environ["VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE"] = "1"
        os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = "0"
        os.environ["VLLM_SELF_SPEC_DRAFT_FULL_REPLICA"] = "1"
        os.environ["VLLM_SELF_SPEC_DRAFT_FULL_CG"] = "1"
        os.environ["VLLM_SELF_SPEC_COMPILE_CONSISTENT"] = "1"
        if mode == "spec_orch":
            os.environ["VLLM_SELF_SPEC_CPU_ORCH"] = "1"

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=MODEL, tensor_parallel_size=1, enable_expert_parallel=True,
        trust_remote_code=False, max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM, enforce_eager=False,
        disable_log_stats=False,
    )
    if spec:
        kwargs["speculative_config"] = {
            "method": "draft_model", "model": MODEL,
            "num_speculative_tokens": K, "draft_tensor_parallel_size": 1,
            **({"quantization": DRAFT_QUANT} if DRAFT_QUANT else {}),
        }
    llm = LLM(**kwargs)
    prompts = [f"{BASE_PROMPT} the year {1900 + i}." for i in range(BATCH)]

    def run(out_len):
        sp = SamplingParams(temperature=0.0, max_tokens=out_len,
                            ignore_eos=True, seed=0)
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        return dt, ntok

    def snap():
        if not (spec and rank == 0):
            return None
        m = llm.get_metrics()
        return (_metric_value(m, "vllm:spec_decode_num_accepted_tokens") or 0.0,
                _metric_value(m, "vllm:spec_decode_num_drafts") or 0.0)

    for _ in range(WARMUP):
        run(OUTLEN); run(SHORTLEN)
    s0 = snap()
    longs, shorts = [], []
    for _ in range(ITERS):
        lt, _ = run(OUTLEN); st, _ = run(SHORTLEN)
        longs.append(lt); shorts.append(st)
    s1 = snap()
    if rank == 0:
        dts = [lt - st for lt, st in zip(longs, shorts)]
        dt_m, dt_s = _mean_std(dts)
        steps = OUTLEN - SHORTLEN
        out_decode = BATCH * steps
        tok_s = out_decode / dt_m if dt_m else None
        res = {"mode": mode, "decode_s": dt_m, "decode_s_std": dt_s,
               "steps": steps, "sys_tok_s": tok_s}
        if spec and s0 and s1:
            acc = s1[0] - s0[0]; nd = s1[1] - s0[1]
            al = (1 + acc / nd) if nd else None
            res["accept_len"] = al
            # cycle_ms = accept_len*batch / tok_s (spec system)
            res["cycle_ms"] = (al * BATCH / tok_s * 1000) if (al and tok_s) else None
        else:
            res["cycle_ms"] = (dt_m / steps * 1000) if dt_m else None
        q.put(res)


def main():
    from vllm.utils.network_utils import get_open_port
    os.makedirs(OUT_DIR, exist_ok=True)
    all_res = {}
    for mode in MODES:
        mode = mode.strip()
        master_port = get_open_port()
        q = Queue()
        procs = [Process(target=worker,
                         args=(r, DP, "127.0.0.1", master_port, mode, q))
                 for r in range(DP)]
        for p in procs:
            p.start()
        try:
            res = q.get(timeout=3600)
        except Exception:
            res = {"mode": mode, "error": "timeout"}
        for p in procs:
            p.join()
        all_res[mode] = res
        print(f"[CS] {mode}: {json.dumps(res)}", flush=True)

    # speedup vs nospec
    ns = all_res.get("nospec", {}).get("sys_tok_s")
    for mode in MODES:
        mode = mode.strip()
        r = all_res.get(mode, {})
        sp = r.get("sys_tok_s")
        if ns and sp:
            print(f"[CS] speedup {mode} vs nospec = {sp/ns:.4f}x "
                  f"(tok/s {sp:.1f} vs {ns:.1f}, cycle_ms={r.get('cycle_ms')})",
                  flush=True)
    path = os.path.join(OUT_DIR, f"cs_{TAG}_b{BATCH}_K{K}.json")
    with open(path, "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"[CS] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
