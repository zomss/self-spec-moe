#!/usr/bin/env python3
"""W1: one replicate boot of llama R5 for the autotune boot matrix (I4/G5).

Boot path is IDENTICAL to 95/run_e0_envelope.py's llama arm (same LLM
kwargs, same policy table, same R5 protocol, seed=0 prompts -- byte-identical
across every boot by G6's seed-0 guarantee) apart from ONE toggle:

    W1_TUNE=0  ->  kernel_config={"enable_flashinfer_autotune": False}
                   (phase 88's R88_NO_AUTOTUNE mechanism; default is ON
                   at O1+, so every phase-95 boot ran WITH autotune)

Matrix (driven by run_w1.sh): arm in {off, w2048} x tune in {1,0} x 3 boots.

Pre-registered discrimination logic (before any boot):
  P-W1a  autotune OFF collapses between-boot swing to < 5% (dense's level)
         in both arms  ->  cause = autotune kernel selection; all future
         model-validation runs set autotune off.
  P-W1b  the arm pattern localizes it: swing in spec arm only -> draft-side
         (small-M) kernels; swing in AR too -> engine-wide.
  P-W1c  if swing persists with autotune OFF, autotune is NOT the cause;
         the co-tenant/clock snapshots (recorded per boot, start+end)
         arbitrate contention vs. an unknown engine source, and llama is
         excluded from model validation per the W1 gate.

env: W1_ARM   off | w2048     W1_TUNE  1 | 0     W1_BOOT  replicate index
     W1_OUT   output json     W1_ITERS timed rounds (default 3, e0 parity)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
P95 = PHASE.parent / "95_c3_deploy"
sys.path.insert(0, str(PHASE.parent / "88_regime_eval/scripts"))
from regime_datasets import load_regime  # noqa: E402

ARM = os.environ["W1_ARM"]            # off | w2048
TUNE = os.environ.get("W1_TUNE", "1") == "1"
BOOT = int(os.environ.get("W1_BOOT", "0"))
ITERS = int(os.environ.get("W1_ITERS", "3"))

MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
DRAFT = os.path.expanduser("~/ckpts/Llama31-8B-Instruct-W4A16-INT4-sym")
SKIP = "3,8"
KMAX = 4


def gpu_snapshot():
    """Clocks/temp/util on ALL GPUs (co-tenant + thermal alternative)."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,clocks.sm,temperature.gpu,"
             "memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        return r.stdout.strip().splitlines()
    except Exception as e:  # snapshot must never kill the boot
        return [f"snapshot-failed: {e}"]


def cpu_snapshot():
    """W1b: affinity + system CPU load (source-B forensics; single-NUMA
    box, so placement means CORES, not sockets)."""
    try:
        return {
            "cpus_allowed": sorted(os.sched_getaffinity(0)),
            "n_cpus_allowed": len(os.sched_getaffinity(0)),
            "loadavg": os.getloadavg(),
        }
    except Exception as e:
        return {"snapshot-failed": str(e)}


def spec_counters(llm):
    acc = drafts = 0
    for m in llm.get_metrics():
        if m.name == "vllm:spec_decode_num_accepted_tokens":
            acc = m.value
        elif m.name == "vllm:spec_decode_num_drafts":
            drafts = m.value
    return acc, drafts


def main():
    from transformers import AutoTokenizer

    from vllm import LLM, SamplingParams

    snap0 = gpu_snapshot()
    spec = None
    if ARM != "off":
        w = ARM.lstrip("w")
        tbl = P95 / "data" / f"policy_llama_q-w4a16_s-b2_w{w}.json"
        os.environ["VLLM_SELF_SPEC_POLICY_FILE"] = str(tbl)
        os.environ["VLLM_SELF_SPEC_DRAFT_KV_WINDOW"] = w
        os.environ["VLLM_SELF_SPEC_DRAFT_SKIP_LAYERS"] = SKIP
        spec = {"method": "draft_model", "model": DRAFT,
                "num_speculative_tokens": KMAX,
                "draft_tensor_parallel_size": 1}

    extra = {}
    if not TUNE:
        extra["kernel_config"] = {"enable_flashinfer_autotune": False}

    t_boot = time.perf_counter()
    llm = LLM(model=MODEL, speculative_config=spec,
              tensor_parallel_size=1, max_model_len=20480,
              gpu_memory_utilization=0.90, max_num_seqs=32,
              enable_prefix_caching=False, disable_log_stats=False,
              async_scheduling=True, max_num_batched_tokens=8192,
              **extra)
    boot_s = time.perf_counter() - t_boot
    tok = AutoTokenizer.from_pretrained(MODEL)

    prompts, gs = load_regime("R5", tok, n=16, seed=0)
    b = gs["batch"]
    sp = SamplingParams(max_tokens=gs["max_tokens"], temperature=0.0)
    llm.generate(prompts[:b], sp, use_tqdm=False)          # warmup
    rates, accepts = [], []
    for _ in range(ITERS):
        a0, d0 = spec_counters(llm)
        t0 = time.perf_counter()
        o = llm.generate(prompts[:b], sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        a1, d1 = spec_counters(llm)
        ntok = sum(len(x.outputs[0].token_ids) for x in o)
        rates.append(ntok / dt)
        if d1 > d0:
            accepts.append(1 + (a1 - a0) / (d1 - d0))
    rates.sort()
    out = {"arm": ARM, "autotune": TUNE, "boot": BOOT, "iters": ITERS,
           "model": MODEL, "draft": DRAFT if spec else None,
           "skip_set": SKIP if spec else None, "kmax": KMAX if spec else 0,
           "toks": round(rates[len(rates) // 2], 1),
           "all": [round(r, 1) for r in rates],
           "accept": round(sum(accepts) / len(accepts), 3)
           if accepts else None,
           "boot_seconds": round(boot_s, 1),
           "pin": os.environ.get("W1_PIN", ""),
           "cpu_snapshot": cpu_snapshot(),
           "gpu_snapshot_start": snap0,
           "gpu_snapshot_end": gpu_snapshot()}
    path = os.environ.get("W1_OUT", str(
        PHASE / "data" / "w1" /
        f"w1_llama_{ARM}_{'tune' if TUNE else 'notune'}_b{BOOT}.json"))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=1))
    print(f"[W1] arm={ARM} tune={int(TUNE)} boot={BOOT} "
          f"toks={out['toks']} all={out['all']} accept={out['accept']}",
          flush=True)
    print("[W1] saved ->", path, flush=True)


if __name__ == "__main__":
    main()
