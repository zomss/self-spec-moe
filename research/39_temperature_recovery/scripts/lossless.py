"""Phase 39: losslessness spot-check under TEMPERATURE (proxy, DP=8 EP forced-PCIe).

Generates the same prompts at a fixed temperature/seed under:
  nospec : plain full-EP sampling (the reference distribution).
  spec   : full-replica comm-free draft + probabilistic draft + standard
           (lossless) rejection sampler.
Reports per-prompt exact-sequence match and per-token agreement vs the nospec
reference, plus spec acceptance rate / mean accept-length. The standard rejection
sampler is distributional-lossless; with a fixed per-request seed the uniform and
recovered draws are seed-coupled, so spec vs no-spec should produce closely
matching tokens (documented W0 caveat: batched-verify near-tie FP flips can still
diverge a few tokens). This confirms the active path is the lossless standard
sampler, not a quality shortcut.

argv[1] one of: nospec | spec  (run serially, one engine at a time).
"""
import json
import os
import sys
from multiprocessing import Process

MODEL = os.environ.get(
    "E_MODEL",
    "/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B"
    "/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9",
)
DP = int(os.environ.get("E_DP", "8"))
TP = int(os.environ.get("E_TP", "1"))
K = int(os.environ.get("E_K", "4"))
TEMP = float(os.environ.get("E_TEMP", "0.7"))
SEED = int(os.environ.get("E_SEED", "0"))
OUTLEN = int(os.environ.get("E_OUTLEN", "64"))
GPU_MEM = float(os.environ.get("E_GPU_MEM", "0.85"))
PROBABILISTIC = os.environ.get("E_PROBABILISTIC", "1") == "1"
OUT_DIR = os.environ.get(
    "E_OUT", "/data/smcho/self-spec-moe/research/39_temperature_recovery/data"
)

PROMPTS = [
    "The capital of France is",
    "Q: What is 2 + 2? A:",
    "Once upon a time, in a small village,",
    "def fibonacci(n):\n    ",
    "The three primary colors are",
    "Water is made of hydrogen and",
    "The largest planet in our solar system is",
    "To make a peanut butter sandwich, first you",
    "The speed of light in a vacuum is approximately",
    "Shakespeare wrote many famous plays, including",
    "The chemical symbol for gold is",
    "A binary search algorithm works by",
    "The first president of the United States was",
    "Photosynthesis is the process by which plants",
    "In Python, a list comprehension is written as",
    "The Great Wall of China was built to",
]


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


def worker(rank, master_ip, master_port, mode):
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(DP)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)
    spec = mode == "spec"
    if spec:
        os.environ["VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE"] = "1"
        os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = "0"
        os.environ["VLLM_SELF_SPEC_DRAFT_FULL_REPLICA"] = "1"
        os.environ["VLLM_SELF_SPEC_DRAFT_FULL_CG"] = "1"
        os.environ["VLLM_SELF_SPEC_COMPILE_CONSISTENT"] = "1"

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=MODEL, tensor_parallel_size=TP, enable_expert_parallel=True,
        trust_remote_code=True, max_model_len=2048, gpu_memory_utilization=GPU_MEM,
        enforce_eager=False, disable_log_stats=False,
    )
    if spec:
        sc = {"method": "draft_model", "model": MODEL,
              "num_speculative_tokens": K, "draft_tensor_parallel_size": TP,
              "rejection_sample_method": "standard"}
        if PROBABILISTIC:
            sc["draft_sample_method"] = "probabilistic"
        kwargs["speculative_config"] = sc
    llm = LLM(**kwargs)
    if rank == 0 and spec:
        sc_resolved = llm.llm_engine.vllm_config.speculative_config
        print("[CFG] draft_sample_method="
              f"{getattr(sc_resolved, 'draft_sample_method', '?')} "
              "rejection_sample_method="
              f"{getattr(sc_resolved, 'rejection_sample_method', '?')}", flush=True)
    sp = SamplingParams(temperature=TEMP, max_tokens=OUTLEN, seed=SEED)
    outs = llm.generate(PROMPTS, sp, use_tqdm=False)

    if rank == 0:
        results = [{"prompt": o.prompt, "token_ids": list(o.outputs[0].token_ids)}
                   for o in outs]
        summary = {"mode": mode, "K": K, "dp": DP, "temp": TEMP, "seed": SEED}
        if spec:
            m = llm.get_metrics()
            acc = _metric_value(m, "vllm:spec_decode_num_accepted_tokens")
            drafted = _metric_value(m, "vllm:spec_decode_num_draft_tokens")
            nd = _metric_value(m, "vllm:spec_decode_num_drafts")
            summary["acceptance_rate"] = (acc / drafted) if (acc and drafted) else None
            summary["mean_accept_length"] = (1 + acc / nd) if (acc and nd) else None
        os.makedirs(OUT_DIR, exist_ok=True)
        tstr = f"t{TEMP:g}".replace(".", "p")
        path = os.path.join(OUT_DIR, f"lossless_{mode}_{tstr}.json")
        with open(path, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)
        print(f"[LL] wrote {path} summary={summary}", flush=True)


def main():
    mode = sys.argv[1]
    assert mode in ("nospec", "spec")
    from vllm.utils.network_utils import get_open_port
    master_ip, master_port = "127.0.0.1", get_open_port()
    procs = []
    for rank in range(DP):
        p = Process(target=worker, args=(rank, master_ip, master_port, mode))
        p.start()
        procs.append(p)
    code = 0
    for p in procs:
        p.join()
        if p.exitcode:
            code = p.exitcode
    sys.exit(code)


if __name__ == "__main__":
    main()
