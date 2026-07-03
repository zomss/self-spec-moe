"""W7-Qwen30B losslessness spot-check (full World A stack, real forced-PCIe EP).

Greedy DP+EP forced-PCIe. Generates the same prompts under:
  nospec : plain full-EP greedy (the reference distribution).
  fp8    : full-replica FP8 comm-free draft spec, FULL_CG (the headline stack).
Reports vs the nospec greedy reference: exact-sequence count and per-token
agreement, plus spec acceptance rate / mean accept-length. Lossless == rejection
sampling preserves the verify's greedy argmax (documented W0 caveat: not
bit-identical due to batched-verify near-tie FP flips).

argv[1] one of: nospec | fp8  (run serially, one engine at a time).
Writes data/w7q_lossless_<mode>.json with per-prompt token_ids.
"""
import json
import os
import sys
from multiprocessing import Process

MODEL = os.environ.get(
    "W7_MODEL",
    "/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B"
    "/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
)
DP = int(os.environ.get("W7_DP", "8"))
TP = int(os.environ.get("W7_TP", "1"))
K = int(os.environ.get("W7L_K", "4"))
DRAFT_QUANT = os.environ.get("W7_DRAFT_QUANT", "fp8").strip()
FULL_CG = os.environ.get("W7_FULL_CG", "1") == "1"
OUT_DIR = os.environ.get(
    "W7_OUT", "/data/smcho/self-spec-moe/research/34_worldA_system/data"
)
OUTLEN = int(os.environ.get("W7L_OUTLEN", "64"))
GPU_MEM = float(os.environ.get("W7_GPU_MEM", "0.90"))

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
    spec = mode != "nospec"
    if spec:
        os.environ["VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE"] = "1"
        os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = "0"
        os.environ["VLLM_SELF_SPEC_DRAFT_FULL_REPLICA"] = "1"
        if FULL_CG:
            os.environ["VLLM_SELF_SPEC_DRAFT_FULL_CG"] = "1"

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=MODEL, tensor_parallel_size=TP, enable_expert_parallel=True,
        trust_remote_code=False, max_model_len=2048, gpu_memory_utilization=GPU_MEM,
        enforce_eager=False, disable_log_stats=False,
    )
    if spec:
        sc = {"method": "draft_model", "model": MODEL,
              "num_speculative_tokens": K, "draft_tensor_parallel_size": TP}
        if DRAFT_QUANT:
            sc["quantization"] = DRAFT_QUANT
        kwargs["speculative_config"] = sc
    llm = LLM(**kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=OUTLEN, seed=0)
    outs = llm.generate(PROMPTS, sp, use_tqdm=False)

    if rank == 0:
        results = [{"prompt": o.prompt, "token_ids": list(o.outputs[0].token_ids)}
                   for o in outs]
        summary = {"mode": mode, "K": K, "dp": DP}
        if spec:
            m = llm.get_metrics()
            acc = _metric_value(m, "vllm:spec_decode_num_accepted_tokens")
            drafted = _metric_value(m, "vllm:spec_decode_num_draft_tokens")
            nd = _metric_value(m, "vllm:spec_decode_num_drafts")
            summary["acceptance_rate"] = (acc / drafted) if (acc and drafted) else None
            summary["mean_accept_length"] = (1 + acc / nd) if (acc and nd) else None
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, f"w7q_lossless_{mode}.json")
        with open(path, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)
        print(f"[W7QL] wrote {path} summary={summary}", flush=True)


def main():
    mode = sys.argv[1]
    assert mode in ("nospec", "fp8")
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
