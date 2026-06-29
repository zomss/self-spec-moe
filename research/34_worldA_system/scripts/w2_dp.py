"""W2a full-replica validation: DP=2 + EP, draft_model spec decode.

Adapted from w0_dp.py. Adds acceptance measurement (from LLM.get_metrics:
spec_decode_num_accepted_tokens_total / spec_decode_num_draft_tokens_total) and
a VLLM_SELF_SPEC_DRAFT_FULL_REPLICA knob.

argv: <mode>
  shard    : W0 EP-shard draft, producer ON (comm-free, low coverage)  -> ~0.758
  replica  : W2a full-replica draft (knob ON), producer OFF (non-EP)    -> ~1.0
  replica_route : full-replica draft + producer ON (mask no-op, sanity)
"""
import json
import os
import sys
from multiprocessing import Process

MODEL = (
    "/home/smcho/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite"
    "/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0"
)
DP = int(os.environ.get("W0_DP", "2"))
TP = int(os.environ.get("W0_TP", "1"))
K = int(os.environ.get("W0_K", "4"))
OUT_DIR = os.environ.get("W0_OUT", "/tmp/w2_out")

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
    total = 0.0
    found = False
    for m in metrics:
        if m.name == name:
            found = True
            # Counter -> .value ; vectors -> .values (list)
            v = getattr(m, "value", None)
            if v is not None:
                total += v
            else:
                vals = getattr(m, "values", None)
                if vals is not None:
                    total += sum(vals)
    return total if found else None


def worker(rank, dp, tp, master_ip, master_port, draft_flag, full_replica):
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(dp)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)
    os.environ["VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE"] = draft_flag
    os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = "0"
    os.environ["VLLM_SELF_SPEC_LOG_A2A_COUNTS"] = "1"
    os.environ["VLLM_SELF_SPEC_DRAFT_FULL_REPLICA"] = full_replica

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=MODEL,
        tensor_parallel_size=tp,
        enable_expert_parallel=True,
        trust_remote_code=False,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        disable_log_stats=False,
        speculative_config={
            "method": "draft_model",
            "model": MODEL,
            "num_speculative_tokens": K,
            "draft_tensor_parallel_size": tp,
        },
    )
    llm = LLM(**kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=64, seed=0)
    outs = llm.generate(PROMPTS, sp)

    if rank == 0:
        metrics = llm.get_metrics()
        acc = _metric_value(metrics, "vllm:spec_decode_num_accepted_tokens")
        drafted = _metric_value(metrics, "vllm:spec_decode_num_draft_tokens")
        n_drafts = _metric_value(metrics, "vllm:spec_decode_num_drafts")
        rate = (acc / drafted) if (acc is not None and drafted) else float("nan")
        accept_len = (1 + acc / n_drafts) if (acc is not None and n_drafts) else float("nan")
        results = [
            {"prompt": o.prompt, "text": o.outputs[0].text,
             "token_ids": list(o.outputs[0].token_ids)}
            for o in outs
        ]
        os.makedirs(OUT_DIR, exist_ok=True)
        mode = os.environ.get("W2_MODE", "run")
        path = os.path.join(OUT_DIR, f"dp_{mode}.json")
        summary = {
            "mode": mode,
            "draft_full_replica": full_replica,
            "draft_local_route": draft_flag,
            "num_accepted_tokens": acc,
            "num_draft_tokens": drafted,
            "num_drafts": n_drafts,
            "acceptance_rate": rate,
            "mean_accept_length": accept_len,
            "K": K,
        }
        with open(path, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)
        print("=" * 70)
        print(f"[W2-DP] mode={mode} full_replica={full_replica} "
              f"draft_local_route={draft_flag}")
        print(f"[W2-DP] ACCEPTANCE RATE = {rate:.4f}  "
              f"(accepted={acc}, drafted={drafted})")
        print(f"[W2-DP] MEAN ACCEPT LEN = {accept_len:.3f}  "
              f"(drafts={n_drafts}, K={K})")
        print(f"[W2-DP] wrote {path}")
        print("=" * 70)


def main():
    mode = sys.argv[1]
    # shard: producer ON, replica off. replica: full replica ON, producer OFF.
    if mode == "shard":
        draft_flag, full_replica = "1", "0"
    elif mode == "replica":
        draft_flag, full_replica = "0", "1"
    elif mode == "replica_route":
        draft_flag, full_replica = "1", "1"
    else:
        raise SystemExit(f"unknown mode {mode}")
    os.environ["W2_MODE"] = mode

    from vllm.utils.network_utils import get_open_port

    master_ip = "127.0.0.1"
    master_port = get_open_port()
    procs = []
    for rank in range(DP):
        p = Process(
            target=worker,
            args=(rank, DP, TP, master_ip, master_port, draft_flag, full_replica),
        )
        p.start()
        procs.append(p)
    exit_code = 0
    for p in procs:
        p.join()
        if p.exitcode:
            exit_code = p.exitcode
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
