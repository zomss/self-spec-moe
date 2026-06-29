"""No-spec greedy baseline in the same DP=2+EP layout for losslessness diff."""
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


def worker(rank, dp, tp, master_ip, master_port):
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(dp)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=tp,
        enable_expert_parallel=True,
        trust_remote_code=False,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=64, seed=0)
    outs = llm.generate(PROMPTS, sp)
    if rank == 0:
        results = [
            {"prompt": o.prompt, "text": o.outputs[0].text,
             "token_ids": list(o.outputs[0].token_ids)}
            for o in outs
        ]
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, "dp_nospec.json")
        with open(path, "w") as f:
            json.dump({"results": results}, f, indent=2)
        print(f"[W2-NOSPEC] wrote {path}")


def main():
    from vllm.utils.network_utils import get_open_port

    master_ip = "127.0.0.1"
    master_port = get_open_port()
    procs = []
    for rank in range(DP):
        p = Process(target=worker, args=(rank, DP, TP, master_ip, master_port))
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
