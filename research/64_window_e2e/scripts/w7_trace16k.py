"""Torch-profiler trace of the window-KV self-draft decode cycle at 16k.

Phase-64 merge of research/52_two_node_e2e/scripts/w7_trace.py (rank-0
profiler window) and w7_2node.py (2-node spawn + Phase-59 16k prompt
padding + chat). GLOBAL rank 0 (h107) traces a ~TRACE_LEN-token decode
window (~TRACE_LEN/accept_len spec cycles); all other ranks run lockstep
untraced. Draft env defaults match the Phase-64 EP-routed bf16 self-draft
(NO quant / NO replica / NO local or node routing).

Run on both nodes with W7_NODE_RANK=0 (h107, writes trace) and 1 (h106).
"""

import os
import sys
from multiprocessing import Process

MODEL = os.environ.get("W7_MODEL", "Qwen/Qwen3-30B-A3B")
NODE_RANK = int(os.environ.get("W7_NODE_RANK", "0"))
NODES = int(os.environ.get("W7_NODES", "2"))
LOCAL_WORLD = int(os.environ.get("W7_LOCAL_WORLD", "8"))
DP = NODES * LOCAL_WORLD
MASTER_IP = os.environ.get("W7_MASTER_IP", "192.168.0.17")
MASTER_PORT = int(os.environ.get("W7_MASTER_PORT", "14500"))
BATCH = int(os.environ.get("W7_BATCH", "8"))
K = int(os.environ.get("W7_K", "2"))
CTX_TOKENS = int(os.environ.get("W7_CTX_TOKENS", "16384"))
MAX_MODEL_LEN = int(os.environ.get("W7_MAX_MODEL_LEN", "20480"))
MNB = int(os.environ.get("W7_MAX_NUM_BATCHED", "8192"))
GPU_MEM = float(os.environ.get("W7_GPU_MEM", "0.90"))
TRACE_LEN = int(os.environ.get("W7_TRACE_LEN", "60"))
TRACE_DIR = os.environ.get(
    "W7_TRACE_DIR",
    "/h/v-sukmincho/self-spec-moe/research/64_window_e2e/data/trace_w512",
)
PROMPT_FILE = os.environ.get(
    "W7_PROMPT_FILE",
    "/h/v-sukmincho/self-spec-moe/research/57_large_ep_spec_strategy/"
    "data/prompts_ondist.txt",
)


def worker(rank, local_rank):
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_rank)
    os.environ["VLLM_DP_SIZE"] = str(DP)
    os.environ["VLLM_DP_MASTER_IP"] = MASTER_IP
    os.environ["VLLM_DP_MASTER_PORT"] = str(MASTER_PORT)

    # Phase-64 EP-routed bf16 self-draft defaults (cf. run_arm.sh SELFSPEC).
    os.environ["VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE"] = os.environ.get(
        "W7_DRAFT_LOCAL_ROUTE", "0"
    )
    os.environ["VLLM_SELF_SPEC_DRAFT_NODE_LOCAL"] = os.environ.get(
        "W7_DRAFT_NODE_LOCAL", "0"
    )
    os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = "0"
    os.environ["VLLM_SELF_SPEC_DRAFT_FULL_REPLICA"] = os.environ.get(
        "W7_DRAFT_FULL_REPLICA", "0"
    )

    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=MODEL,
        tensor_parallel_size=1,
        enable_expert_parallel=True,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM,
        enforce_eager=False,
        disable_log_stats=False,
        max_num_batched_tokens=MNB,
    )
    # W7_NOSPEC=1 -> plain decode (no draft) for a same-harness no-spec trace.
    if os.environ.get("W7_NOSPEC", "0") != "1":
        kwargs["speculative_config"] = {
            "method": "draft_model",
            "model": MODEL,
            "num_speculative_tokens": K,
            "draft_tensor_parallel_size": 1,
        }
        if os.environ.get("W7_DRAFT_QUANT", "").strip():
            kwargs["speculative_config"]["quantization"] = \
                os.environ["W7_DRAFT_QUANT"].strip()
    if rank == 0:
        kwargs["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": TRACE_DIR,
            "torch_profiler_with_stack": bool(
                int(os.environ.get("W7_TRACE_STACK", "0"))
            ),
        }
    llm = LLM(**kwargs)

    # Phase-59 16k padding, condensed from w7_2node.py: rotated natural-text
    # filler + unique doc marker inside the user turn, then chat template.
    tok = llm.get_tokenizer()
    with open(PROMPT_FILE) as f:
        bank = [ln.strip() for ln in f if ln.strip()]
    filler = (
        "In the broader study of natural language and machine reasoning, "
        "researchers have long observed that context shapes meaning in subtle "
        "and far-reaching ways, and that the surrounding passage a model reads "
        "before it answers can change every prediction that follows. The city "
        "sat at the edge of a wide river, and each morning the markets filled "
        "with traders carrying grain, salt, cloth, and stories gathered from "
        "distant provinces. Over the centuries, scholars debated the nature of "
        "memory, the structure of language, the movement of the planets, and "
        "the slow accumulation of knowledge that turns observation into theory. "
        "A traveller who kept a careful journal recorded the weather, the price "
        "of bread, the names of the ships in the harbour, and the arguments of "
        "the philosophers who gathered in the shaded courtyards to reason about "
        "cause and consequence, about what can be known and what must be "
        "supposed. These records, though ordinary in their day, later became a "
        "window into a vanished world of commerce, curiosity, and quiet labor. "
    )
    corpus = tok(filler, add_special_tokens=False).input_ids
    while len(corpus) < 2 * CTX_TOKENS + 64:
        corpus = corpus + corpus
    prompts = []
    for i in range(BATCH):
        p = bank[i % len(bank)]
        p_ids = tok(p, add_special_tokens=False).input_ids
        marker = f"Document #{i}: "
        m_ids = tok(marker, add_special_tokens=False).input_ids
        need = max(0, CTX_TOKENS - len(p_ids) - len(m_ids))
        off = (i * 997) % (len(corpus) - need) if need > 0 else 0
        fill = tok.decode(corpus[off:off + need])
        prompts.append(
            tok.apply_chat_template(
                [{"role": "user",
                  "content": f"{marker}{fill}\n\nNow answer this question. {p}"}],
                add_generation_prompt=True, tokenize=False,
            )
        )

    def gen(n):
        sp = SamplingParams(
            temperature=0.0, max_tokens=n, ignore_eos=True, seed=0
        )
        llm.generate(prompts, sp, use_tqdm=False)

    gen(24)
    gen(24)
    if rank == 0:
        llm.start_profile()
    gen(TRACE_LEN)
    if rank == 0:
        llm.stop_profile()
        print(
            f"[TRACE16K] K={K} batch={BATCH} ctx={CTX_TOKENS} "
            f"trace_len={TRACE_LEN} -> {TRACE_DIR}",
            flush=True,
        )


def main():
    os.makedirs(TRACE_DIR, exist_ok=True)
    procs = []
    for local_rank in range(LOCAL_WORLD):
        rank = NODE_RANK * LOCAL_WORLD + local_rank
        p = Process(target=worker, args=(rank, local_rank))
        p.start()
        procs.append(p)
    rc = 0
    for p in procs:
        p.join()
        rc |= p.exitcode or 0
    sys.exit(rc)


if __name__ == "__main__":
    main()
