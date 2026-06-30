"""Phase 40 Step 1: dump per-token MoE internals for config A vs C (same prompt).

On DP rank 0, at one chosen MoE layer, captures for the FIRST (prefill) forward:
  (a) router logits, (b) selected expert ids + weights, (c) post-combine MoE
  output, and (d) the final next-token logits. Configs A and C are run on the
  SAME fixed short prompt so the dumps can be diffed:
    - (a)+(b) MATCH  -> same routing (not a routing bug)
    - (c) DIVERGES   -> the comm-free-vs-EP bf16 reduce structure (quantify)
    - (d) argmax flips = the per-token rejections.

EAGER (no cudagraph) so the env-gated torch.save in the MoE/logits forward is not
captured into a graph. The collapse is present eager (phase 38), so eager is a
faithful probe of the reduce-structure divergence.

Env:
  CFG     A | C   (required)  A=full-replica comm-free, C=EP all-to-all
  E_DP    DP size (=EP)       default 8
  E_LAYER dump layer index    default 0
  E_OUT   dump dir            default research/40_num_divergence/data/dump_<cfg>
  E_FP32  1 -> fp32-accum     default 0
"""
import os
from multiprocessing import Process

MODEL = os.environ.get(
    "E_MODEL",
    "/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B"
    "/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9",
)
CFG = os.environ["CFG"].strip().upper()
DP = int(os.environ.get("E_DP", "8"))
LAYER = int(os.environ.get("E_LAYER", "0"))
FP32 = os.environ.get("E_FP32", "0")
OUT = os.environ.get(
    "E_OUT",
    f"/data/smcho/self-spec-moe/research/40_num_divergence/data/dump_{CFG}"
    + ("_fp32" if FP32 == "1" else ""),
)

# Main-model reduce-structure probe (no spec decode):
#   A = full replica + comm-free local sum  -> enable_expert_parallel=False,
#       LOCAL_ROUTE=1 (expert_map=None -> all 60 experts resident, masking is a
#       no-op, combine() passes -> ONE local bf16 moe_sum over all top-k).
#   C = EP all-to-all (= the verify)         -> enable_expert_parallel=True,
#       LOCAL_ROUTE=0 (60/8 experts/rank -> per-shard bf16 moe_sum then the
#       cross-rank bf16 reduce_scatterv combine).
# Same model, same prompt, same routing -> the ONLY difference is the reduce
# structure, which is exactly the divergence under test.
CFG_MAP = {
    "A": ("0", "1"),  # full replica + comm-free local sum
    "C": ("1", "0"),  # EP all-to-all (= the verify)
    "D": ("0", "0"),  # plain baseline: EP off, NO local-route skip (canonical
                      # standard MoE; at DP=1 = single-GPU ground truth)
}
assert CFG in CFG_MAP, f"CFG must be A/C/D, got {CFG}"
ENABLE_EP, LOCAL_ROUTE = CFG_MAP[CFG]


def worker(rank, master_ip, master_port):
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(DP)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

    # Drive the MAIN (non-draft) MoE path through the chosen reduce structure so
    # the dumped layer output reflects config A's local sum vs config C's EP
    # combine. LOCAL_ROUTE on the verify reader makes combine() skip the
    # reduce-scatter (config A); off keeps the real all-to-all (config C).
    os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = LOCAL_ROUTE
    os.environ["VLLM_SELF_SPEC_MOE_FP32_ACCUM"] = FP32
    # Only rank 0 writes the dump (record_* checks VLLM_DP_RANK == "0").
    os.environ["VLLM_SELF_SPEC_MOE_NUM_DUMP"] = OUT
    os.environ["VLLM_SELF_SPEC_MOE_DUMP_LAYER"] = str(LAYER)

    from vllm import LLM, SamplingParams

    # No speculative_config: we are probing the MoE forward reduce structure
    # directly (the draft IS the same model; the divergence is the MoE reduce,
    # exercised here on the main model under the chosen LOCAL_ROUTE/FULL_REPLICA).
    kwargs = dict(
        model=MODEL,
        tensor_parallel_size=1,
        enable_expert_parallel=(ENABLE_EP == "1"),
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        enable_prefix_caching=False,  # force the real prompt forward to recompute
        disable_log_stats=True,
    )
    llm = LLM(**kwargs)

    # Fixed short prompt (same for A and C). Batch on each rank is identical.
    prompt = (
        "The history of artificial intelligence began in antiquity, with "
        "myths and stories of artificial beings endowed with intelligence by "
        "master craftsmen. In modern times, the field of AI research was "
        "founded at a workshop held on the campus of Dartmouth College"
    )
    sp = SamplingParams(temperature=0.0, max_tokens=1, seed=0)

    # Arm AFTER init/profiling (so those forwards are never dumped) but before
    # the single real generate. The dump's min-distinct-rows guard additionally
    # skips any DP dummy/coordination forward, so it latches on the real prompt.
    # Single generate -> no DP-lockstep double-generate hang.
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "ARM"), "w").close()
    llm.generate([prompt], sp, use_tqdm=False)
    if rank == 0:
        print(f"[DUMP] cfg={CFG} fp32={FP32} layer={LAYER} -> {OUT}", flush=True)


def main():
    from vllm.utils.network_utils import get_open_port
    master_ip, master_port = "127.0.0.1", get_open_port()
    procs = [Process(target=worker, args=(r, master_ip, master_port))
             for r in range(DP)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=900)
    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join(timeout=30)


if __name__ == "__main__":
    main()
