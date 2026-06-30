"""Deepest probe: dump per-topk-slot expert outputs at TritonExperts.moe_sum.

`intermediate_cache3` is [num_tokens, top_k, K] -- the per-expert contributions
BEFORE the final sum. If at DP=8 some of the 4 slots are zero/smaller than DP=1
(with identical routing), an expert's GEMM output is being dropped.
"""
import os
from multiprocessing import Process

MODEL = os.environ.get(
    "E_MODEL",
    "/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B"
    "/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9",
)
CFG = os.environ.get("CFG", "A").strip().upper()
DP = int(os.environ.get("E_DP", "8"))
OUT = os.environ["E_OUT"]
CFG_MAP = {"A": ("0", "1"), "C": ("1", "0"), "D": ("0", "0")}
ENABLE_EP, LOCAL_ROUTE = CFG_MAP[CFG]


def worker(rank, master_ip, master_port):
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(DP)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)
    os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = LOCAL_ROUTE

    import torch
    from vllm import LLM, SamplingParams
    from vllm.model_executor.layers.fused_moe.experts import triton_moe as TM

    cls = TM.TritonExperts
    orig = cls.moe_sum
    state = {"done": False}

    def patched_moe_sum(self, inp, output):
        # inp: [num_tokens, top_k, K]
        if rank == 0 and not state["done"] and inp.dim() == 3:
            feo = inp.float()
            m = feo.shape[0]
            slot_norm = feo.norm(dim=-1)  # [m, topk]
            row_norm = slot_norm.norm(dim=-1)
            real = row_norm > 0
            n_real = int(real.sum())
            if n_real >= 8:
                sn = slot_norm[real]  # [n_real, topk]
                frac_zero = (sn < 1e-6).float().mean(dim=0)
                torch.save({
                    "n_real": n_real, "topk": sn.shape[1],
                    "slot_norm_mean": sn.mean(dim=0).cpu(),
                    "frac_zero_per_slot": frac_zero.cpu(),
                    "per_token_slot_norm": sn.cpu(),
                    "total_out_norm": output_norm_after(self, inp, output),
                }, OUT)
                print(f"[MOESUM rank0] n_real={n_real} topk={sn.shape[1]} "
                      f"slot_norm_mean={[round(x,4) for x in sn.mean(dim=0).tolist()]} "
                      f"frac_zero_per_slot={[round(x,4) for x in frac_zero.tolist()]}",
                      flush=True)
                state["done"] = True
        return orig(self, inp, output)

    def output_norm_after(self, inp, output):
        return float(inp.float().sum(dim=1)[(inp.float().sum(dim=1).norm(dim=-1) > 0)].norm())

    cls.moe_sum = patched_moe_sum

    kwargs = dict(
        model=MODEL, tensor_parallel_size=1,
        enable_expert_parallel=(ENABLE_EP == "1"),
        trust_remote_code=True, max_model_len=2048,
        gpu_memory_utilization=0.85, enforce_eager=True,
        enable_prefix_caching=False, disable_log_stats=True,
    )
    llm = LLM(**kwargs)
    prompt = (
        "The history of artificial intelligence began in antiquity, with "
        "myths and stories of artificial beings endowed with intelligence by "
        "master craftsmen. In modern times, the field of AI research was "
        "founded at a workshop held on the campus of Dartmouth College"
    )
    llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=1, seed=0),
                 use_tqdm=False)


def main():
    from vllm.utils.network_utils import get_open_port
    master_ip, master_port = "127.0.0.1", get_open_port()
    procs = [Process(target=worker, args=(r, master_ip, master_port))
             for r in range(DP)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=600)
    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join(timeout=30)


if __name__ == "__main__":
    main()
