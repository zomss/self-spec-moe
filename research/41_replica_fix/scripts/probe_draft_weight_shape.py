"""Phase 41: confirm the post-fix full-replica DRAFT MoE holds FULL unsharded
experts (w13 intermediate 2816, not the 352 = 2816/8 TP-shard from Bug B) and
local_num_experts=60 on every rank."""
import os
from multiprocessing import Process

MODEL = os.environ.get(
    "E_MODEL",
    "/home/smcho/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B"
    "/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9",
)
DP = int(os.environ.get("E_DP", "8"))
OUTDIR = os.environ["E_OUTDIR"]


def worker(rank, master_ip, master_port):
    os.environ["VLLM_DP_RANK"] = str(rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
    os.environ["VLLM_DP_SIZE"] = str(DP)
    os.environ["VLLM_DP_MASTER_IP"] = master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)
    os.environ["VLLM_SELF_SPEC_DRAFT_FULL_REPLICA"] = "1"
    os.environ["VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE"] = "1"
    os.environ["VLLM_SELF_SPEC_LOCAL_ROUTE"] = "0"

    from vllm import LLM, SamplingParams
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.forward_context import self_spec_local_route_enabled

    seen = {"d": False}
    logfile = os.path.join(OUTDIR, f"w_rank{rank}.txt")
    orig_fwd = RoutedExperts.forward_modular

    def patched(self, x, topk_weights, topk_ids, *a, **k):
        if (self_spec_local_route_enabled() and not seen["d"]
                and "0.mlp" in str(self.layer_name)):
            mc = self.moe_config.moe_parallel_config
            # find a w13 param to read its intermediate dim
            w13 = None
            em = getattr(self, "expert_map", "MISSING")
            for n, p in self.named_parameters():
                if "w13" in n or "w1" in n:
                    w13 = (n, tuple(p.shape))
                    break
            with open(logfile, "a") as f:
                f.write(
                    f"[DRAFT r{rank} {self.layer_name}] "
                    f"local_num_experts={getattr(self, 'local_num_experts', '?')} "
                    f"global_num_experts={getattr(self, 'global_num_experts', '?')} "
                    f"expert_map={'None' if em is None else 'not None'} "
                    f"tp_size={mc.tp_size} ep_size={mc.ep_size} "
                    f"w13_param={w13}\n"
                )
            seen["d"] = True
        return orig_fwd(self, x, topk_weights, topk_ids, *a, **k)

    RoutedExperts.forward_modular = patched

    kwargs = dict(
        model=MODEL, tensor_parallel_size=1, enable_expert_parallel=True,
        trust_remote_code=True, max_model_len=2048,
        gpu_memory_utilization=0.85, enforce_eager=True,
        enable_prefix_caching=False, disable_log_stats=True,
        speculative_config={
            "method": "draft_model", "model": MODEL,
            "num_speculative_tokens": 4, "draft_tensor_parallel_size": 1,
        },
    )
    llm = LLM(**kwargs)
    llm.generate(["The history of AI began in antiquity with myths of "
                  "artificial beings. The field was founded in 1956."],
                 SamplingParams(temperature=0.0, max_tokens=6, seed=0),
                 use_tqdm=False)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
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
