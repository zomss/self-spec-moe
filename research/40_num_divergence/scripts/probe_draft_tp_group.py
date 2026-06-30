"""Check the DRAFT full-replica MoE's tp_size + whether the TP all-reduce
group is trivial (size 1) in a DP deployment -> 1/8 sharded output."""
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
    from vllm.distributed.parallel_state import get_tp_group, get_dp_group

    seen = {"d": False}
    logfile = os.path.join(OUTDIR, f"tp_rank{rank}.txt")
    orig_fwd = RoutedExperts.forward_modular

    def patched(self, x, topk_weights, topk_ids, *a, **k):
        if self_spec_local_route_enabled() and not seen["d"] and "0.mlp" in str(self.layer_name):
            mc = self.moe_config.moe_parallel_config
            try:
                tpg = get_tp_group()
                tp_world = tpg.world_size
            except Exception as e:
                tp_world = f"ERR{e}"
            with open(logfile, "a") as f:
                f.write(
                    f"[DRAFT r{rank} {self.layer_name}] "
                    f"moe_tp_size={mc.tp_size} moe_ep_size={mc.ep_size} "
                    f"moe_dp_size={mc.dp_size} use_ep={mc.use_ep} "
                    f"GLOBAL get_tp_group().world_size={tp_world} "
                    f"=> all_reduce over {tp_world} ranks "
                    f"(no-op if 1 -> 1/{mc.tp_size} sharded output)\n"
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
