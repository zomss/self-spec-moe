"""Distinguish DRAFT vs VERIFY MoE state. The draft forward sets the local-route
forward-context key (config A); the verify does not. Log state separately for
forwards WITH the key (draft) vs WITHOUT (verify)."""
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

    seen = {"draft": False, "verify": False}
    logfile = os.path.join(OUTDIR, f"state_rank{rank}.txt")
    orig_fwd = RoutedExperts.forward_modular

    def patched(self, x, topk_weights, topk_ids, *a, **k):
        is_draft = self_spec_local_route_enabled()
        tag = "DRAFT" if is_draft else "VERIFY"
        lname = str(self.layer_name)
        if not seen[tag.lower() if tag != "VERIFY" else "verify"] and "0.mlp" in lname:
            w13 = getattr(self, "w13_weight", None)
            wstr = "no-w13"
            if w13 is not None and w13.numel() > 0:
                pe = w13.float().flatten(1).norm(dim=-1)
                nz = int((pe > 1e-6).sum())
                wstr = f"w13.shape={tuple(w13.shape)} nonzero={nz}/{w13.shape[0]}"
            em = self.expert_map
            try:
                use_ep = self.use_ep
            except Exception as e:
                use_ep = f"ERR{e}"
            nres = int((em >= 0).sum()) if em is not None else "ALL60"
            line = (f"[{tag} r{rank} {lname}] use_ep={use_ep} "
                    f"expert_map_None={em is None} "
                    f"local_num_experts={self.local_num_experts} "
                    f"n_resident={nres} {wstr}\n")
            with open(logfile, "a") as f:
                f.write(line)
            seen["draft" if is_draft else "verify"] = True
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
    llm.generate(["The history of artificial intelligence began in antiquity, "
                  "with myths of artificial beings. In modern times the field "
                  "was founded in the year 1956 at Dartmouth."],
                 SamplingParams(temperature=0.0, max_tokens=8, seed=0),
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
