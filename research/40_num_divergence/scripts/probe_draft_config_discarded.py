"""Confirm: draft_parallel_config has EP=False (override applied) but the
proposer builds the draft from vllm_config.parallel_config (EP=True), so the
override is discarded. Patch _create_draft_vllm_config to log both."""
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
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

    orig = SpecDecodeBaseProposer._create_draft_vllm_config

    def patched(self):
        cfg = orig(self)
        if rank == 0:
            sc = self.speculative_config
            tgt_ep = self.vllm_config.parallel_config.enable_expert_parallel
            draft_pc = getattr(sc, "draft_parallel_config", None)
            draft_ep = (draft_pc.enable_expert_parallel
                        if draft_pc is not None else "NONE")
            used_ep = cfg.parallel_config.enable_expert_parallel
            with open(os.path.join(OUTDIR, "cfg.txt"), "w") as f:
                f.write(
                    f"FULL_REPLICA_env={os.environ.get('VLLM_SELF_SPEC_DRAFT_FULL_REPLICA')}\n"
                    f"target_parallel_config.enable_expert_parallel = {tgt_ep}\n"
                    f"speculative.draft_parallel_config.enable_expert_parallel = {draft_ep}\n"
                    f"draft is SAME obj as target_pc: "
                    f"{draft_pc is self.vllm_config.parallel_config}\n"
                    f"_create_draft_vllm_config USED parallel_config.enable_expert_parallel = {used_ep}\n"
                    f">>> BUG if draft_ep=False but used_ep=True (override discarded)\n"
                )
        return cfg

    SpecDecodeBaseProposer._create_draft_vllm_config = patched

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
    llm.generate(["hello world"],
                 SamplingParams(temperature=0.0, max_tokens=2, seed=0),
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
