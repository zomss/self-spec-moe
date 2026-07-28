#!/usr/bin/env python3
"""Phase 93 Step 1b: which kernel ACTUALLY executes per quant format.

Boots the quantized ckpt as the (only) model, walks the loaded modules
worker-side, and reports the quant-method/kernel class per layer kind
(attn proj, dense MLP, MoE expert, router). This is the per-cell
kernel record for the efficiency defense -- authoritative because it
inspects the loaded model, not a synthetic GEMM shape.

env: KM_MODEL (ckpt path or HF id), KM_TP (1), KM_OUT (append jsonl)
"""
import json
import os
from pathlib import Path

MODEL = os.path.expanduser(os.environ["KM_MODEL"])
TP = int(os.environ.get("KM_TP", "1"))
OUT = os.environ.get("KM_OUT", str(
    Path(__file__).resolve().parents[1] / "data" / "kernel_matrix.jsonl"))


def rpc_walk(worker):
    """Collect quant scheme / kernel class names per module kind."""
    kinds = {}
    model = worker.model_runner.model
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        detail = ""
        qm = getattr(mod, "quant_method", None)
        if qm is not None:
            detail = type(qm).__name__
            # compressed-tensors: the scheme (and its kernel) hang off
            # the LAYER, not the method; probe both places
            for holder in (qm, mod):
                sch = getattr(holder, "scheme", None)
                if sch is not None:
                    detail += f"/{type(sch).__name__}"
                    k = getattr(sch, "kernel", None)
                    if k is not None:
                        detail += f"/{type(k).__name__}"
                    break
            k = getattr(qm, "kernel", None)
            if k is not None:
                detail += f"/{type(k).__name__}"
        if not detail:
            continue
        if "experts" in name:
            kind = "moe_experts"
        elif ".gate" in name and "proj" not in name:
            kind = "router_gate"
        elif any(p in name for p in ("q_proj", "k_proj", "v_proj",
                                     "qkv_proj", "o_proj", "kv_a", "kv_b",
                                     "q_a", "q_b")):
            kind = "attn_proj"
        elif "lm_head" in name:
            kind = "lm_head"
        else:
            kind = "mlp"
        kinds.setdefault(kind, {}).setdefault(detail, 0)
        kinds[kind][detail] += 1
    return kinds


def main():
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    from vllm import LLM
    llm = LLM(model=MODEL, tensor_parallel_size=TP, max_model_len=2048,
              gpu_memory_utilization=0.85, enforce_eager=True,
              trust_remote_code=True)
    kinds = llm.collective_rpc(rpc_walk)[0]
    row = {"model": MODEL, "tp": TP, "kinds": kinds}
    print(json.dumps(row, indent=1))
    Path(OUT).parent.mkdir(exist_ok=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    print("[KM] appended ->", OUT)


if __name__ == "__main__":
    main()
