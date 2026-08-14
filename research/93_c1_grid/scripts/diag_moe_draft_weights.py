#!/usr/bin/env python3
"""Diag: are the quant-MoE DRAFT's loaded (post-repack) weights identical
to the same ckpt loaded as TARGET?

Boot A: MoE bf16 target + W4 draft (spec engine)  -> checksums(drafter)
Boot B: W4 ckpt as target                          -> checksums(model)
Deterministic load+repack => sums must match bit-exact. Mismatch =>
draft LOADER bug; match => runtime method/forward-context bug.
"""
import os

import torch

CKPT = os.path.expanduser("/data/smcho/ckpts/Qwen3-30B-A3B-W4A16-INT4-sym")
TP = 2

PICK = ("layers.1.mlp.experts", "layers.1.self_attn.qkv_proj",
        "layers.1.mlp.gate", "layers.2.mlp.experts")


def _sums(model):
    out = {}
    for name, p in model.named_parameters():
        if any(k in name for k in PICK):
            t = p.detach()
            out[name] = (tuple(t.shape), str(t.dtype),
                         float(t.float().sum().item()),
                         float(t.float().abs().sum().item()))
    for name, b in model.named_buffers():
        if any(k in name for k in PICK):
            t = b.detach()
            out["BUF:" + name] = (tuple(t.shape), str(t.dtype),
                                  float(t.float().sum().item()),
                                  float(t.float().abs().sum().item()))
    return out


def rpc_draft_sums(worker):
    return _sums(worker.model_runner.drafter.model)


def rpc_target_sums(worker):
    return _sums(worker.model_runner.model)


def main():
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    from vllm import LLM

    spec = {"method": "draft_model", "model": CKPT,
            "num_speculative_tokens": 2,
            "draft_tensor_parallel_size": TP}
    llm = LLM(model="Qwen/Qwen3-30B-A3B", speculative_config=spec,
              tensor_parallel_size=TP, max_model_len=2048,
              gpu_memory_utilization=0.90, enforce_eager=True)
    a = llm.collective_rpc(rpc_draft_sums)[0]
    del llm
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    llm2 = LLM(model=CKPT, tensor_parallel_size=TP, max_model_len=2048,
               gpu_memory_utilization=0.90, enforce_eager=True)
    b = llm2.collective_rpc(rpc_target_sums)[0]

    keys = sorted(set(a) | set(b))
    n_bad = 0
    for k in keys:
        va, vb = a.get(k), b.get(k)
        if va is None or vb is None:
            print(f"ONLY-ONE {k}: draft={va is not None} target={vb is not None}")
            n_bad += 1
        elif va != vb:
            print(f"MISMATCH {k}:\n  draft ={va}\n  target={vb}")
            n_bad += 1
    print(f"[diag] {len(keys)} tensors compared, {n_bad} mismatches")


if __name__ == "__main__":
    main()
