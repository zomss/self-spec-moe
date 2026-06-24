#!/usr/bin/env python3
"""Extract real router logits from a Hugging Face MoE checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts-jsonl", type=Path)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-prompts", type=int, default=16)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_prompts(path: Path | None, max_prompts: int) -> list[str]:
    if path is None:
        prompts = [
            "The future of artificial intelligence is",
            "Explain mixture of experts models in simple terms.",
            "Write a short Python function to add two numbers.",
            "What is the capital of France?",
            "Solve: if x + 3 = 7, what is x?",
            "Summarize why distributed inference needs communication.",
            "List three benefits of expert parallelism.",
            "Translate 'hello world' to Korean.",
        ]
        return prompts[:max_prompts]

    prompts = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            prompts.append(record["prompt"] if isinstance(record, dict) else str(record))
            if len(prompts) >= max_prompts:
                break
    return prompts


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def normalize_router_logits(router_logits) -> np.ndarray:
    if router_logits is None:
        raise ValueError("model output did not include router_logits")
    tensors = list(router_logits) if isinstance(router_logits, (tuple, list)) else [router_logits]
    arrays = []
    for tensor in tensors:
        if tensor is None:
            continue
        # HF MoE router logits are usually [tokens, experts] per sparse layer.
        arrays.append(tensor.detach().float().cpu().numpy())
    if not arrays:
        raise ValueError("router_logits was empty")
    min_tokens = min(array.shape[0] for array in arrays)
    arrays = [array[:min_tokens] for array in arrays]
    return np.stack(arrays, axis=0)


def find_router_tensor(output, num_experts: int) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        if output.ndim >= 2 and output.shape[-1] == num_experts:
            return output
        return None
    if isinstance(output, (tuple, list)):
        for item in output:
            found = find_router_tensor(item, num_experts)
            if found is not None:
                return found
    if isinstance(output, dict):
        for item in output.values():
            found = find_router_tensor(item, num_experts)
            if found is not None:
                return found
    return None


def install_router_hooks(model, num_experts: int):
    captured: list[torch.Tensor] = []
    handles = []

    def hook(_module, _inputs, output):
        router_tensor = find_router_tensor(output, num_experts)
        if router_tensor is not None:
            captured.append(router_tensor.detach().float().cpu())

    for name, module in model.named_modules():
        lowered = name.lower()
        if (
            lowered.endswith("gate")
            or lowered.endswith("router")
            or "gating" in lowered
        ):
            handles.append(module.register_forward_hook(hook))
    return captured, handles


def main() -> int:
    args = parse_args()
    prompts = load_prompts(args.prompts_jsonl, args.max_prompts)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_from_name(args.dtype),
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
    )
    model.to(args.device)
    model.eval()
    num_experts = (
        getattr(model.config, "num_experts", None)
        or getattr(model.config, "num_local_experts", None)
    )
    if num_experts is None:
        raise ValueError("Could not infer num_experts from model config")
    captured_router_logits, hook_handles = install_router_hooks(model, num_experts)

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    ).to(args.device)

    with torch.inference_mode():
        outputs = model(
            **encoded,
            use_cache=False,
            output_router_logits=True,
            logits_to_keep=1,
        )
    try:
        router_logits = normalize_router_logits(outputs.router_logits)
    except ValueError:
        if not captured_router_logits:
            raise
        router_logits = normalize_router_logits(captured_router_logits)
    finally:
        for handle in hook_handles:
            handle.remove()

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        router_logits=router_logits,
        prompts=np.array(prompts, dtype=object),
    )

    summary = {
        "model": args.model,
        "num_prompts": len(prompts),
        "router_logits_shape": list(router_logits.shape),
        "device": args.device,
        "dtype": args.dtype,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
