#!/usr/bin/env python3
"""Activation-quant (communication-axis) draft acceptance vs bf16 target.

Phase 22 measured weight-only (W4A16) FP4 expert drafts -> 0.92. But the EP
all-to-all moves *activations*, so quantizing the dispatched/combined activations
shrinks the communication directly (A8 -> 2x less, A4 -> 4x less). In speculative
decoding the draft can use activation precisions too lossy to serve directly,
because the bf16 verify corrects it. This measures the acceptance cost of that.

We fake-quantize the two activations that cross the wire in Qwen3MoeExperts:
  dispatch = the per-token hidden state entering the experts
  combine  = each expert's output after down_proj (before the weighted sum)
per-token (dynamic), combined with bf16 or NVFP4 weights. W4 = NVFP4 weights
(Phase 22's best). Acceptance = rejection sampling vs the exact bf16 target.

Configs (weight x activation):
  w16a8   : bf16 weights, FP8 activations        (isolated 2x-comm cost)
  w16a4   : bf16 weights, NVFP4 activations       (isolated 4x-comm cost)
  w4a16   : NVFP4 weights, bf16 activations        (Phase 22 anchor ~0.92)
  w4a8    : NVFP4 weights, FP8 activations          (realistic 2x comm)
  w4a4nv  : NVFP4 weights, NVFP4 activations          (aggressive 4x comm)
  w4a4mx  : NVFP4 weights, MXFP4 activations            (4x comm, coarser act)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_moe import modeling_qwen3_moe as qwen3_moe

PROMPTS = [
    "The future of artificial intelligence is",
    "Summarize why distributed inference needs communication.",
    "Describe how a mixture-of-experts layer routes tokens.",
    "Hi! Can you recommend a good book for a long flight?",
    "I'm feeling stressed about work. Any advice?",
    "What's a fun fact about the ocean?",
    "Write a Python function to compute the nth Fibonacci number.",
    "Implement binary search over a sorted list in Python.",
    "Explain what a race condition is and how to avoid it.",
    "Solve step by step: if x + 3 = 7, what is x?",
    "What is the derivative of x^2 + 3x with respect to x?",
    "A train travels 60 km in 45 minutes. What is its speed in km/h?",
    "Explain the theory of relativity in one paragraph.",
    "List the steps to bake sourdough bread.",
    "Compare TCP and UDP for video streaming.",
    "Write a haiku about autumn leaves.",
]

FP4_MAGS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_GRID = {}              # device -> (grid, mids)
_ACT = {"scheme": "none"}  # read by the patched experts forward

# config table: label -> (weight_scheme, activation_scheme)
CONFIGS = [
    ("w16a8", "bf16", "a8"),
    ("w16a4", "bf16", "a4nv"),
    ("w4a16", "nvfp4", "none"),
    ("w4a8", "nvfp4", "a8"),
    ("w4a4nv", "nvfp4", "a4nv"),
    ("w4a4mx", "nvfp4", "a4mx"),
]


def grid_for(dev):
    if dev not in _GRID:
        g = torch.tensor(FP4_MAGS, device=dev)
        _GRID[dev] = (g, (g[1:] + g[:-1]) / 2.0)
    return _GRID[dev]


def fp4_qdq(w, group_size, scale_mode):
    grid, mids = grid_for(w.device)
    last = w.shape[-1]
    g = group_size if last % group_size == 0 else last
    wf = w.float().reshape(-1, g)
    absmax = wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = absmax / 6.0
    if scale_mode == "pow2":
        scale = torch.pow(2.0, torch.round(torch.log2(scale)))
    elif scale_mode == "e4m3":
        scale = scale.to(torch.float8_e4m3fn).float().clamp_min(1e-8)
    sign = torch.sign(wf)
    ax = (wf / scale).abs().clamp(max=6.0)
    idx = torch.bucketize(ax, mids)
    q = sign * grid[idx]
    return (q * scale).reshape(w.shape).to(w.dtype)


def fp8_perrow(x):
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = amax / 448.0
    q = torch.clamp(x.float() / scale, -448.0, 448.0).to(torch.float8_e4m3fn).float()
    return (q * scale).to(x.dtype)


def nvfp4_weight(w):
    return fp4_qdq(w, 16, "e4m3")


def act_quant(x, scheme):
    if scheme == "a8":
        return fp8_perrow(x)
    if scheme == "a4nv":
        return fp4_qdq(x, 16, "e4m3")
    if scheme == "a4mx":
        return fp4_qdq(x, 32, "pow2")
    return x


def patched_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Mirror of Qwen3MoeExperts.forward with activation quant on the
    dispatched input and the combined per-expert output."""
    scheme = _ACT["scheme"]
    if scheme != "none":
        hidden_states = act_quant(hidden_states, scheme)  # dispatch
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        chs = self.act_fn(gate) * up
        chs = F.linear(chs, self.down_proj[expert_idx])
        if scheme != "none":
            chs = act_quant(chs, scheme)  # combine
        chs = chs * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, chs.to(final_hidden_states.dtype))
    return final_hidden_states


def accept(p_logits, q_logits, n, gen):
    p = torch.softmax(p_logits, -1)
    q = torch.softmax(q_logits, -1)
    overlap = float(torch.minimum(p, q).sum())
    d = torch.multinomial(q, n, replacement=True, generator=gen)
    acc = torch.minimum(torch.ones_like(p[d]), p[d] / q[d].clamp_min(1e-30))
    draws = torch.rand(n, generator=gen)
    return overlap, float((draws < acc).float().mean()), float(p.argmax() == q.argmax())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sample-count", type=int, default=512)
    ap.add_argument("--pos-from", type=int, default=4)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--local-files-only", action="store_true")
    a = ap.parse_args()

    AutoConfig.from_pretrained(a.model, local_files_only=a.local_files_only)
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=a.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=a.local_files_only,
    ).to(a.device).eval()
    layers = model.model.layers
    print(f"[cfg] {a.model} layers={len(layers)}")

    # patch experts forward once (no-op when _ACT['scheme']=='none')
    qwen3_moe.Qwen3MoeExperts.forward = patched_experts_forward

    # snapshot original expert weights for restore
    expert_w = []
    for layer in layers:
        if hasattr(layer.mlp, "experts"):
            for p in layer.mlp.experts.parameters():
                if p.dim() >= 2:
                    expert_w.append(p)
    orig_cpu = [p.data.detach().to("cpu", copy=True) for p in expert_w]
    print(f"[snap] {len(expert_w)} expert weight tensors")

    # bf16 target logits (no act quant, original weights)
    _ACT["scheme"] = "none"
    enc_cache, target = [], []
    for prompt in PROMPTS:
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=64).to(a.device)
        enc_cache.append(enc)
        with torch.inference_mode():
            lg = model(**enc, use_cache=False).logits[0].float()
        T = lg.shape[0]
        pos = list(range(a.pos_from, T))
        target.append((pos, lg[pos].cpu()))
    n_pos = sum(len(p) for p, _ in target)
    print(f"[target] cached {n_pos} positions")

    gen = torch.Generator().manual_seed(0)

    def measure():
        ov, sa, t1 = [], [], []
        for enc, (pos, p_cpu) in zip(enc_cache, target):
            with torch.inference_mode():
                q = model(**enc, use_cache=False).logits[0].float().cpu()
            for j, t in enumerate(pos):
                o, s, m = accept(p_cpu[j], q[t], a.sample_count, gen)
                ov.append(o); sa.append(s); t1.append(m)
        return (round(float(np.mean(ov)), 4), round(float(np.mean(sa)), 4),
                round(float(np.mean(t1)), 4))

    results = {}
    for label, w_scheme, a_scheme in CONFIGS:
        with torch.inference_mode():
            for p, o in zip(expert_w, orig_cpu):
                p.data.copy_(o.to(a.device))
                if w_scheme == "nvfp4":
                    p.data = nvfp4_weight(p.data)
        _ACT["scheme"] = a_scheme
        ov, sa, t1 = measure()
        _ACT["scheme"] = "none"
        results[label] = {"weight": w_scheme, "act": a_scheme,
                          "expected": ov, "sampled": sa, "top1": t1}
        print(f"{label:>8} (W={w_scheme},A={a_scheme}): "
              f"sampled={sa:.4f} expected={ov:.4f} top1={t1:.4f}")

    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps(
        {"model": a.model, "n_positions": n_pos, "sample_count": a.sample_count,
         "results": results}, indent=2))
    print(f"[saved] {a.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
