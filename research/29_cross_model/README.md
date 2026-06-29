# Phase 29: Cross-model validation (Phases 26-28 are all Qwen3)

**Source:** Phases 26-28 (accept-length frontier, tree drafting, dynamic cache / batched
memory) -- all measured on Qwen3-30B only. This phase checks whether the core enablers
generalize across architectures.

**Objective:** measure the routing-skew enablers on other MoE families:
- union(B): distinct routed experts B concurrent requests need (the per-device resident
  requirement for the comm-free draft) vs batch.
- global-hot top-C coverage: how well a fixed batch-independent cache covers routing
  (skip-cold beta proxy / globally-hot cache viability).

**Models:** DeepSeek-V2-Lite (deepseek_v2; 64 routed top-6 + 2 SHARED) and GPT-OSS-20B
(gpt_oss; 32 experts top-4, no shared, MXFP4-dequantized) vs Qwen3-30B (128 top-8, no
shared).

**Extraction:** routed-expert selection per token per layer via forward hooks; DeepSeek's
gate is a Linear called functionally (`F.linear(x, gate.weight)`) so its MoE block is
pre-hooked and routing recomputed (greedy softmax top_k == top_k of logits, verified from
config: topk_method=greedy, n_group=1). GPT-OSS via router forward-hook. GPT-OSS loaded
with `Mxfp4Config(dequantize=True)` (needs accelerate).

**Commands:** `python cross_model_skew.py --model <id> --output-json data/<name>.json`

**Scope:** validates the SKEW enablers (union, coverage) cross-model. The beta refinements
(skip-cold renorm recovery, dynamic last-token cache, depth decay) are Qwen3-validated and
expected to generalize qualitatively; per-model beta needs the per-model local-routing
patch (future).
