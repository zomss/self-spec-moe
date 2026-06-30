# 37 — torch.compile numerical-consistency fix for self-spec MoE drafts

**Source phase:** `35_gqa_fullcg_debug` (GQA FULL-CG attn capture) and
`36_dp_accept` (isolated the compile-vs-eager MoE divergence: Qwen1.5-MoE DP1 K=4
compiled accept 1.57 vs eager 4.78). That phase only had a partial mitigation
(`VLLM_SELF_SPEC_DRAFT_EAGER`, recovers ~1.6->1.9). This phase finds the true
root cause and a fix that keeps the draft COMPILED (fast).

**Objective.** Make the COMPILED self-spec draft numerically match the COMPILED
verify so the draft's greedy tokens are accepted, recovering Qwen1.5-MoE DP1 K=4
accept ~4.78 with the draft still compiled/cudagraphed under FULL-CG.

**Result.** Root cause = batch-shape-dependent kernel numerics (draft runs at
decode shape ~1 tok/seq, verify at the larger verify shape; default cuBLAS
split-k / shape-tiled Triton MoE / FA split scheduling are batch-variant). Fix =
new env `VLLM_SELF_SPEC_COMPILE_CONSISTENT=1` (default off) auto-enables vLLM's
batch-invariant numerics for self-spec draft_model -> draft==verify with BOTH
compiled. DP1 accept 1.57 -> **5.0** (perfect), draft stays compiled+cudagraphed.

See `results_W7_compile_consistency.md`.
