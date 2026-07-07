# Phase 71 — MoE-under-cudagraph investigation (read-only)

Source: Phase 70 (`research/70_fullcg_coverage/results.md`). P70's honest
conclusion was correct that the draft-chain FULL cudagraph does not win at
16k/W512/cap544/b8, but its *leading hypothesis* for WHY —

> "the CUDA-graph (FULL) MoE must be fixed-shape, so it cannot use the eager
> path's sparse (active-experts-only) routing and pays much more weight traffic
> per token" (results.md:110-116)

— was challenged by a reviewer: vLLM's ordinary no-spec/verify decode also runs
under FULL cudagraphs with sparse MoE and is fast, so "CUDA graph ⇒ dense MoE"
cannot be vLLM's design.

## Objective

Adversarially test the P70 hypothesis against the vLLM MoE kernel code and the
P69/P70/P65 profiler data. Decide whether "full-CG forces dense MoE" is TRUE,
FALSE, or MISATTRIBUTED, and identify the real mechanism behind the 25.2 ms
draft forward.

## Method

Pure code + data reading (no GPU runs). Trace: (1) the fused-MoE Triton kernel's
expert-weight selection and its cudagraph behavior; (2) the FP8 full-replica
local-route draft MoE dispatch (which experts impl it selects); (3) what the
`draft_forward` profiler region actually encloses; (4) re-attribute the 25.2 ms
against a weight-traffic arithmetic ceiling and the P65 piecewise number.

Result: `FINDINGS.md`. No `vllm/` behavior changed.
