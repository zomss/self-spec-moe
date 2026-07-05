# Phase 58: literature_restart — re-grounding the research in the literature

**Why restart.** Phases 00-57 established, on real 2-node hardware, that on
comm-bound MoE-EP: (i) draft-side cleverness cannot win at serving batch
(the comm-free MoE draft floors at ~0.6x a forward -- backbone-dominated,
Phase 46; quant/local-routing cheapen only the expert third); (ii) lossless
verify-side comm reduction is IMPOSSIBLE (the (k+1)-token verify all-to-all is
irreducible -- to know which tokens to skip you need the forward that routes
them); (iii) spec decoding is a latency-regime tool (measured 1.4x @ b8, LOSS
@ serving b64, Phase 57). The remaining hope: reduce comm cost at BOTH draft
AND verify, possibly with new mechanisms the field has developed. So we
re-read the literature with a sharp cost lens before committing a direction.

**The lens (read every method for these).**
- DRAFT cost: compute FLOPs, memory-I/O (weight reads), communication (EP a2a).
- VERIFY cost: same three axes; plus the (k+1)-token-volume factor on MoE-EP.
- Regime: does it help at SERVING batch (throughput) or only low batch
  (latency)? Distributed / multi-node / EP-aware?

**Two candidate directions to sharpen (the user's framing).**
- **(A) Cheap draft** -- reduce the draft's comm AND/OR compute AND/OR
  memory-I/O. NOT MoE-specific (layer-skip, early-exit, quantized self-draft,
  small heads all live here). Question: is there a draft that is cheap on ALL
  three axes, or a comm-specific draft trick unexplored?
- **(B) Cheap verify** -- reduce the verify's cost. On MoE-EP the (k+1) a2a is
  the wall; is there a lossless or bounded-lossy verify the field has that we
  missed (partial-EP verify, hierarchical/cascade verify, self-verify)?

## Step 1 -- literature review (3 areas, parallel, real papers only)
1. MoE inference acceleration: EP all-to-all reduction (DeepEP/DBO/overlap),
   quantization, expert offload/caching, comm-bound serving systems.
2. Speculative decoding (general): EAGLE(1/2/3)/Medusa/MTP/tree(Sequoia)/
   lookahead; batched/serving/high-throughput spec decode; verify-cost work.
3. Self-speculative decoding: layer-skip/early-exit/Draft&Verify, self-draft
   heads, quantized self-draft (QuantSpec), draft-free; the cheap-draft axis.

## Step 2 -- synthesize into directions A and B
Map each surveyed method onto the cost lens; identify what is DONE vs OPEN for
A (cheap draft) and B (cheap verify), especially anything comm-aware or
serving-batch-positive. Output: `directions.md` with the sharpened A/B and the
specific open problem(s) each could target.

**Artifacts.** `litreview_moe.md`, `litreview_specdec.md`, `litreview_selfspec.md`,
`directions.md`.
