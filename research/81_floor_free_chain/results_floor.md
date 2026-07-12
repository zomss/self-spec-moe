# Phase 81 results

## E0 — floor anatomy of the composed dense chain: GO for E1 (corrected gate)

Kineto traces of the exact 80-E3 config (W4-Marlin draft_model + window-KV,
16k), `data/trace_combo_*`, `scripts/anatomy.py`:

| | chain step | GPU-active | in-forward idle | inter-step gap | kernels/step |
|---|---|---|---|---|---|
| b8 K=4 | 5.90 ms | 3.66 (62%) | 2.23 (38%) | ~0.01 | 374 |
| **b32 K=6 (miss cell)** | **6.47 ms** | **3.94 (61%)** | **2.26 (35%)** | 0.27 | 362 |
| verify @b32 | 15.66 ms | 10.86 (69%) | 4.80 (31%) | — | 404 |

Consistency check: anatomy-implied per-token wall at b32/K6 =
(6×6.47 + 15.7 + ~0.6)/5.70 ≈ 9.7 ms ≈ measured 9.8 ms (3266 tok/s × 32) ✓.

**The README's ≥50%-floor-share gate was mis-calibrated — the correct gate is
cycle arithmetic vs the 1.75× target:**

- 1.75× ⇒ per-token ≤ 8.24 ms ⇒ cycle ≤ 47 ms ⇒ draft step ≤ 5.1 ms —
  requires recovering only **~55% of the 2.53 ms/step floor**.
- Full floor removal ⇒ draft step ≈ 3.94 ms ⇒ cycle ≈ 40 ms ⇒ **ceiling
  ≈ 1.98×** — above the 1.91× standalone roofline, because the harness
  chain's GPU-active (3.94) still exceeds the CUDA-graphed standalone step
  (3.80 total), i.e., a graphed chain may recover some active-side
  inefficiency too.
- Additional headroom not in the gate: the verify forward idles 31%
  (4.8 ms) — graphing/optimizing verify is a second dial.

**Notes for E1:**
1. Inter-step gaps are already ~zero at b8 (the LIGHT_MD/DP-coord stack did
   its job); the floor lives INSIDE the eager forward as launch idle across
   ~370 kernels — exactly what CUDA-graph capture removes.
2. Draft GPU-active (3.94 ms) is ~2× the byte theory (~2.0 ms): the eager
   kernel train is inefficient on-GPU as well — CG replay won't shrink this,
   but a fused/captured decode path might (the standalone graphed step
   suggests ~3.3-3.8 ms is reachable).
3. Ops trap encountered: the box is now DYNAMICALLY shared (another user's
   job moved across GPUs 0/6/7 mid-session; one trace arm failed on a
   transiently-occupied GPU 0 and passed on rerun) — runners should probe
   free memory at launch rather than assume the GPU set.

**E0 verdict: PROCEED to E1** (TRITON_ATTN chain-CG A/B first, accept-gate
mandatory per P35).

## E1 — chain-CG A/B: all three routes fail AS-IS; the live path is a scoped fix

Dense b32/16k, composed draft, K=6 (`scripts/run_e1.sh`, logs/e1_*):

| arm | tok/s | accept (ref 5.685) | verdict |
|---|---|---|---|
| pw (PIECEWISE baseline) | 3442 ±236 (1.56×) | 5.685 | reference (reproduces 80-E3) |
| sp (P69 window-scratchpad FULL-CG) | **952 (0.43×)** | **5.692 — bit-exact holds** | accept PASS, speed FAIL |
| tr (TRITON_ATTN + captured chain) | 527 | **1.954 — collapsed** | **P74's TRITON hypothesis REFUTED** — its decode is not replay-safe either |
| fa3cg (control: FA3 + captured) | 1138 | 1.930 — collapsed | the P35 trap on cue — **gate sensitivity validated** |

**sp anatomy** (`data/trace_sp_K6_b32`): chain step 15.1 ms, **87% GPU-ACTIVE,
950 kernels/step** (vs eager 6.5 ms / 362 kernels) — the graph is not
launch-bound, it does ~2× the GPU work. Root cause read from the kernel table
(9,555 elementwise + 2,380 gemv + 4,760 gathers per trace): the scratchpad
DOES call F.scaled_dot_product_attention, but at q_len=1 with an additive
mask torch dispatches the MEM-EFFICIENT backend → gemv/elementwise
decomposition, ~0.25 ms/layer ≈ 7-9 ms/step, where FA3's fused decode kernel
does the same work in ~30 µs. This also retro-explains P70's MoE result.

**The scoped fix (E1b)**: the P35 freeze breaks on a GROWING sequence; the
scratchpad's KV length is CONSTANT (sinks+window = 528). FA3's host-side
schedule over a fixed-shape dense scratchpad is capture-stable → swap the
scratchpad SDPA for a fixed-shape flash call (flash_attn_with_kvcache on the
dense scratchpad, seq len constant). Expected: ~30 µs/layer attention inside
ONE replayable graph → chain step ≈ 4-4.5 ms → past the 5.1 ms gate.
Accept re-validation mandatory (same A/B harness).

## E1b — the fused capture-safe chain WORKS; the wall moves to CPU orchestration

Implementation (`vllm/v1/spec_decode/scratchpad_attn.py`): the scratchpad SDPA
(mem-efficient soup) replaced by PAGED FA3 varlen over the compacted window
block table — no gather at all; constant geometry (max_seqlen_k=cap=544,
fixed table width); live lengths/pages device-read from the persistent
buffers. Fallback env `VLLM_SELF_SPEC_SCRATCHPAD_SDPA=1`. (First attempt
hit `cu_seqlens_k and seqused_k cannot be provided at the same time` — the
paged form is the correct and better call.)

| arm | tok/s | accept | chain step (trace) |
|---|---|---|---|
| pw (PIECEWISE ref) | 3442 ±236 | 5.685 | 6.47 ms (61% active) |
| sp (fused chain-CG) | 3222 ±90 | **5.655 ✓** | **3.70 ms (94% active, idle 0.21)** |
| sp0 (+ step-0 FULL-CG) | 3281 ±44 | 5.683 ✓ | — |

**Won — and it is a reusable systems law**: *FA3's captured schedule is
replay-safe iff the attention geometry is CONSTANT.* The P35 freeze needs a
growing sequence; the window scratchpad clamps geometry, so the whole chain
step replays as ONE graph — floor eliminated (6.47 → 3.70 ms, launch idle
2.26 → 0.21 ms), accept bit-preserved. This also retro-explains P69/P70:
the mechanism was right, its SDPA payload was the cost.

**Not cashed — the wall moved OFF the GPU**: e2e stays ~parity because with
the chain at ~18.5 ms/cycle, ~20 ms of the ~55 ms cycle is CPU-side
orchestration (scheduler, rejection sampler, python between forwards; the
same cost was always there, previously shadowed by the longer GPU floor).
The 1.75× gate is NOT met by chain work alone. Remaining lever = cycle-level
serving engineering (async scheduling / overlapped step-0 / batched
sampler) — the same class of work SparseSpec's "unified scheduler + delayed
verification" does; substantial engineering, separate decision.

**Delivery(γ, R) story for the paper improves either way**: the launch-floor
component of delivery is now REMOVABLE (measured), and the residual is
generic serving overhead — quantified at ~2.5 ms/token here, with the
production remedy known and citable.

## E2 — the cycle-boundary sync IS the wall (diagnosis complete)

Orchestration knob A/B on top of the fixed chain (sp0 = chain+step0 FULL-CG):

| arm | tok/s | accept | note |
|---|---|---|---|
| sp0 | 3281 ±44 | 5.683 | reference |
| + async_scheduling | 3383 ±90 | 5.661 | +3% — scheduler is not the wall |
| + local_argmax | INIT FAIL | — | `Qwen2ForCausalLM has no get_top_tokens` (P74 path is MoE-class-only; TP1 rationale moot anyway) |

**With-stack trace attribution of the copy storm** (`trace_stack_K6_b32`):
`rejection_sampler.parse_output` = **469 ms / 32 calls ≈ 14.7 ms PER CYCLE**
— single-handedly the CPU wall. Everything else is noise (window compaction
8 ms total, h2d staging ~5 ms). Mechanism: `output_token_ids.cpu()` is a
SYNCHRONOUS D2H that drains the whole GPU queue each cycle; the CPU then
parses/schedules while the GPU idles — the per-cycle serialization point.
The GPU-idle 29% and this 469 ms are the same phenomenon from two sides.

**E2b — the fix (scoped, next)**: the cycle boundary is NOT semantically
serial on the critical path — the ACCEPTED tokens already exist ON GPU (the
rejection sampler wrote them); the next cycle's step-0 can consume them
device-side, and the CPU parse (stop checks, detokenize, streaming) can LAG
one cycle behind (safe under ignore_eos + preallocated blocks; general case
needs a late-abort). This is device-side cycle chaining — the P32 overlap
idea landed at the cycle boundary, and the same design point SparseSpec's
"delayed verification" engineering occupies. Expected recovery: most of
~14.7 ms/cycle → cycle ~40 ms → ~4,500 tok/s ≈ 2.0× — the phase gate and
the paper headline in one step. Implementation surface: proposer step-0
input plumbing (GPU token feed) + deferred output processing in the
harness/runner path.

## E2b (in progress) — the wall is the RAGGED STEP-0, and the fix exists but doesn't engage

Fine-profiler (regions, VLLM_SELF_SPEC_PROFILE_OUT — note: earlier "CPU
orchestration" attribution was WRONG in detail; the region profile names it):

| region | ms/cycle (taxed) | note |
|---|---|---|
| **draft_forward_first (step-0)** | **52.9** | THE wall (~20 ms clean): ragged K+1-token re-ingest, mode=PIECEWISE, uniform=False |
| verify | 34.6 | |
| draft_forward (chain step) | 3.73 | the E1b fix holding |
| cpu_reject_parse | 0.09 | CPU_ORCH's fast parse engaged — parse SOLVED |
| everything else | ≤1.5 | |

Knob results: CPU_ORCH alone 3265 (wash — parse was only part of the
boundary); +async 3470 ±161 (≈ baseline). So the earlier "cycle-boundary CPU"
reading conflated parse (real, now fixed) with THE step-0 forward cost.

**The pre-built fix**: P67's compacted step-0
(`VLLM_SELF_SPEC_SHARED_KV_STEP0_DECODE`) — under shared KV, step-0 needs
only a q=1 decode of the appended token (accepted tokens' KV already written
by verify). P67 shelved it (saving ~2.4 ms vs accept −0.04 in THEIR config);
here it would turn a ~20 ms ragged forward into a ~3.7 ms captured chain step
→ cycle ≈ 41 ms ≈ 4,400 tok/s ≈ 1.98×.

**Measured attempt (spsd): 3196 tok/s, accept 5.108 — REGRESSION, and the
log proves the compaction NEVER ENGAGED** (decode-phase step-0 still
`uniform=False mode=PIECEWISE`, 224 tokens/27 reqs). Two findings:
1. `_step0_decode_active` requires `num_rejected_tokens_gpu is not None` —
   not plumbed on our harness path → inactive.
2. The flag STILL dropped accept by 0.57 while inactive — a side effect
   (something else keys on `_shared_kv_step0_decode`); bug to isolate.

**Next (exact target)**: (a) plumb `num_rejected_tokens_gpu` through the
sync/W7 propose path (it exists GPU-side from the rejection sampler);
(b) isolate + fix the inactive-flag accept side effect; (c) route the
compacted q=1 step-0 through the SAME scratchpad FULL graph as chain steps
(it is exactly a chain step); (d) accept gate ≥ 5.6 and the 1.75× check.
