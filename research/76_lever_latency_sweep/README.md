# Phase 76 — lever × regime draft-latency sweep (the empirical cost map)

Source phases: 74 (window win / quant parity on MoE, unifying principle), 75
(dense weight-quant win, E4 MLA refutation → active-weight law), 24 (PCIe-SHM
forced-comm recipe), 17 (layer-skip acceptance negative — cost side unmapped).

## Objective

Measure, for every candidate self-spec draft lever, the **draft-step latency
ratio** `R(lever, cell) = t_step(lever) / t_step(bf16)` across the regime axes
(batch × context × architecture × fabric). This is the empirical Tq/Tp surface —
the COST half of `speedup = τ / (γ·R + 1)` (formula validated end-to-end in P75
E3 to ~10%). Accept length τ is deliberately OUT of scope here: it is mostly
lever-intrinsic, largely already measured (window ~4.75/5 @K=4, fp8 draft
β~0.95, W4 β~0.92, layer-skip collapse P17, local-routing β story P24-29), and
composes with R afterwards.

**Deliverable:** a per-lever heatmap R over (batch × ctx) for each
(model, fabric), plus the slope/intercept decomposition per cell so the Phase-77
cost model can be FIT to this data, not just motivated by it.

## Levers (draft-only candidates)

Quantization is THREE distinct sub-levers (different terms, different kernels):

| id | lever | term it cuts | knob |
|---|---|---|---|
| L0 | bf16 baseline | — | denominator of every cell |
| L1a | weight-only quant | weight-read | dense: W4A16 compressed-tensors, Marlin AND auto/Machete (kernel choice is batch-dependent, E1; P75 ckpts reusable); MoE: fp8 W8A16 Marlin (`moe_backend=marlin`) |
| L1b | weight+activation quant | MAC peak (2× fp8) + possibly EP a2a bytes | dense: fp8 W8A8 cutlass_scaled_mm (native fp8 MACs, SM90); MoE: `fp8_per_block` → FI-CUTLASS (native, P74-resolved path). NOT the TRITON per-tensor path (0.65× tax, P74) |
| L1c | KV-cache quant | KV-read | `kv_cache_dtype=fp8_e4m3` — the ONLY format on the FA3 fast path (P74) |
| L2 | sparse attention (sinks+window 512) | KV-read | FA3 native sliding-window (AOT path, `flash_attn.py:277-441`), uniform window via hf-override; cross-check vs W7 window draft attention |
| L3 | layer-skip (s = 25%, 50%) | all terms ∝ | `--hf-overrides num_hidden_layers` truncated model (cost-exact proxy; P17 measured acceptance, never cost) |
| L4 | local routing / comm-free | EP comm | `VLLM_SELF_SPEC_SKIP_A2A` (P24 B2a); defined only for EP>1 |

**L1c framing caveat (P74):** in the CURRENT harness KV-quant cannot be
draft-only (SHARED_KV → global → no ratio change by construction). In this
standalone-engine sweep its R measures what a draft-only fp8 KV pool WOULD buy
— i.e. it decides whether building the second-pool plumbing is ever worth it.
Report it with that label, never as an implemented strategy.

**L1b comm question (E0 item):** whether the W+A arm also shrinks EP a2a bytes
depends on whether the dispatch moves fp8 or bf16 on this path (the TRITON
per-tensor path moved fp8 + a scale all-gather; FI-CUTLASS block path unknown).
Determine at E0 — it decides whether L1b is also a comm lever (P23's rationale).

## Axes

- **batch:** 1, 8, 32 (+64 only on the compute-bound corner cells)
- **context:** 2k, 16k, 32k
- **architecture:** Qwen2.5-7B-Instruct (dense GQA), Qwen3-30B-A3B (MoE GQA),
  DeepSeek-V2-Lite (MoE MLA)
- **fabric/parallelism:** dense @TP1; MoE @**attention-DP4+EP4** NVLink (the
  P74/P24 fabric); MoE-GQA @DP4+EP4-PCIe-SHM (`NCCL_P2P_DISABLE=1
  NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1`, verify "via SHM/direct" in NCCL log —
  P24 3e recipe) as the comm-bound point on this box.
  **NOT pure TP4+EP4** — E0 measured that pure TP+EP selects
  `MoEPrepareAndFinalizeNoDPEPModular`: tokens are already replicated by TP, so
  there is NO dispatch collective at all — no a2a term for L4 to cut, and the
  fabric axis would be vacuous. (Deviation vs P75-E4, which used TP4+EP4; its
  within-model quant ratios remain valid, but its step times are not
  comm-comparable.) Batch semantics under DP4: batch is GLOBAL (split ~B/4 per
  rank); keep B multiples of 4.

## E0 — infrastructure-parity preflight (MANDATORY, before any sweep cell)

> **STATUS: DONE — ALL THREE GATES OPEN** (dense 9/9, MoE 9/9, MLA 6/7 + one
> legitimate broken cell: MLA×fp8-KV flips FLASH_ATTN_MLA→FLASHMLA). Findings,
> kernel-path map, and E1 consequences in `results_e0.md`. Runner:
> `scripts/e0_preflight.sh [dense|moe|mla|all]`.

The base run is FA3 + CUDA-graphed decode. A lever arm that silently switches
attention backend, MoE kernel path, or CUDA-graph mode confounds the lever with
the infrastructure delta (the P74 traps: TRITON W8A8's hidden a2a scale
all-gather; fp8_e5m2 KV silently → FlashInfer). Every ratio R must be
arm/L0 **at identical infrastructure**, asserted not assumed:

1. **Attention backend identity per arm-pair** — grep the engine log for the
   selected backend; arm and denominator must match (FA3 for dense/GQA; the MLA
   backend for V2-Lite — MLA never uses FA3, fine, ratios are within-model).
   Known traps: `fp8_e5m2`→FlashInfer, `*_per_token_head`→TRITON_ATTN,
   `nvfp4` unreachable on SM90 (P74).
2. **L2 stays on FA3** — verified in-tree: FA3 has a native AOT sliding-window
   decode path (`flash_attn.py:277-441`) requiring a UNIFORM window config →
   override every layer, then assert no fallback. Also assert the
   window+fp8-KV combination (future composed arm) keeps FA3.
3. **CUDA-graph mode identity** — assert the same cudagraph mode line
   (FULL/PIECEWISE) for arm and denominator; standalone engines should all be
   FULL-CG decode. Any arm that forces eager is a broken cell, not a data point.
4. **MoE kernel path per arm** — log which backend engaged (Marlin W8A16 /
   FI-CUTLASS block-fp8 / TRITON per-tensor). The TRITON per-tensor path is a
   documented infrastructure TAX, not a lever ceiling — exclude or report as a
   separate labeled row.
5. **L1b dispatch bytes** — determine (log/trace) whether EP a2a moves fp8 or
   bf16 under FI-CUTLASS block-fp8 (see L1b comm question above).
6. **Kernel execution check on THIS box** — cloud-9ezI3Q: Marlin+Machete OK
   (P75 E0; h106 has the PTX trap — do NOT move boxes mid-sweep). flashinfer
   0.6.12 present. `ninja` 1.13.0 in `.venv/bin` → PATH before MLA runs.
7. **L3 truncated model sanity** — assert t_step scales ~linearly in layer
   count at b32/2k (compute regime) before trusting the b1 cells.
8. **dtype/dummy-weight policy** — if dummy weights are used anywhere, note MoE
   routing skew differs from real weights (affects expert-kernel M
   distribution); use real weights for MoE arms.

E0 pass = a table (arm × model): attention backend, MoE kernel, CG mode,
engaged-kernel log line — all matching the L0 row except the intended delta.

## Matrix (pruned, ~180-200 slope runs)

- **Tier A (core factorial):** per model at native fabric, all applicable arms ×
  3 batch × 3 ctx: dense (L0, L1a×2 kernels, L1b, L1c, L2, L3×2 → ~8 arms → 72),
  MoE-GQA NVLink (L0, L1a, L1b, L1c, L2, L3, L4 → 7 arms → 63), MoE-MLA NVLink
  (no L2/L1c variants that don't apply → ~5 arms → 45). Prune within tiers:
  L1c needs the ctx axis more than the batch axis; L3's second skip fraction
  only at {b1, b32} × {2k, 32k} corners.
- **Tier B (fabric flip):** MoE-GQA @PCIe-SHM, all arms, cells {b8, b32} × 16k
  (+b32×32k if stable) → ~14. This is where L4 must flip from no-op to dominant,
  and where L1b's a2a-byte cut (if real) must show.
- Composed levers (L1+L2 etc.) deferred to a follow-up tier after the singles map.

> **E1 STATUS: dense + moe COMPLETE** (mla/pcie deferred by user). Ratio maps,
> crossovers, prediction scorecard (2 confirmed refutations → law refinements)
> in `results_sweep.md`; tables `data/e1/final_tables.md`, CSV
> `data/e1/summary.csv`.
>
> **E2 STATUS: strategy map COMPLETE** (`results_strategy_map.md`,
> `scripts/e2_strategy_map.py`). R×τ composed via the P75 formula;
> back-predicts P75-E3 1.21× and P74's window wins within 2.5-7%, and
> reproduces the γ*(ctx) law. Headline: dense = W4-Marlin region ∪ window
> region; MoE = "don't speculate" region ∪ window region.
>
> **E3 STATUS: forward-prediction spot-check PASSED** (`e3_spotcheck.sh`):
> dense b32/16k window **1.54× measured** (map 1.40×, dense β=0.97 beats the
> assumed 0.94); MoE b32/32k window **1.64× measured** (map 1.90×, 86% harness
> delivery at γ=6) — largest measured lossless single-node win at serving
> batch. Winners + depths confirmed in both never-before-measured cells.

## Method (one primitive — REVISED at E1 build; E0 forced serve-mode)

**Serve-mode TPOT** (`vllm bench serve`, mean time-per-output-token = decode-step
time; prefill excluded by construction), standalone engine, NOT the spec harness.
One server launch per arm serves ALL (batch × ctx) cells. Replaces the two-length
slope plan because MoE cells must run attention-DP4+EP4 and the offline LLM API
refuses internal DP (E0 finding 1); dense uses the same method for uniformity.
**Method cross-validated**: dense bf16 b1/2k TPOT 6.05 ms vs P75-E1 offline slope
5.86 ms (3%, analyzer gate S1). Per arm a `.meta` sidecar records the engaged
attention backend / MoE kernel / dispatch class / lever markers (E0 parity, live).
Runner: `scripts/e1_sweep.sh <dense|moe|mla|pcie> [arm-filter]`
(`E1_RUNS`, `E1_CELLS`, `E1_OUT` overrides); analyzer: `scripts/e1_analyze.py`
(TPOT + ratio tables, sanity gates S1-S4, `data/e1/summary.csv`).

## Pre-registered predictions (falsification targets)

1. L1a dense: R best at b1/short-ctx (~0.6 W4 Marlin, E1), degrades toward 1.0
   as ctx/batch grow (KV/compute take over). L1a MoE: R ≈ 1.0 everywhere
   (P74/E4); >1.0 at b1 (Marlin overhead, E4).
1b. L1b: the ONLY quant arm that can win the compute-bound corner (b32-64/2k,
   fp8 MACs 2× peak) — R < 1 there iff the GEMMs are large enough to be
   MAC-bound; on MoE the draft's per-expert M is tiny (P74: b64 ≈ 4 tok/expert,
   latency-bound) so R ≈ 1.0 unless batch is very large. On PCIe, R < 1 iff the
   dispatch moves fp8 bytes (E0 item 5).
1c. L1c: R improves with ctx (KV term grows), ~flat in batch; on dense long-ctx
   it approaches the KV-read fraction (P74: nospec +12% @16k, +23% @32k
   translated to R ≈ 0.89/0.81); weakest on MLA (KV already small).
2. L2: R improves monotonically with ctx; ~no-op at 2k; smaller effect on MLA
   than GQA (KV already 3.2× smaller).
3. L3: R sub-linear in s at b1 (fixed floor eats the saving, P72 ~2400-kernel
   floor), approaches linear at b32. The only lever whose R is
   ~architecture-independent.
4. L4: R ≈ 1.0 on NVLink (P12), large cut on PCIe-SHM growing with batch
   (f = 0.37→0.65 with batch, P24 3e).
5. Cross-cutting: no single lever has R < 0.8 in every cell — the premise of the
   strategy-selection paper.

## Guardrails (encode in runner, learned P74/75)

- Same box for ALL cells of a ratio; never compare raw tok/s across boxes.
- batch-invariant OFF, profiler OFF, CUDA-graph state matched between arm and
  its L0 denominator.
- ITERS ≥ 6, keep per-iter values, bimodality check (P74 fp8 16k stall) →
  median + flag, rel-std guardrail.
- Log which kernel engaged per cell (Marlin/Machete/FI-CUTLASS auto-pick can
  change with batch — E1 found Marlin>Machete at M=1; unknown at b32).
- `ninja` on PATH before MLA runs (E4 JIT).
- W4-asym unloadable on this vLLM (P75) — sym only; W8 Machete accept anomaly
  (P75) — avoid W8 as a lever arm.

## Decision criteria

- Sweep is a SUCCESS if the R-surface shows ≥2 crossovers (cells where the
  best lever changes), each outside noise (rel-std < 3%, ≥2× gap) — that is the
  empirical evidence the strategy-selection thesis needs.
- Any pre-registered prediction refuted → record as a law-refinement (the E4
  pattern), not a failure.
- If layer-skip cost-side cannot be implemented cleanly in ~1 day, drop L3 to a
  truncated-forward microbench and mark the cell estimated.

## Expected next artifact

`results_sweep.md` with the R heatmaps + per-cell slope/intercept CSV in
`data/`; then Phase 77 fits the term-decomposed cost model to this CSV and
composes with τ for the strategy map.
