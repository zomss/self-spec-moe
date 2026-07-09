# Phase 74 results

## HEADLINE (final): K-tuned window self-spec WINS 1.16-1.34x across 16k-32k (lossless, training-free)

Clean measurement (batch-invariant OFF, profiler OFF), bf16 **window** self-draft,
b8, tok/s vs no-spec:

| ctx | K=2 | K=3 | K=4 | no-spec | best |
|---:|---:|---:|---:|---:|---|
| 16k | **693 (1.16x)** | 639 (1.07x) | 524 (0.88x) | 599 | K*=2 -> 1.16x |
| 32k | 435 (1.00x) | **586 (1.34x)** | 493 (1.13x) | 436 | K*=3 -> 1.34x |

**The win grows with context; the optimal speculation depth K\* increases with
context** (2 at 16k -> 3 at 32k) -- the verify-scaling law made concrete: a more
memory-bound verify (longer ctx) makes its K+1 tokens freer, so more draft steps
pay off. Draft D/V ~= 0.56 (kineto). Lossless (rejection sampling), training-free
(bf16 sparse-attention self-draft). The "loses ~2x" below was FOUR stacked
non-method issues: (1) batch-invariant kernels tax the spec path ~2x
(COMPILE_CONSISTENT); (2) profiler syncs tax it ~25% (VLLM_SELF_SPEC_PROFILE);
(3) K fixed at 4 (too deep for 16k -> tune K to context); (4) fp8 per-tensor
fallback + "between-forward overhead" were red herrings (trace: gap overhead ~4%).
Caveats: b8, single-node NVLink, ~3-5% slope noise on some cells; higher batch
and the exact K*(ctx) curve still to map.

## QUANT OPTIMIZATION (pre-experiment): weight-quant ceiling = Marlin, and it's PARITY not a win

> **CORRECTION (later, agent-verified): Marlin was NOT the fp8 ceiling.** All the fp8
> below used per-tensor `quantization=fp8`, which routes to the taxed Triton (0.65x)
> or dequant Marlin (parity, bf16 MACs). The **native fp8 tensor-core** path was never
> tested: **`quantization=fp8_per_block`** quantizes bf16→128-block-fp8 ON-THE-FLY at
> load (no checkpoint) and the oracle auto-selects **FlashInfer-CUTLASS** for Hopper
> block-fp8 EP>1 (`oracle/fp8.py:106-115`; `flashinfer_cutlass_moe.py:168-175` allows
> `kFp8Static128BlockSym` at SM90). It does fp8×fp8 MACs (no dequant tax) AND the block
> scheme avoids the per-tensor a2a scale-gather that crippled Triton. flashinfer 0.6.12
> is ALREADY installed — **no DeepGEMM, no checkpoint needed.** So the weight-quant BAR
> is `fp8_per_block`→FI-CUTLASS, UNMEASURED, and could turn parity→modest win at low
> batch. `fp8_per_channel` (llm-compressor FP8_DYNAMIC shape) is a second native option.
> DeepGEMM is redundant. A calibrated block-fp8 checkpoint only marginally improves
> draft ACCEPT (better scales) + load time. `--quantization fp8` = per-tensor = the
> taxed path (what "Marlin ceiling" below measured). NEXT (when GPUs free): measure
> `W7_DRAFT_QUANT=fp8_per_block` at 16k/32k b8 vs bf16 base. The rows below stand as
> the *per-tensor* fp8 story; they are NOT the fp8 ceiling.

Before running the quant-only K×context map, optimize each quant draft strategy to
its max. **Weight-quant (fp8) is now done.** All @ 16k, b8, K=2, no window, clean
(batch-invariant OFF, profiler OFF). Refs: no-spec=599, bf16 base=592, window=693.

| fp8 draft config | tok/s | vs bf16 (592) | note |
|---|---:|---:|---|
| auto → **TRITON W8A8** | 383 | 0.65x | crippled: activation-quant kernel + extra per-rank a2a scale all-gather |
| `moe_backend=marlin` (weight-only W8A16) | 584–606 | ~parity | halves expert weight read; removes both TRITON taxes |
| + `VLLM_TEST_FORCE_FP8_MARLIN=1` (dense qkv/o also weight-only) | 583 | ~parity | **no-op** vs Marlin-MoE-only |

**Findings (agent-verified, `oracle/fp8.py`):**
1. Auto picks TRITON (first backend passing per-tensor on-the-fly fp8); it is
   **W8A8** → runs `segmented_max_reduction` activation quant AND rides an extra
   per-rank scale all-gather in the EP dispatch (`naive_dp_ep.py:58-71`). That a2a
   growth is the real tax, not the fp8 weight read.
2. `moe_backend=marlin` = **weight-only W8A16** (`oracle/fp8.py:511-522`): fp8
   weights dequant→bf16 GEMM against bf16 activations → **no** activation quant,
   **no** scale collective (a2a back to bf16-baseline size). This is the fp8 fix.
3. The Rank-2 dense-linear lever (`VLLM_TEST_FORCE_FP8_MARLIN=1`, also flips qkv/o
   to Marlin W8A16; safe because target is bf16) is a **no-op**: 583 vs 584. In
   30B-A3B the experts are ~95% of the weight read; the dense projections are noise.

**Conclusion:** Marlin weight-only fp8 is the H100-no-DeepGEMM ceiling, and it lands
at **bf16 parity, not a win.** Reason = machine balance: the b8/long-context draft is
latency/KV-bound, so halving the weight bytes (which Marlin does) doesn't convert to
throughput — weight-read isn't the binding constraint. This is the *same* reason
**window wins** (693, 1.16x): window attacks the **KV read**, the binding axis.
Beating bf16 via weight-quant needs native fp8 tensor cores (Blackwell / DeepGEMM
block-fp8, unavailable) or a bigger read cut like INT4 (needs a pre-quantized ckpt).

**KV-quant optimization (agent-verified kernel path):** the optimized config is
**`kv_cache_dtype=fp8_e4m3`** — the ONLY fp8 KV format that stays on the **FA3 fast
path** (native fp8 read: zero-copy `.view()`→float8_e4m3fn, on-chip descale, no
dequant-to-bf16; `flash_attn.py:186-193, 787-790, 870-895`). Traps that silently
fall OFF FA3 to a slower kernel: `fp8_e5m2` (→ FlashInfer), `*_per_token_head`
(→ TRITON_ATTN), `turboquant_*` (→ Triton). `nvfp4` is SM100-only (unreachable on
H100). Two structural caveats:
- **Global, not draft-only.** `kv_cache_dtype` is a single `CacheConfig` field with
  no per-draft override; under `SHARED_KV=1` the draft aliases the target's KV
  tensors (`llm_base_proposer.py:3304-3349`) and a dtype mismatch is rejected
  (`:3336`). So fp8 KV quantizes the pool **verify** reads too → **bounded-lossy**
  (lossless only w.r.t. the fp8-KV model), NOT a lossless draft-only lever like
  weight-quant. A true draft-only fp8 KV would need new code (2nd pool + per-step
  quantize-copy of the prefix) — doesn't exist.
- **Speeds the no-spec baseline too.** Because it's global, fp8 KV also cuts the
  verify-only (no-spec) KV read → the honest comparison is spec-under-fp8KV vs
  **nospec-under-fp8KV**, not vs nospec-under-bf16KV. (Contrast window, which cuts
  only the DRAFT's KV read; verify keeps full bf16 KV.)
- **Scales default to 1.0** (no calibrated checkpoint for Qwen3-30B-A3B;
  `attention.py:107-109`) → e4m3 with ~2 sig-digits; on-the-fly `--calculate-kv-scales`
  is deprecated (one-shot abs-max, latches off). Accuracy risk is real but bounded.

**KV-quant measurement (16k, b8, K=2, no window; FA3 confirmed engaged, no fallback):**

| arm | KV dtype | tok/s | note |
|---|---|---:|---|
| nospec | bf16 (ref) | 599 | |
| nospec | **fp8_e4m3** | **672 (+12%)** | fp8 KV speeds verify-only baseline |
| spec | bf16 (ref) | 592 | ~parity vs nospec (0.99x) |
| spec | **fp8_e4m3** | **638 (+8%)** | accept 2.994/3 (near-perfect: draft=verify) |

Honest comparison (both under fp8 KV): **spec 638 vs nospec 672 = 0.95x — spec LOSES.**
fp8 KV *worsened* the spec/nospec ratio (0.99x → 0.95x) because nospec gains more
(+12%) than the spec draft (+8%): the spec path has more fixed/plumbing cost that
fp8 KV doesn't touch, so its variable-cost (KV-read) saving is diluted.

## THE UNIFYING PRINCIPLE (why quant-alone doesn't win, but window does)

A self-spec win requires making the **draft cheaper than verify on the BINDING cost
axis — and RELATIVE to verify** (draft-only, so the ratio moves). Each lever:

| lever | draft-only? | cost axis | binding at b8/long-ctx? | ratio moves? | result |
|---|---|---|---|---|---|
| **window** (sparse-attn) | ✅ (verify keeps full KV) | **KV read** | ✅ yes | ✅ **yes** | **WIN 1.16x** |
| weight-quant (marlin fp8) | ✅ (verify keeps bf16 W) | weight read | ❌ no (not binding) | ✅ but wrong axis | parity |
| KV-quant (fp8_e4m3) | ❌ **global** (SHARED_KV) | KV read | ✅ yes | ❌ **no** (scales both) | 0.95x (loses) |

- **Window** is special: the ONLY lever that is BOTH draft-only AND on the binding
  (KV) axis → it changes the draft/verify ratio in the draft's favor → win.
- **Weight-quant** is draft-only but on the WRONG axis (weight-read isn't binding at
  b8) → no effective ratio change → parity. (Would win only if weight-read were
  binding, e.g. very high batch, or with a bigger cut like INT4.)
- **KV-quant** is on the RIGHT axis but CAN'T be draft-only (global under SHARED_KV)
  → it's a system-wide multiplier that lifts nospec and spec together → no ratio
  change → no win (and it's bounded-lossy for nothing). To make it a win you'd need
  a draft-only fp8 KV pool (unsupported; needs new code) OR pair it with window (fp8
  KV lifts the whole system, window supplies the ratio) — but that's window again.

**Bottom line for the quant-only direction:** neither weight-quant nor KV-quant
alone produces a self-spec win at b8. Quant helps the *system* (fp8 KV: +12%
everywhere) but not the *speculation ratio*. The win lever remains window (draft-only
KV reduction). Quant's role is orthogonal/complementary (a global multiplier), not a
self-spec driver at low batch.

### 32k confirmation — escape hatch REVERSED (KV-quant loses MORE at long ctx)

Hypothesis was: at 32k the draft is more KV-bound (less fixed-overhead-bound), so
global fp8 KV might stop diluting and help the spec ratio. **It did the opposite.**
All 32k, b8, no window, clean:

| arm | KV dtype | K | tok/s | accept |
|---|---|---:|---:|---:|
| nospec | bf16 | — | 458 | — |
| nospec | **fp8_e4m3** | — | **565 (+23%)** | — |
| spec | fp8_e4m3 | 2 | 389 | 2.986 |
| spec | fp8_e4m3 | 3 (K*) | **504** | 3.974 |

Honest comparison (both fp8 KV): best spec **504 vs nospec 565 = 0.89x** — spec loses
*more* at 32k than 16k (0.95x → 0.89x). fp8 KV helped **nospec** more at 32k (+23% vs
+12% @16k) because the 32k verify is nearly pure KV-read → captures the full fp8
benefit; the spec path's KV saving stays diluted by fp8-INVARIANT cost (K draft
forwards' MoE/weight compute + DP-coord + chain metadata). So global KV-quant makes
the ratio **monotonically worse** with context. K* = 3 at 32k (matches window),
confirming K-tuning, but even at K* the quant-only spec loses.

### High-batch confirmation — escape hatch CLOSED (self-spec loses; Marlin hurts)

b64, short 2k ctx (compute/weight-read bound, NOT KV-bound), clean:

| arm | K | tok/s | accept | vs nospec |
|---|---:|---:|---:|---|
| nospec | — | 4001 | — | — |
| spec bf16 | 2 | 3126 | 2.998 | **0.78x (loses)** |
| spec fp8-Marlin | 2 | 2737 | 2.937 | **0.68x (worse)** |

Both predictions confirmed: (1) **self-spec loses at high batch** — the machine is
compute-saturated, so the verify's B*(K+1) tokens are wasted compute with no spare
capacity to exploit (speculation needs a memory-bound machine with idle FLOPs). (2)
**Marlin fp8 makes it WORSE** (0.68x < 0.78x) — at high batch the draft is
GEMM-compute-bound and Marlin dequants fp8->bf16, so it's pure added overhead with no
native-fp8 tensor-core payoff (native fp8 MoE unavailable on H100-no-DeepGEMM).

## COMPLETE QUANT VERDICT — settled across ALL regimes (no quant self-spec win)

| regime | machine | self-spec base | + weight-quant (Marlin) | + KV-quant (fp8) |
|---|---|---|---|---|
| low batch, 16k | mem-bound | 0.99x | ~parity | 0.95x |
| low batch, 32k | mem-bound | ~parity | — (structural parity) | 0.89x |
| high batch, 2k | compute-bound | **0.78x** | **0.68x** | — |

**Quant is NEVER a self-spec win on this stack.** Low batch: wrong axis (weight) or
global (KV) → no draft/verify ratio change. High batch: self-spec loses regardless
(compute-saturated), and Marlin dequant hurts. The self-spec win lever is **window**
(draft-only KV cut), and only in the **memory-bound regime** (low batch, long ctx).

**This REINFORCES the RL-rollout thesis:** self-spec pays off exactly when the machine
is memory-bound with spare FLOPs = low batch / long generation. In RL rollout the
batch SHRINKS as sequences finish, so the run spends increasing time in this
favorable zone (cf. EfficientRollout). Sustained high batch (b64 here) is the wrong
regime — speculation has no idle compute to trade into latency. Quant's proper role
is orthogonal: a global system multiplier (fp8 KV: +12-23% for *everyone*), not a
speculation driver. (A draft-only fp8 KV *pool* — unsupported, needs new code — would
turn KV-quant into a milder window; not pursued.)

## WHY fp8 ≤ bf16 (the mechanism) — Marlin does bf16 MACs, not fp8

The counterintuitive core: fp8 weight-quant is at best parity and at high batch
*slower* than bf16. Source-verified reason (agent, csrc + oracle):

**The reachable fp8 MoE kernel (forced `moe_backend=marlin`) is weight-only W8A16: it
dequantizes fp8 weights → bf16 INSIDE the kernel and issues `mma.sync…bf16.bf16`
(both operands bf16), NOT fp8×fp8.** So its arithmetic throughput = bf16 (1×), not
native-fp8 (2×). Evidence: `moe_wna16_marlin_gemm` ("**w**eight, **n**ot
**a**ctivation, **16**-bit"); a_type stays bf16 (`csrc/.../marlin_moe_wna16/ops.cu:567`);
in-kernel fp8→bf16 dequant before the MMA (`dequant.h:1-2`, `marlin_template.h:1297`);
the emitted MMA is `…f32.bf16.bf16.f32` (`marlin_mma.h:68-75`). A true fp8 MMA
(`…e4m3.e4m3…`, `marlin_mma.h:76-83`) exists but compiles only when the *activation*
is fp8 — unreachable in W8A16.

**So fp8-Marlin = a bf16 GEMM + a weight-decompression tax.** Its ONLY benefit is
halving the weight bytes read from HBM. Therefore:
- **Low batch (mem/bandwidth-bound):** halved weight read helps, but the dequant tax +
  fp32-reduce + block-padding offset it → **parity**.
- **High batch (compute-bound at bf16 peak):** the halved read is fully amortized and
  buys nothing; the dequant tax remains and Marlin's MMA schedule is less efficient
  than the tuned bf16 Triton MoE → **slower** (2737 vs 3126). vLLM's own warning:
  Marlin "may degrade performance for compute-heavy workloads" (`marlin_utils_fp8.py:234`).

**H100 DOES have fp8 tensor cores (~2× bf16)** (`cutlass_scaled_mm_supports_fp8` true
for cap≥90) — the load-time "GPU does not have native support for FP8" is a hardwired
generic Marlin string, NOT literally true here. The auto **TRITON W8A8** path *does*
hit fp8 tensor cores (`tl.dot` on fp8 a & b) — but it pays per-forward activation
quant (`scaled_fp8_quant`) + an extra EP a2a **scale all-gather** (`naive_dp_ep.py:55`),
which at the low-batch draft dwarf the tiny GEMM → **0.65× (383)**. No in-tree
fp8×fp8 MoE path is reachable here: VLLM_CUTLASS stripped (`allow_vllm_cutlass=False`),
DeepGEMM off + needs block-fp8 ckpt, FlashInfer TRTLLM/CUTLASS SM100-only.

**Net:** on H100-no-DeepGEMM there is no fp8 config that gives the draft native-fp8
compute AND avoids the activation-quant/comm tax. Marlin trades the tax for zero
compute speedup (bf16 MACs); Triton gets the compute speedup but pays the tax. Either
way fp8 ≤ bf16.

### Empirical GEMM microbench (benchmark_marlin_p74.py, Qwen expert/dense shapes)

Isolated dense GEMM, fp16 cuBLAS (`pytorch_gemm`) vs fp8-Marlin (`marlin_gemm`), µs,
per-tensor g=-1, GPU4. **fp8-Marlin is ~2x SLOWER than fp16 at EVERY M** (not a
crossover — uniformly slower for these dense shapes):

| shape K×N | M | fp16 | fp8-Marlin | ratio |
|---|---:|---:|---:|---:|
| 2048×1536 (expert-w13) | 1 | 12.5 | 26.6 | 2.1x |
| 2048×1536 | 8 | 12.7 | 26.9 | 2.1x |
| 2048×1536 | 128 | 11.9 | 27.2 | 2.3x |
| 2048×1536 | **4096** | 42.7 | 84.9 | **2.0x** |
| 2048×5120 (qkv) | 4096 | 133.3 | 301.0 | 2.3x |
| 768×2048 (expert-w2) | 4096 | 22.3 | 53.9 | 2.4x |

**The M=4096 point is the smoking gun for "no fp8 tensor cores":** it is compute-bound,
so native-fp8 MACs would be ~2x FASTER (~21µs); instead fp8-Marlin is 2x SLOWER
(84.9µs) — a **~4x gap from ideal fp8**, i.e. it runs bf16-throughput MACs plus a
decompression tax, exactly as the source analysis says.

**Honest caveat — why the dense kernel is 2x slower but the e2e MoE reached PARITY
(not 2x):** (1) different kernel — the MoE uses grouped `moe_wna16_marlin_gemm`, not
this dense `marlin_gemm`; (2) these dense weights are small (≤10 MB, ~L2-resident on
H100's 50 MB L2), so at M≤1024 the GEMM is **fixed-overhead-bound, not bandwidth-bound**
— the fp8 half-bytes saving is moot and Marlin's ~2x higher kernel launch/workspace
overhead dominates (fp16 floor ~12µs vs fp8 floor ~26µs). The 30B MoE's expert weights
are many GB and genuinely HBM-bandwidth-bound, so there the fp8 read-halving is real and
offsets the dequant tax → parity. Both facts cohere: fp8-Marlin never accelerates
compute (no fp8 cores); its only lever is weight-bandwidth, which pays off only when
actually bandwidth-bound (large weights, low batch) — never enough to beat bf16, at
best matching it.

## (superseded) HEADLINE CORRECTION: self-spec WINS 1.13x at 32k once measurement confounds are removed

The "self-spec loses ~2x" below was **three stacked measurement artifacts on the
spec path** that the no-spec baseline never paid:
1. **batch-invariant kernels** (`VLLM_SELF_SPEC_COMPILE_CONSISTENT`) forced slow
   Triton matmuls on draft+verify -> +60% when removed.
2. **profiler syncs** (`VLLM_SELF_SPEC_PROFILE`, cuda.synchronize per region,
   ~6/cycle) stalled only the spec pipeline -> +25% when removed.
3. **per-tensor fp8 fallback** made fp8 look bad (fp8 not needed anyway).

Clean measurement (batch-invariant OFF, profiler OFF), bf16 **window** self-draft,
b8:

| ctx | no-spec | base | window |
|---:|---:|---:|---:|
| 16k | 599 | 462 (0.77x) | 524 (0.88x) |
| 32k | 436 | 368 (0.84x) | **493 (1.13x WIN)** |

The real draft forward is **cheaper than verify** (kineto: D 7.5ms vs V 13.4ms,
D/V~=0.56; the profiler region's 14.3ms was ~2x sync-inflated). The win GROWS with
context (memory-bound: verify's K+1 tokens get free, window's KV slice dominates)
-- the long-context RL-rollout regime. Lossless (accept 4.79, verify corrects),
training-free. Caveats: b8 / 32k / single-node; higher batch + the 16k crossover
still to map; it's the window (sparse-attn) draft that wins, not fp8.

Everything below predates this correction (measured with the taxes on).

---

# Phase 74 results (superseded headline) — single-node draft-step-cost stack: self-spec LOSES ~2x to tuned no-spec

Qwen3-30B-A3B, 16k context, single NVLink node **DP4/EP4 (GPUs 4-7 of
cloud-9ezI3Q)**, K=4, chat + on-dist prompts, two-length decode slope, greedy.
Arms (cumulative, NO local routing -- deferred as a large-EP lever): **base** =
EP-routed bf16 self-draft; **window** = + sinks+window(512) draft attention;
**kvq** = window + fp8 KV-cache quant. Denominators = plain decode (no-spec) at
the matched KV precision. Data: `data/`, logs: `logs/`, analyzer:
`scripts/analyze_stack.py`.

## Headline

**On this compute/KV-bound single node, self-speculative decoding at 16k loses
~2x to a properly-tuned no-spec baseline, and the I/O levers cannot close it --
they help the no-spec baseline MORE than the launch-bound self-draft.**

## Throughput (tok/s, two-length slope; accept in parens)

| batch | nospec bf16 | base (bf16) | window (bf16) | nospec fp8 | kvq (fp8+win) |
|---:|---:|---:|---:|---:|---:|
| 8  | 447 | 235 (5.00) | 233 (4.76) | 475 | 247 (4.56) |
| 16 | 638 | 395 (5.00) | 446 (4.75) | 875 | 423 (4.63) |
| 24 | (over-pool) | — | — | ~1075* | 588 (4.61) |
| 32 | (over-pool) | — | — | 1275 | 791 (4.56) |

*b24 no-spec-fp8 interpolated. bf16 base/window cap at ~b18 (pool), so their
high-batch cells are blank; kvq's fp8 pool is why it reaches b24/b32.

**Speedup = spec / matched-precision no-spec:**

| batch | window / nospec-bf16 | kvq / nospec-fp8 |
|---:|---:|---:|
| 8  | 0.52x | 0.52x |
| 16 | 0.70x | 0.48x |
| 32 | — | 0.62x |

Every spec arm is < 1.0x everywhere. Best case 0.70x (window vs the weaker bf16
baseline at b16); against the correctly-tuned fp8 baseline, 0.48-0.62x.

## Why (mechanism)

1. **The draft chain runs eager/PIECEWISE, not CUDA-graphed.** Log:
   `Draft FULL-CG: running the draft chain attention eagerly ... the captured
   decode graph is not replay-safe for the draft's growing sequence`
   (the Phase-35 blocker: full CUDA graphs collapse draft accept). So each of
   the K=4 draft forwards pays the full ~20 ms kernel-launch floor (Phase-72:
   ~2400 tiny serial kernels), while no-spec decode is CUDA-graphed and cheap.
   Four eager draft forwards per cycle >> the KV-read time they save -> loss.
2. **draft_forward is fixed-overhead / launch bound, NOT KV-bandwidth bound**
   (profiler, batch-insensitive):

   | arm | draft_forward b8 -> b16 |
   |---|---|
   | base | 24.6 -> 26.8 ms |
   | window | 18.2 -> 19.1 ms |
   | kvq | 18.0 -> 19.4 ms |

   Window/kvq cut it only ~25% (24->18 ms) -- they recover the ~6 ms KV-read
   fraction; the ~18 ms launch floor is untouched. (Profiler `verify_ms`/`df1`
   at b16 are prefill/step-0-contaminated -> unreliable; measured tok/s used
   for throughput.)
3. **The I/O lever helps no-spec MORE than self-spec.** No-spec decode is
   KV-bound, so fp8 KV speeds it up a lot (b16 638 -> 875, b8->b32 scaling
   1.84x/2x). The self-draft is launch-bound, so fp8 barely helps it. Applying
   the same lever to both therefore *widens* the relative gap (kvq/nospec-fp8
   0.52 -> 0.48 at b8->b16). This is the trap in "optimize vLLM for self-spec
   with I/O levers": you strengthen the baseline faster than the method.
4. **The draft costs residency too.** Pool tokens/rank: no-spec bf16 520k /
   fp8 1040k; spec bf16 304k / fp8 608k -- the draft's footprint takes ~40% of
   the pool, and bf16 spec caps at ~b18 (why kvq's fp8 pool is needed to reach
   b32 at all).

## The one lever that helped, and its limit

**Residency (fp8 KV) + batch amortization** is the only thing that moved the
needle: the draft's fixed per-step cost amortizes over a bigger batch (Phase 73),
and kvq's 2x pool lets spec reach b32 where bf16 base livelocks. Against the fair
fp8 baseline the gap narrows at high batch (0.48 b16 -> 0.62 b32) as no-spec's
own scaling saturates (1.46x/2x at b32). But kvq scaling (1.87x/2x b16->b32)
extrapolated against no-spec (1.46x/2x) does NOT cross 1.0x within the residency
ceiling (~b30 fp8) -- b64 projects ~0.79x, and b64 is unreachable on this pool.

## Verdict & implications

- **Single-node NVLink is not a home for self-spec at 16k** -- confirms the
  Phase-57/58 serving-batch loss, now with the mechanism pinpointed: the
  **eager draft chain's launch floor** (Phase-35 CUDA-graph blocker), not
  bandwidth. The user's I/O levers (window, KV quant) cut the draft ~25% but
  cannot fix a launch-bound draft, and they favor the no-spec baseline.
- **The lever that would matter is CUDA-graphing the draft chain** (sub-problem
  C, Phase-35-blocked), or a genuinely cheaper-compute draft (small head /
  heavy structured prune / smaller K) -- none of which is an I/O lever.
- **This does not speak to the project's real target** -- comm-bound multi-node
  EP, where no-spec pays the inter-node all-to-all that a comm-free draft
  avoids. Single-node NVLink is comm-free, so it only ever tested the draft's
  compute/launch cost, and that cost is the wall here.

## Follow-up: CUDA-graph investigation (corrects the "launch-bound" read)

Investigated whether CUDA-graphing the draft chain is the missing lever (Phases
35/69/70 + a code read). Findings:

- **The FA3 freeze is real and understood.** FA3's captured decode kernel bakes a
  **seq-len-dependent work distribution / launch grid** at capture; the draft
  chain replays one graph while the sequence grows by 1 each step -> the frozen
  schedule attends a truncated context -> accept collapses ~4.9->1.9 (Phase 35).
  Refreshing `scheduler_metadata` per step does NOT fix it (the frozen quantity
  is the graph-baked launch geometry, not the device buffer); MLA is exempt.
- **It is ~80% already built.** Fix (a) fixed-shape masked-SDPA window scratchpad
  (`DRAFT_FULLCG`, Phase 69) fully CG-captures the whole draft forward. Phase 70
  measured it: **SLOWER than PIECEWISE at 16k (49.5 vs 19.9 ms/step)**.
- **The decisive caveat -- CG is necessary but not sufficient.** The PIECEWISE
  draft (~18-24 ms) IS launch-bound and a full CG roughly HALVES it to **~11 ms**
  (Phase 72 steady-state). But ~11 ms is the **batch-1 sparse-MoE execution
  floor** (~2400 tiny per-layer kernels, 86% GPU-active) -- CG removes launch
  idle, not the kernel-count floor. Best-case arithmetic: 4x11 + 26 verify ~= 70
  ms / 4.6 tok = 15.3 ms/tok vs no-spec ~17.9 ms/tok -> only **~1.1x at b8**, and
  Phase 70's real full-CG did NOT even reach that (measured slower). So my
  "launch-bound -> CUDA-graph it" framing above is half right: PIECEWISE is
  launch-bound, but the floor under it is MoE-execution-bound and CG can't cross
  it. The draft is a full-model forward; its MoE weight/execution cost is the
  wall.
- **One clean untried lever: TRITON_ATTN.** Its decode grid is
  `(num_q_blocks, num_heads_kv)` -- **seq-len-independent** -> CG-replay-safe
  where FA3 is not, allowing a **window-free** full-CG draft. Risk: the b1
  small-batch 3D split-K kernel may re-introduce a frozen split count; needs an
  accept A/B. ~1-day experiment, marginal expected payoff (~11 ms floor).

**CG verdict:** not the lever. It halves the draft step at best (~20->11 ms),
which is marginal against no-spec and the prior full-CG measured slower. The real
levers are a cheaper-compute draft (fewer experts/layers, smaller K) or the
comm-bound regime where the baseline is handicapped.

## Follow-up 2: fp8 EP-routed draft (the "D < V via lower precision" test)

Tested the real self-spec premise -- a draft genuinely cheaper than the bf16
verify -- via an **fp8 EP-routed draft** (`W7_DRAFT_QUANT=fp8`, no replica, no
local routing; draft weights fp8, target/verify stays bf16, accept ~0.95).

| batch | base bf16 | window bf16 | fp8d | fp8dw | fp8 draft_fwd vs bf16 |
|---:|---:|---:|---:|---:|---|
| 8  | 235 | 233 | 152* | 153 | 21.4 vs 24.6 ms (-13%) |
| 16 | 395 | 446 | 241 | 290 | 22.6 vs 26.8 ms (-16%) |

(*b8 fp8d tok/s noisy ±37; b16 clean ±0.2. accept: fp8d 4.88/4.82, fp8dw
4.70/4.67 -- fp8 is a near-perfect draft, as expected.)

**The fp8 draft is a NET LOSS at every point** (best fp8dw b16 = 290 = 0.65x the
bf16 window, 0.45x no-spec) -- *worse* than the bf16 draft, despite a cheaper
profiled forward.

**The paradox:** every GPU-timed profiler region is cheaper for fp8
(draft_forward -13/-16%, chain, verify all down), yet wall-clock decode is ~1.6x
longer (6.95 vs 4.36 s). GPU work down, wall-clock up => **the fp8 draft is
host/orchestration-bound, not GPU-bound.** The fp8 EP MoE ran on the NAIVE
`MoEPrepareAndFinalizeNaiveDPEPModular` path (load log), whose per-layer
host-side dispatch/combine + activation-quant orchestration stalls the pipeline
-- cost a CUDA-event profiler doesn't see but the wall clock does. The window
doesn't help (fp8dw b8 153 ~= fp8d b8 152) -> the loss is the fp8 EP path, not KV
read.

**Two compounding reasons fp8 fails (the core of "what makes the draft cheaper"):**
1. At serving batch the draft forward is overhead/kernel-bound (Phase 72: MoE
   ~27% of it), so quantizing the FULL model buys only ~15% -- nowhere near D<<V.
2. Precision-quantizing the full model doesn't make it a SMALLER computation
   (still 48 layers, all experts) + adds an fp8 EP tax. D<<V needs a
   STRUCTURALLY reduced draft (fewer layers/experts, cut the forward 2-4x), which
   the project found collapses accept (Phases 03-08 reduced routing, 17 layer-skip).

**FINE profile confirms the overhead is host-side / unwrapped** (`prof_fine_fp8d_b8`,
`PROFILE_FINE=1`): every per-step sub-region is small (sample 1.4, pos/slot 1.3,
input-buffering 1.3, build-attn-md 1.2, rejection-sample 0.3 ms). chain 145 +
verify 26.5 = 171 ms accounts for only ~60% of the ~286 ms cycle; **~114 ms/cycle
is in unwrapped scheduler / DP-coord / EP dispatch-combine gaps -- vs ~26 ms for
the bf16 base, i.e. ~88 ms/cycle of pure fp8-EP host orchestration.** Not GPU
compute. **Salvageability:** even if that overhead were fully removed, fp8 cuts
the GPU forward only ~15% -> best case ~1.15x the bf16 draft = ~0.71x no-spec,
STILL a loss. Precision alone cannot make a full-model self-draft win here.

**The cheap-accurate-draft triangle (the wall):** on compute-bound hardware no
draft is simultaneously (a) << verify in compute AND (b) high-accept. Lower
precision keeps accuracy but barely cuts compute (and the EP tax reverses it);
structural reduction cuts compute but destroys accuracy. This is why self-spec's
real home is the COMM-bound regime, where the draft need only be comm-cheaper
(free via local routing), not compute-cheaper.

## Caveats

- Greedy accept (base 5.0 = perfect full-model draft; window ~4.75; kvq fp8-KV
  ~4.56, bounded-lossy). RL-temperature accept untested (deferred).
- One model, one node (DP4/EP4), 16k. tok/s noise ~5-11% at some points
  (b16 ±8-11%); the ~2x gap is far outside noise.
- base b24 (bf16 ceiling) point: expected over-pool livelock -- confirms the
  residency ceiling that motivates kvq.
