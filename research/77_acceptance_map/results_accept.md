# Phase 77 results

## β sweep complete: dense (30 cells) + MoE (36 cells), 1152 positions each

`data/beta.csv`. β_greedy at 2k / 16k / 32k:

| lever | dense Qwen2.5-7B | MoE Qwen3-30B-A3B | portable? |
|---|---|---|---|
| win128 | .984 / .976 / .972 | .949 / .969 / .978 | ✓ (±0.03) |
| win512 | .983 / .978 / .975 | .953 / .971 / .983 | ✓ |
| win2048 | .996 / .983 / .981 | .987 / .975 / .983 | ✓ |
| q_fp8 | .984 / .978 / .980 | .990 / .993 / .990 | ✓ |
| q_int4 | .933 / .924 / .933 | .974 / .972 / .973 | ~ (+0.05 on MoE) |
| kvq_fp8 (draft-only) | **.75 / .84 / .60** | **.98 / .97 / .98** | **✗ INVERTED** |
| skip125 | .47 / .45 / .45 | .71 / .70 / .71 | ✗ (+0.25 on MoE) |
| skip25 | .09 / .09 / .10 | .40 / .41 / .41 | ✗ |
| skip50 | .03 / .03 / .03 | .15 / .15 / .15 | ✗ |
| lr25 / lr50 | n/a | .69-.72 / .83-.84 | (MoE-only; ctx-stable) |

**Prediction scorecard update:**
- P1 (window ctx-stable) **CONFIRMED**, and stronger: window SIZE barely moves
  β on either model (win128 ≈ win2048) → the smallest window strictly
  dominates (same β, better R). P74's 512 was not optimal.
- P2 (skip collapse) **CONFIRMED on dense** (0.47/0.09/0.03 — super-linear,
  all below the 0.78 break-even). On MoE the curve is ~0.24 HIGHER at every
  fraction (0.71 at 12.5%) — architecture-dependent, though still below
  break-even after composing with its R.
- P3 (quant monotone in bits, ctx-flat) **CONFIRMED**.
- P4 (only local-route architecture-unstable) **REFUTED — law refined**:
  kvq_fp8 and skip swing MORE across architectures than local-route's own
  variation. Only window and fp8-weight β are portable. A strategy selector
  MUST measure β per architecture — the phase's premise, now its conclusion.

## kvq_fp8 instability: SOLVED — it is the dense model's K outliers (no QK-norm)

`scripts/kvq_probe.py`, dense, same refs (paired), 1152 positions/cell:

| variant | 2k | 16k | 32k |
|---|---|---|---|
| base (K+V e4m3, scale 1.0 = vLLM default) | .751 | .840 | .604 |
| **K only** | .748 | .837 | .610 |
| **V only** | **.992** | **.990** | **.992** |
| keep sinks bf16 | .744 | .839 | .609 |
| per-token amax rescale (K+V) | .545 | .569 | .558 |
| per-channel amax rescale (K, KIVI-style) | .542 | .480 | .543 |

1. **The damage is entirely K-side** (k_only ≈ base; v_only ≈ 0.99 — V
   quantization is FREE). Softmax amplifies K perturbations into attention
   reshuffles; V errors average linearly.
2. **Uniform across prompts** (per-prompt β all depressed) — systematic, not
   sampling variance. Not the sinks (protecting them changes nothing).
3. **Rescaling makes it WORSE** — both per-token and per-channel amax scaling
   drop β to ~0.5. The scale-1.0 e4m3 cast is already the best of the three
   for this tensor distribution.
4. **Root correlate, measured from the caches**: dense Qwen2.5 K amax = 426
   (just under e4m3's 448 saturation; quant spacing ~32 ≈ 7.5% on the outlier
   channels that dominate attention logits), max/rms = 206. Qwen3-MoE (QK-norm)
   K amax = 150, max/rms = 70 — and its kvq β is a stable 0.97-0.98.
   **Hypothesis: QK-norm decides whether a draft-only fp8-K read is viable.**
   (Third architecture — MLA, which compresses KV differently — will test it.)
5. **Design guidance for the hypothetical draft-only KV pool**: on
   non-QK-norm models quantize V ONLY (β 0.99, 25% of KV bytes for free);
   fp8-K only on QK-norm architectures. The map's kvq reference column is
   architecture-conditional.


## Anchor gate: OPEN — offline β reproduces both e2e accept lengths

`scripts/anchor_gate.py` (offline teacher-forced window-draft acceptance,
W7-faithful prompts, draft decode step against target KV sliced to
sinks16+window512 with original RoPE), composed via geometric τ and compared
to the two 76-E3 end-to-end measurements:

| cell | β_greedy (positions) | τ_geom | E3 measured | err | verdict |
|---|---|---|---|---|---|
| dense Qwen2.5-7B, 16k, K=4 | 0.9766 (384) | 4.771 | 4.850 | **−1.6%** | PASS |
| MoE Qwen3-30B, 32k, K=6 | 0.9757 (1152) | 6.510 | 6.493 | **+0.3%** | PASS |

Data: `data/anchor_gate.json`.

**The method is calibrated**: one-step offline β + geometric composition
predicts real spec-decoding accept lengths to ≤2% on both architectures. The
sweep (`score_accept.py`, next) inherits this credibility.

### Measurement requirements learned (bind the sweep design)

1. **≥12 prompts × 96 positions (~1000+) per cell.** Per-prompt β spread is
   large (0.91–1.00 across the same bank); 4 prompts missed by −9% (the first
   gate attempt FAILed on sampling variance alone — running mean climbed
   0.93→0.976 monotonically as prompts accumulated).
2. **Per-prompt β has run-to-run jitter** (same prompt: 1.000 in one process,
   0.927 in another — MoE router argmax flips at numerics boundaries amplify
   close calls; cf. P75-E2's batch-invariance finding). The AGGREGATE over
   ≥1000 positions is the stable statistic; never quote per-prompt β.
3. T=1.0 overlap tracks greedy β closely at these levels (0.9754 vs 0.9757 on
   MoE) — both fall out of the same forwards; report both.
4. The token-exact window slice (vs vLLM's 16-token page granularity) does NOT
   produce a measurable discrepancy at window 512 — the gate closed to 0.3%
   without modeling pages.
