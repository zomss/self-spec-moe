# Results: Intra-Node Throughput via Reduced Expert Activation

Date: 2026-06-24

Tests the throughput mechanism of a local-only draft: it activates fewer unique
experts than full routing, so at the memory-bound decode regime the draft step
reads fewer expert weights and is cheaper -- no communication involved. Driven by
vLLM's real `fused_experts` Triton kernel; per-layer (per MoE call) time in ms.

## MoE-FFN time vs active experts M (Qwen3-30B-A3B shapes: H=2048, I=768, E=128, K=8)

| batch | M=32 | M=64 | M=96 | M=128 | **M64/M128** |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0.226 | 0.229 | 0.240 | 0.287 | 0.80 |
| 64 | 0.251 | 0.235 | 0.338 | 0.433 | **0.54** |
| 256 | 0.230 | 0.256 | 0.359 | 0.464 | **0.55** |
| 1024 | 0.260 | 0.345 | 0.427 | 0.529 | 0.65 |
| 4096 | 0.782 | 0.831 | 0.869 | 0.937 | 0.89 |

## GPT-OSS-20B shapes (H=2880, I=2880, E=32, K=4)

| batch | M=8 | M=16 | M=24 | M=32 | **M16/M32** |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.210 | 0.320 | 0.457 | 0.589 | **0.54** |
| 256 | 0.274 | 0.350 | 0.478 | 0.614 | 0.57 |
| 1024 | 0.511 | 0.553 | 0.718 | 0.785 | 0.71 |
| 4096 | 1.810 | 1.912 | 1.942 | 1.958 | 0.98 |

## Mechanism: confirmed

At serving batch (B ~ 64-1024) the MoE FFN is **memory-bound on expert weights**, so
restricting to `M = E/2` experts (the `phi=0.5` draft) costs **~0.54-0.71x** of full
routing. Two boundaries:
- **B=16:** ratio ~0.65-0.80 -- a time floor (overhead, and full routing activates
  fewer than E experts at small batch).
- **B=4096:** ratio ~0.89-0.98 -- compute-bound; token GEMMs dominate, so fewer
  experts barely helps.
- **Floor at low M:** below `M = E/2` the time stops dropping (M=8/16/32 are flat),
  while acceptance collapses -- so `M = E/2` is the sweet spot (cheap *and* still
  ~0.78-0.85 acceptance via affinity/EPLB).

## Throughput envelope (end-to-end)

A draft step is `T_attn + r_moe * T_moe`; let `phi_moe` = MoE fraction of the decode
step. With measured `r_moe ~= 0.55`, verify creep `T_verify/T_full ~= 1.15`
(verify routes (k+1)x tokens), and `beta = 0.85` (affinity G=2):

`gain = E_tokens(beta,k) / (k * T_draft/T_full + T_verify/T_full)`,
`T_draft/T_full = (1-phi_moe) + 0.55*phi_moe`

| phi_moe | T_draft/T_full | gain (k=2) | gain (k=3) |
| ---: | ---: | ---: | ---: |
| 1.00 (MoE-dominated) | 0.55 | **1.14x** | 1.14x |
| 0.80 | 0.64 | 1.06x | 1.04x |
| 0.60 | 0.73 | 0.98x | 0.95x |

## Reading

- The mechanism is real and intra-node (no communication), but the end-to-end gain
  is **modest: ~1.05-1.14x**, and only clearly positive when the MoE FFN dominates
  the step (`phi_moe >= ~0.8`) with a short draft (`k=2`) and high acceptance.
- For expert-heavy models at short sequences (Qwen3-30B is ~28/30B params in
  experts; small KV) `phi_moe` is high -> gain toward 1.1x. Long sequences (large KV
  read) raise `T_attn`, lower `phi_moe`, and erode the gain toward break-even.

## Honest caveats

- **Sub-optimal kernel config.** The Triton `fused_experts` ran with the default
  (non-autotuned) config for these shapes; absolute times are inflated and the
  scaling is qualitative. A production-tuned kernel would sharpen the numbers.
- **`phi_moe` unmeasured.** The gain hinges on the MoE fraction of the full decode
  step, not measured here; the envelope is parameterized by it.
- **Verify creep (Cascade).** Verify processes (k+1)x tokens at full routing; the
  memory creep (~1.15x) directly eats the gain.
- **Mechanism overlaps SS-MoE** (expert-subset self-speculation for on-device memory
  savings). The contribution here is the EP-serving *throughput* framing and the
  composition with the affinity/EPLB acceptance machinery -- it is the variant that
  flips Cascade's "spec hurts MoE" to a small positive at memory-bound batch.

## Pinned phi_moe: real end-to-end vLLM decode

Instead of leaving `phi_moe` as a free parameter, measured it directly: ran real
vLLM single-GPU decode (Qwen3-30B-A3B, dummy weights, TP=1) with `num_experts`
overridden via `hf_overrides` (E=32/64/128). Same attention, fewer expert weights,
so `T(E) = T_attn_and_other + T_moe(E)` and `T(E/2)/T(E)` IS `T_draft/T_full`
end-to-end. ms/token:

| num_experts | B=64 | B=256 |
| ---: | ---: | ---: |
| 32 | 10.68 | 15.61 |
| 64 | 15.6 | 20.96 |
| 128 (full) | 24.87 | 30.63 |

Linear in E (active experts saturate to E at these batches), confirming the model:
`T = 6.07 + 0.147*E` (B=64). Backing out:

- **phi_moe = 0.76 (B=64), 0.65 (B=256)** -- MoE FFN is ~65-76% of the decode step.
  Expert-heavy model -> MoE-dominated -> the favorable regime.
- **T_draft/T_full = 0.63 (B=64), 0.68 (B=256)** measured end-to-end (matches the
  `(1-phi)+0.55*phi` model from the kernel sweep).

### Pinned throughput gain (measured T_draft/T_full + measured verify creep)

Verify processes `B*(k+1)` tokens; from the E=128 batch scaling, full-routing time
grows ~0.030 ms per extra token. Gain at B=64:

| beta | k=2 | k=3 | k=4 |
| ---: | ---: | ---: | ---: |
| 0.85 | **1.07x** | 1.02x | 0.97x |
| 0.90 | **1.12x** | 1.10x | 1.07x |

## Conclusion

Unlike the latency story (intra-node = structurally ~1.0x), the **throughput**
mechanism is genuinely present intra-node: a `phi=0.5` local draft halves the MoE
FFN weight read, and with `phi_moe ~= 0.7` (now pinned by real vLLM decode) the
draft step costs ~0.63x of a full step end-to-end. But the net gain is **modest:
~1.07x (beta=0.85) to ~1.12x (beta=0.9) at k=2**, eroding to break-even by k=4
because the verify step pays for `(k+1)x` tokens. It is a real, communication-free,
fully-intra-node positive -- just a small one (~1.1x), conditional on short drafts +
high acceptance, and incremental over SS-MoE's expert-subset mechanism.
