# Results: accept-length-per-GB frontier (single-node baseline)

Date: 2026-06-29. Qwen3-30B-A3B (128 experts, no shared). Accept length = E[accepted]
at k=4 from measured one-step beta (Phase 25 Exp2 NVFP4 local-routing); validated vs
Phase 24 B1 real k-sweep (2.74 measured vs 2.64 geometric at 0.5E). Memory = per-device
resident draft experts (the only extra cost over the bf16 verify shard).

## The frontier

| config | coverage | mem GB/dev | beta | accept len (k4) | acc/GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP4 C=8 | 0.06 E | 1.0 | 0.24 | 0.32 | 0.32 |
| FP4 C=16 | 0.12 E | 2.0 | 0.40 | 0.65 | 0.33 |
| FP4 C=32 | 0.25 E | 4.1 | 0.60 | 1.30 | 0.32 |
| **FP4 C=64** | **0.50 E** | **8.2** | **0.84** | **2.64** | **0.32** |
| FP4 C=96 | 0.75 E | 12.2 | 0.90 | 3.10 | 0.25 |
| FP4 C=128 | 1.00 E | 16.3 | 0.92 | 3.26 | 0.20 |
| FP8 full | 1.00 E | 29.0 | 0.95 | 3.52 | 0.12 |
| bf16 full | 1.00 E | 58.0 | 1.00 | 4.00 | 0.07 |

## Key finding: linear to 0.5E, then a sharp knee

Marginal accept-length per extra GB along the FP4 sweep:

```
->0.06E .. ->0.50E : ~0.32 acc/GB   (CONSTANT -- linear frontier)
->0.75E            :  0.115 acc/GB  (3x worse)
->1.00E            :  0.039 acc/GB  (8x worse)
FP4->FP8 full      :  0.02  acc/GB
```

Every GB buys ~0.32 accept length up to **0.5E coverage**, then returns collapse. Cause:
router mass concentrates in the top ~half of experts (beta rises ~linearly with C to
0.5E: 0.24->0.40->0.60->0.84), so the cache scales efficiently until the rarely-routed
long tail, which adds memory but little mass -> beta saturates (0.84->0.90->0.92).

**Low-bit dominates the frontier.** At a fixed memory budget, FP4 buys 2x the coverage
of FP8, and full-coverage-FP4 (beta 0.92) beats half-coverage-FP8 (~0.84). So FP4 is the
right bit-width for the memory-constrained regime; FP8/bf16 full are off the frontier
(acc/GB 0.12 / 0.07).

## Baseline operating points (to beat)

- **Efficiency knee:** FP4 0.5E -- **8.2 GB/dev, accept 2.64 (beta 0.84)**. Best acc/GB
  while still high acceptance. Past here you pay ~3-8x more GB per unit accept length.
- **Max accept (practical):** FP4 full -- 16.3 GB/dev, accept 3.26 (beta 0.92). 2x the
  knee's memory for +0.6 accept length.
- **Realized single-node-PCIe speedup** (f~0.62, Phase 24): knee ~1.4x, full ~1.7x.

## Targets the enhancement levers must beat

The frontier is `acc/GB <= 0.32` (linear region) and **caps at accept 3.26 for 16.3 GB**.
To push it UP:

1. **Tree / multi-candidate drafting** -- raises accept length at the *same* memory
   (extra accept comes from cheap comm-free draft compute, not GB). Moves points
   vertically up; the cleanest way past the 3.26 ceiling without more memory.
2. **Verify-warmed dynamic cache** -- raises effective coverage per GB (cache the experts
   verify just revealed as hot, not a static per-request top-C) -> extends the linear
   region / lifts the knee leftward (more accept at <8 GB).
3. **Shared-expert anchor** (shared-expert models) -- free always-local mass shifts the
   whole curve up (Phase 25); N/A for Qwen3 (no shared), strong for DeepSeek/Moonlight.
4. **Hot-cold mixed precision** (FP8 hot + FP4 cold) -- better beta per GB than uniform
   if the hot set is small.

Beat-bar in one line: **> 0.32 acc/GB below 8 GB, or accept > 3.26 at <= 16 GB/dev.**
