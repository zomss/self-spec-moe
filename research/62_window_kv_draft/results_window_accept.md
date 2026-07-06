# Phase 62 results — window-KV draft accept ablation at 16k (Qwen3-30B-A3B)

Setup: single node h107, DP8/EP8 (W7_NODES=1, W7_LOCAL_WORLD=8), K=4, batch
8/rank, W7_CTX_TOKENS=16384, W7_MAX_MODEL_LEN=20480, chat + on-dist prompts
(phase-57 bank), MNB=8192, GPU_MEM=0.90, 2 iters + 1 warmup, greedy.
Harness: w7_2node.py (phase 52), runner scripts/run_arm.sh. Accept metrics
from rank 0 (8 requests); tok/s = two-length-slope decode rate at b8.
Per-token rate r solved from accept_len = 1 + r + r^2 + r^3 + r^4.

## Part 2 — canary (~2k ctx, K=2, bf16 EP-routed self-draft)

| arm | window | accept_len | solved r | gate |
|---|---|---|---|---|
| canary_off | off | **3.000** | 1.000 | == K+1 exactly (lossless self-draft) PASS |
| canary_w64 | 64 | **2.540** | 0.838 | < 3.0 (engages), > 2.0 PASS |

Engagement proof: `[kv-window] ... true_max_seq=2051 win_seq=83` (1 sink
block + ~4 window blocks); zero kv-window lines in the OFF run.

## Part 3 — 16k ablation (K=4)

| arm | config | accept_len | solved r | tok/s (b8, secondary) |
|---|---|---|---|---|
| A canary-16k | bf16 EP-routed, window OFF | **5.000** | 1.000 | 170.1 |
| B1 window-only | bf16 EP-routed, W=1024 | **4.559** | 0.954 | 198.7 |
| B2 window-only | bf16 EP-routed, W=512 | **4.584** | 0.957 | 201.3 |
| B3 window-only | bf16 EP-routed, W=256 | **4.581** | 0.956 | 204.2 |
| C composed | fp8 FULL-REPLICA + local-route, W=512 | **4.547** | 0.953 | 73.7* |
| D quant-ref | fp8 FULL-REPLICA + local-route, window OFF | **4.810** | 0.981 | 124.2* |

All rows: 1 engine launch, try 1, no retries needed. Data: `data/w72n_w62_*.json`.

Window engagement in the 16k runs (from `[kv-window]` debug lines, once per
500 draft-step builds): steady-state draft steps saw kv_len ~530-544 (W=512),
~1026-1056 (W=1024), ~274-288 (W=256) against true seq ~16,350-16,480 —
i.e. **30x / 16x / 58x less KV read by the draft**, sustained through decode.

## Readouts

1. **B2 (W=512 at 16k): per-token r = 0.957 >= 0.85.** The hypothesis holds
   on an MoE — with only sinks(16) + 512 trailing tokens (3.3% of the KV),
   the self-draft still matches the full-KV target ~96% per token. The
   window curve is remarkably flat: W=256 -> 1024 all land at r ~ 0.954-0.957
   (differences within run noise; B1 < B2 is noise, not signal). The knee is
   below 256: the short-context canary (W=64 at 2k) gives r ~ 0.84.

2. **Composition: better than multiplicative.** Predicted
   r_C = r_B2 x (r_D / r_A) = 0.957 x 0.981 = 0.938 -> accept_len 4.42.
   Measured C: r = 0.953, accept_len 4.547 (+0.13 vs prediction). The window
   and FP8 factors do NOT compound fully — the two error sources overlap
   (tokens the windowed draft misses are largely the same "hard" tokens FP8
   misses), so the composed stack keeps ~0.95/token, essentially the
   window-only rate.

3. tok/s (secondary, b8 only): window arms gain ~17-20% over A even in this
   compute-light regime (draft chain reads 30x less KV; verify unchanged).
   (*) C/D tok/s are NOT comparable to A/B: the fp8 full replica halves the
   KV pool (119,552 vs 221,664 tokens/rank), so 8x16.6k requests don't fit
   concurrently and the scheduler runs them in waves (observed bs 2-7 per
   step, 3x "Waiting" log lines); per-drafted-token accept is unaffected.

## Mechanism validity notes

- Verify pass metadata untouched — the target remains full-KV exact, so the
  pipeline stays lossless regardless of W (A = 5.000 exactly confirms the
  end-to-end path, incl. the K=4 chain, is healthy under the window build).
- KV writes use the TRUE block table (slot mapping computed before the
  windowed view is derived); the compacted view only changes what draft
  attention READS. Prompt ingest stays full-KV (window gated to
  decode-shaped proposes), so prompt KV is written exactly; KV of generated
  tokens is written under windowed reads (StreamingLLM-recurrent semantics
  for the generated suffix).
- accept_len -> r assumes a uniform per-token rate; with step-0 windowed in
  steady state all K draft tokens are produced under the same policy, so the
  solve is self-consistent.

## Conclusion

Window-KV drafting is essentially free in accept at 16k on Qwen3-30B-A3B:
r ~ 0.96 at W=512 (30x KV reduction), and it composes sub-multiplicatively
(favorably) with the FP8 full-replica draft (r ~ 0.95 combined vs 0.98 for
FP8 alone). At long context the draft's KV read — the KV-bound term of the
step — can be cut ~30x for a ~4% per-token accept cost.
