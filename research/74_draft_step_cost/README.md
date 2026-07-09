# Phase 74 — make the self-spec DRAFT step cheap enough to be useful

Source phases: 62 (window-KV draft), 64 (16k E2E LOSS = pure propose-path cost),
65 (per-step overhead fixes), 66 (shared-KV / batch residency), 72 (kernel trace
= latency floor), 73 (draft forward is FIXED-OVERHEAD-dominated, amortizes with
per-rank batch), 58 (RL-regime motivation), 59/63 (long-context KV-bound regime).

## Motivation (the WHY, one paragraph)

The deployment that makes lossless self-speculative decoding compelling is **RL
post-training rollout**: long reasoning generations (KV-bound, the MagicDec /
Phase-59 regime where spec wins at serving batch), high rollout batch, and a
policy whose weights update every step -- so a *training-free* self-draft that
tracks the policy for free and preserves the exact sampling distribution
(lossless -> no gradient bias) is exactly what's needed, where a trained EAGLE
head goes stale. But none of that matters until the draft step is cheap enough
on the real fabric. **Phase 64 proved the accept side already works (window-512
K4 accept 4.55, r=0.957) and the E2E is a LOSS (0.34-0.51x) purely from
propose-path systems cost.** So this phase optimizes the draft step FIRST;
acceptance-at-RL-temperature and the regime/RL eval are Phase 75.

## What is already fixed (do not re-litigate)

Phases 65-73 already removed the gross overhead Phase 64 saw (the ~75-98 ms/cycle
fixed + ~38 ms/step marginal): DP-coordination sync storm (`DRAFT_DP_COORD_CPU`),
chain metadata dispatch (`DRAFT_CHAIN_LIGHT_MD`), draft comm (fp8 full-replica
local route), and draft-KV duplication (`SHARED_KV`). **Result (Phase 73,
single-node DP8/EP8, SHORT 2k so batch scales free of the pool cap):**

| per-rank b | draft_fwd ms | df/tok ms | verify ms | accept | tok/s |
|---:|---:|---:|---:|---:|---:|
| 1  | 6.87 | 6.87 | 19.11 | 4.84 | 64 |
| 8  | 9.74 | 1.22 | 23.49 | 4.53 | 350 |
| 64 | 13.33 | 0.21 | 49.04 | 4.54 | 1608 |

`draft_forward(b) = F_fixed + b*m` with **F_fixed ~= 6.6-6.9 ms (97-100% of the
b1 forward)** and `m` falling to ~0.04 ms/tok by b64. The forward grows only
1.94x for 64x tokens. Kernel trace (Phase 72): the b1 forward is ~2400 tiny
serial kernels (median 1.9 us, 86% GPU-active) -- **latency / kernel-count
bound, not FLOP or bandwidth bound.**

## The residual wall (what THIS phase attacks)

The draft forward is fixed-overhead-dominated and amortizes with per-rank batch,
so at SHORT context it is already a win. **The problem is that the target regime
-- 16k long context on 2-node EP16, where the RL/MagicDec win lives -- PINS
per-rank batch below the amortization knee** (Phase 66: b32 = 106% of the KV
pool, preemption livelock; measurements stuck at b6-b24). At that low batch
F_fixed (~6.6 ms) is paid per draft step against a ~7 ms economic budget, so the
K-step cycle loses (0.86x, Phase 66).

**Therefore the highest-leverage lever is FOOTPRINT REDUCTION to raise the
achievable per-rank batch at 16k until F_fixed amortizes** -- exactly the levers
the user named. Two distinct sub-problems, do not conflate them:

| sub-problem | levers | mechanism |
|---|---|---|
| **A. batch residency at 16k** (the dominant one) | KV-cache quant; weight quant (frees HBM for pool); shared-KV (done P66) | more resident tokens/GB -> higher per-rank batch -> F_fixed amortizes |
| **B. per-forward cost** (F_fixed + m) | sparse/window draft attention (cuts m's KV read); weight quant (cuts m's GEMM); local routing (cuts draft comm) | cheaper single draft forward |
| **C. F_fixed floor** (the 2400-kernel launch cost) | CUDA-graph / kernel fusion | fewer launches -- **blocked by Phase 35** (FULL-CG freezes FA3 decode work-distribution -> accept collapses); the one hard open item |

Sub-problem A is where the 16k loss actually lives, and KV-cache quant + weight
quant are the direct fixes. B is incremental. C is the fundamental floor and is
tracked but not assumed solvable here.

## Losslessness subtlety (must be measured, not assumed)

Every draft-side lever (window/sparse draft attention, draft weight quant, draft
pruning, local routing) is **strictly lossless** -- it only changes the *draft*;
the full-KV bf16 full-EP *verify* keeps the output exact via rejection sampling.
**KV-cache quant is the exception:** to shrink the *pool* it must quantize the KV
that *verify* reads (the pool is the target KV under shared-KV), which makes the
verify itself fp8/int4-KV -> a small, bounded distribution shift, NOT strictly
lossless. So run two stacks and report both:

- **Strict-lossless stack (single-node lead):** shared-KV + sparse/window draft
  attention. Strictly lossless (draft-only); cuts the draft forward but does NOT
  shrink the pool (per-token KV stays bf16), so batch gain is limited.
- **Bounded-lossy stack:** + fp8/int4 KV-cache quant on the target pool. Large
  batch gain, but verify is now quantized-KV -> Phase-75/RL must adjudicate
  whether the rollout-distribution shift is acceptable (fp8 KV is production-
  standard; this is the Phase-58 "B2 bounded-lossy" frontier, made concrete).

## Assumptions

- Harness `research/52_two_node_e2e/scripts/w7_2node.py` + a one-line optional
  `W7_KV_CACHE_DTYPE` kwarg (for arm 2). Env `scripts/env_cloud4.sh` (this box):
  EP-routed bf16 draft + Phase-65/66 flags (`DRAFT_DP_COORD_CPU`,
  `DRAFT_CHAIN_LIGHT_MD`, `SHARED_KV`), step-0 full-CG + PIECEWISE chain.
- Model: Qwen3-30B-A3B (GQA, 96 KiB/token KV) primary; DeepSeek-V2 (MLA, tiny KV)
  as the contrast that isolates *how much* of the win is the KV-footprint lever.
- 16k context (`W7_CTX_TOKENS=16384 W7_MAX_MODEL_LEN=20480`) is where the wall
  is; short 2k for the cheap F_fixed/m batch-scaling. Cost measured single-node
  DP4/EP4 on GPUs 4-7 of this box; the comm-bound 2-node regime is deferred.
- Accept is HELD at the Phase-62 window value and only re-checked as a guardrail;
  full accept-vs-temperature is Phase 75.

## Measurement protocol (draft-step cost FIRST)

Primary metric per lever-arm: **(1) max achievable per-rank decode batch at 16k
(residency-limited), (2) the draft-step cost D and cycle tok/s at that batch,
(3) the `F_fixed + b*m` refit.** Guardrail: accept_len must stay >= ~4.4.

Cumulative arms (each adds one lever to the previous). Lead with the levers that
help on a single NVLink node (where EP is compute-bound); the replica-coupled and
comm-elimination levers are deferred to the large-EP phase.

0. **base** — EP-routed bf16 self-draft (NO replica, NO local route) + P65 fixes
   + shared-KV, full KV. The honest single-node baseline (Phase-64 arm A).
1. **+window** — sparse (sinks+window) draft attention (`DRAFT_KV_WINDOW=512`);
   strictly lossless, cuts the draft KV read at long context (sub-problem B,
   per-step cost).
2. **+kvq** — fp8 KV-cache quant (`--kv-cache-dtype fp8`, needs a one-line
   `W7_KV_CACHE_DTYPE` knob in the harness); shrinks the pool -> raises the
   resident batch ceiling at 16k (sub-problem A). Bounded-lossy: measure accept.
3. **+prune** *(stretch)* — structured draft-only pruning (frees HBM + FLOPs);
   accept-bounded, Phase-17 layer-skip caution — keep only if accept holds.

**Deferred to the large-EP / multi-node phase (NOT single-node levers):** local
routing and the replica-coupled draft weight quant. Their only payoff is
eliminating the EP all-to-all, which matters only when comm is the bottleneck
(large EP); on NVLink EP is compute-bound and the FP4/FP8 replica they need
(~19 GB) *shrinks* the KV pool -- the opposite of the residency goal.

For each arm record: tokens/rank pool, max resident batch, D, cycle tok/s, accept,
and the F_fixed/m refit. Attribute each lever to sub-problem A (raised batch
ceiling) vs B (cheaper forward).

## Economic budget (the bar)

Speedup `= accept_len / (K*(D/V) + 1)`. Break-even needs `D < ((accept-1)/K)*V`
~= `0.9*V` at K=4/accept4.55; the ~1.9x target needs `D ~= 0.35*V ~= 7 ms` at
V~=20 ms (matches the Phase-65 budget). So: **get the draft step under ~7-18 ms
AT a per-rank batch that is actually resident at 16k.**

## Decision criteria

- **GO:** some stack reaches a per-rank batch at 16k where the draft step
  amortizes under budget and accept holds -> measured/projected E2E > 1.0x at
  16k serving batch. -> Phase 75 (accept@temperature + regime eval + one RL run).
- **Diagnostic (always):** per-lever attribution — which lever bought the batch
  ceiling (A) vs the per-forward cost (B); how much of the 16k win is the KV
  lever (Qwen3-GQA vs DeepSeek-MLA contrast).
- **Partial/NO:** if even full KV+weight quant cannot lift per-rank batch past
  the amortization knee at 16k on this fabric, the residual is F_fixed ->
  escalate the Phase-35 CUDA-graph accept-collapse as the true blocker (sub-
  problem C) and decide whether the 16k self-spec win *requires* the CG track.

## Expected next artifact

`results_draft_cost.md` (the lever-stack table + F_fixed/m refits + GO/NO), then
the Phase-75 README: accept-vs-temperature de-risk, the diverse-styles regime
eval (context x fabric x GQA/MLA), and one real RL rollout demonstration.
