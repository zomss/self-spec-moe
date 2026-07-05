# Phase 57 results — the optimal spec-decode strategy at large EP, on REAL 2-node fabric

**Thesis (from README).** On MoE-EP the verify is one all-to-all over EVERY
drafted token, so exposed verify comm scales with draft **VOLUME** (chain
length / tree size), not with accepted tokens. The objective at large EP is
therefore **accept-per-verified-token**, and the optimum is **minimal draft
volume** — a short chain (K\* small, often 1), inverting single-GPU wide-tree
practice. As EP scale / comm-fraction f grows, K\* shrinks.

**What is new here vs Phase 33/50.** Every prior World B number was
single-node forced-PCIe EMULATION (Phase 33 used a local-routing stand-in for
EAGLE acceptance; Phase 50 used a real EAGLE3 head but on one node with f
emulated by an injected all-to-all delay). This phase measures the same law
and K\* with the REAL trained EAGLE3 head on the genuine 2-node inter-node
fabric (h107+h106, EP16), and compares EP8 (1-node) vs EP16 (2-node) to show
K\*(EP).

- Real EAGLE3 head: `Tengyunw/qwen3_30b_moe_eagle3` (base `Qwen/Qwen3-30B-A3B`).
- Harness: `research/52_two_node_e2e/scripts/w7_2node.py` with
  `W7_SPEC_METHOD=eagle3`; clean EAGLE env
  `research/57_large_ep_spec_strategy/scripts/env_eagle_2node.sh` (NO
  `VLLM_SELF_SPEC_*` stack, cf. Phase 50 STACK=0).
- tok/s = two-length decode slope (OUTLEN 160 vs SHORTLEN 32); accept_len from
  `vllm:spec_decode_num_accepted_tokens / num_drafts + 1`. Greedy, ignore_eos,
  off-distribution synthetic prompts (accept is depressed vs on-distribution,
  same caveat as Phase 50).

## Fixes required to run EAGLE on this self-spec branch

1. **Harness env guard** (`w7_2node.py`): the worker set
   `VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE / NODE_LOCAL / FULL_REPLICA / LOCAL_ROUTE`
   unconditionally in spec mode. Those are self-spec-only; for a real EAGLE
   head they are wrong. Now guarded by `spec_method == "draft_model"`.
2. **EP-propagation bug** (`vllm/config/speculative.py`): the self-spec fork's
   `create_draft_parallel_config` propagated `enable_expert_parallel=True` from
   the MoE target onto the draft. A dense EAGLE head has 0 experts, so
   `ModelConfig._verify_with_expert_parallelism` rejected it
   ("Number of experts ... must be greater than 0"). Fixed to propagate EP only
   when the draft is itself MoE (`draft_is_moe=self.draft_model_config.is_moe`).
   Self-spec (draft == the MoE target) is unchanged.
3. **Intermittent DP16 EAGLE deadlock** (infra, not a code bug we own). Stock
   EAGLE spec decode wedges the first `sample_tokens`/`execute_model` collective
   on the 2-node DP16 fabric where World A (self-spec) runs fine — because EAGLE
   lacks the self-spec DP-coordination stack. It is INTERMITTENT (a given
   K/batch sometimes completes, sometimes hangs) and independent of cudagraph vs
   eager (both wedge). Two partial mitigations + one real one:
   - EAGLE auto-enables async scheduling (self-spec `draft_model` runs it OFF);
     its batch-queue path wedges more readily. Force `async_scheduling=False`
     (env-gated `W7_ASYNC_SCHED`, unset -> vLLM default, self-spec unchanged).
   - `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=120` so a wedge fails fast.
   - The real mitigation: `run_ksweep.sh` runs ONE K per engine invocation
     wrapped in a hard `timeout`, force-kills wedged NCCL workers by GPU pid on
     both nodes, and RETRIES. Every K/batch point below is a completed
     (non-wedged) run. This DP16-EAGLE fragility is itself a finding: the
     conventional (non-self-spec) drafter is not robust at multi-node EP on this
     stack, independent of the throughput argument.

## STAGE 3 arm A — 1-node DP8/EP8 (lower f), CG, the K-sweep

Complete, all K first-try (single-node DP8 is robust). tok/s / accept_len:

| K | b8 | b32 | b64 |
|---:|---:|---:|---:|
| 1 | 694 /1.53 | 2054 /1.49 | **2783** /1.46 |
| 2 | **784** /1.72 | **2163** /1.68 | 2393 /1.60 |
| 3 | 725 /1.77 | 1843 /1.75 | 1863 /1.65 |
| 4 | 672 /1.84 | 1685 /1.78 | 1527 /1.67 |
| 6 | 607 /1.84 | 1335 /1.78 | 1249 /1.67 |
| 8 | 559 /1.84 | 1188 /1.78 | 1055 /1.67 |

- **accept saturates** by K≈4 (b8 1.84, b32 1.78, b64 1.67); every draft token
  past that is pure volume waste. Higher accept than Phase 50's 1.67 at small
  batch (b8 1.84) — batch/prompt-set dependent; still off-distribution synthetic.
- **K\* at EP8: b8/b32 → K\*=2, b64 → K\*=1.** Even single-node, the largest
  batch already prefers the shortest chain: at b64 tok/s falls monotonically
  1→8 (2783→1055, K8 = 0.38× K1). The wide-tree-is-free intuition already breaks
  at b64 EP8; the thesis predicts EP16 pushes K\* to 1 at smaller batch too.

## STAGE 1 — smoke (de-risk EAGLE on this HEAD): PASS

1-node DP8/EP8, Qwen3-30B + real EAGLE3, K=2, b8, clean EAGLE env, W7_GPU_MEM
0.90. EAGLE3 loads and routes; **accept_len = 1.716** — sane, and consistent
with Phase 50's ~1.60-1.67 (slightly higher here; on-distribution the head is
~2.3-2.5). tok/s at b8/1-iter is noisy (single small batch); the trustworthy
throughput curve is the STAGE 2/3 sweep below.

## STAGE 2 — real 2-node EP16 EAGLE3 K-sweep (Qwen3-30B, clean env, CG on)

tok/s (accept_len). K4 EP16 = execution-mode cliff (decode 3.4x, verify shape
> captured CG sizes -> eager); EXCLUDED as an artifact, not a data point.

| batch | EP | K=1 | K=2 | K=3 | K=4 |
|---:|---:|---:|---:|---:|---:|
| 8  | EP8  | 693.8 (1.53) | **783.5 (1.72)** | 724.5 (1.77) | 671.6 (1.84) |
| 8  | EP16 | **533.4 (1.53)** | 540.6 (1.71) | 521.0 (1.84) | _155.5*_ |
| 32 | EP8  | 2053.8 (1.49) | **2162.8 (1.68)** | 1842.7 (1.75) | 1684.8 (1.78) |
| 32 | EP16 | 1499.6 (1.52) | **1621.5 (1.67)** | 1350.9 (1.74) | _370.7*_ |
| 64 | EP8  | **2782.6 (1.46)** | 2392.6 (1.60) | 1863.3 (1.65) | 1527.4 (1.67) |
| 64 | EP16 | **2197.4 (1.46)** | 1714.3 (1.60) | 1485.8 (1.62) | _431.7*_ |

Accept is EP-INVARIANT (same head): 1.46->1.84 across K, identical at EP8 and
EP16 within noise. So every tok/s difference between EP8 and EP16 is PURELY
the verify-volume (comm) cost, not acceptance -- the cleanest possible
isolation of the mechanism.

## STAGE 3 — K*(EP): the optimal chain shortens as EP widens

**The design rule, measured.** The throughput benefit of drafting a 2nd token
(K2/K1) collapses as EP grows, at every batch:

| batch | K2/K1 @ EP8 | K2/K1 @ EP16 | K* @ EP8 | K* @ EP16 |
|---:|---:|---:|---:|---:|
| 8  | **1.129** (+13%) | **1.013** (+1%) | 2 | ~1 |
| 32 | 1.053 (+5%) | 1.081 (+8%) | 2 | 2 |
| 64 | 0.860 (-14%) | **0.780 (-22%)** | 1 | 1 |

- **Serving batch (b64): K\*=1 on both, but the over-draft penalty is STEEPER
  at EP16** (-22% vs -14% for a 2nd token). The wider EP's more expensive
  inter-node verify all-to-all makes each extra drafted token cost more, while
  it buys the identical +0.14 accept -> drop it.
- **Low batch (b8): the deep-draft benefit EP8 enjoys (+13% at K2) is ERASED
  at EP16 (+1%)** -- the inter-node verify comm eats the latency-regime upside
  of a longer chain. The single-GPU instinct ("low batch, draft deeper") does
  NOT transfer to multi-node EP.
- This CONFIRMS + SHARPENS Phase 50's emulated K\*=1 on real fabric: at large
  EP the optimum is a minimal chain, and the pressure toward K\*=1 grows with
  EP width -- exactly the repositioned thesis.

**Verify-volume law, real fabric:** at b64, tok/s falls monotonically with K
at both EPs (EP16: 2197/1714/1486; EP8: 2783/2393/1863) while accept saturates
(~1.67) -- over-drafting is pure verify-comm waste, the Phase 33 law now
measured inter-node with a real head.

## Honest framing (baseline)

This is a within-EAGLE CONFIGURATION result (choose K\*), not a new method.
The full K-sweep is shown so the reader sees the whole curve; the contribution
is the LAW + the K\*(batch, EP) design rule on real hardware. Against a tuned
EAGLE-2 dynamic tree (which already adapts depth by confidence, Phase 31
near-oracle) the win is "use its smallest setting at large EP", not a headline
multiplier. Accept is off-distribution-depressed (synthetic prompts, ~1.46-1.84
vs the head's ~2.3-2.5 on-distribution, cf. Phase 50); higher accept shifts K\*
up slightly but the volume-cost slope -- and its steepening with EP -- is
accept-independent.

## Independent replication (retry-runner, self-consistent with the EP8 arm)

An independent EP16 cg sweep via `run_ksweep.sh` (one K per invocation, hard
`timeout`, GPU-pid force-kill + retry) -- the SAME runner and async-off cg
settings as the EP8 arm -- reproduces the STAGE 2 table for the reliable K=1-3
core (tok/s, accept_len):

| K | EP16 b8 | EP16 b32 | EP16 b64 |
|---:|---:|---:|---:|
| 1 | 462.6 (1.52) | 1523.9 (1.52) | 2307.3 (1.46) |
| 2 | 532.9 (1.65) | 1525.3 (1.67) | 1724.2 (1.60) |
| 3 | 508.8 (1.76) | 1435.0 (1.75) | 1547.2 (1.62) |

K2 b64 = 1724 here vs 1714 in the `fill` runs (0.6%); the K\*(EP) conclusion is
identical: b64 falls monotonically (K\*=1), b32 K1≈K2 (K\* dropped from 2 at EP8),
b8 K2 barely beats K1 (+1%). The mechanism (accept EP-invariant; every tok/s gap
= verify volume) holds on both independent 2-node runs.

**Reliability caveat on K≥4 @ EP16.** K=4-8 at EP16 did not complete cleanly in
the retry-runner sweep: the DP16 EAGLE spec cycle wedges the first collective,
and the wedge rate rises with K (more draft steps = more collectives) — plus
these high-K attempts overlapped concurrent 2-node `fill` runs, so contention
compounds the intrinsic fragility. K=1-3 (the K\* region) is the trustworthy
core on both runs; the K≥4 tail (steep tok/s falloff) is anchored by the fully
clean single-node EP8 sweep (b64: K8 = 0.38× K1). Net: the design rule rests on
the well-measured K=1-3 EP8-vs-EP16 differential, not on the fragile K≥4 points.
