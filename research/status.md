# Self-MoE-Spec Research Status

Date: 2026-06-24

## Executive Summary

The original naive Self-MoE-spec idea is now **high-risk**:

```text
local top-k expert draft
+ exact full-MoE verification
```

The timing side still says multi-node EP could provide enough communication
savings, but the model side currently looks weak. Real local-only routing on the
tested MoE checkpoints does not preserve the target model's next-token
distribution well enough.

Do **not** proceed directly to runtime implementation of naive local-top-k
drafting. The best next direction is to pivot to cheaper/hybrid drafts.

## Phase Summary

| Phase | Status | Main Finding |
| --- | --- | --- |
| `00_proposal` | Complete | Idea is plausible and not clearly duplicated, but the viable regime is narrow. |
| `01_communication_advantage` | Complete | Communication-disabled draft only helps when exposed all-to-all is large. |
| `02_timing_envelope_experiment` | Complete for local/emulated setting | Single-node EP is weak; forced IB is unstable; delay emulation is available. |
| `03_local_routing_acceptance` | Initial real-model tests complete | Real router coverage and next-token proxy are weak for tested models. |
| `04_expert_placement_acceptance` | Complete | Optimized placement + hot replication helps only modestly; top-1 locality is high but full top-k locality is low. |
| `05_top1_local_draft` | Complete | Cheaper top-1/2/4 local drafts still weak; best ~0.40 sampled acceptance on Qwen1.5-MoE. |
| `06_local_draft_policy` | Complete | Simple confidence/locality gating cannot isolate a high-acceptance subset. |
| `07_recent_model_acceptance` | Complete | Recent models better at low EP but collapse at high EP; acceptance and communication advantage are anti-correlated. |
| `08_negative_result` | Complete | Negative result scoped to no-shared-expert local-only drafting. |
| `09_rebalancing_ceiling` | Complete | Static replication needs ~0.5-0.8E experts per device for beta >= 0.8. |
| `10_temporal_locality_cache` | Complete | Verify-warmed/request-local cache halves the needed cache, but still scales with EP. |
| `11_intranode_ep_draft` | Complete | Hierarchical draft acceptance is feasible; timing needs high inter-node A2A fraction. |
| `12_ep_width_latency` | Complete | Single-node NVLink EP-width changes barely help; benefit is inter-node-bound. |
| `13_affinity_placement_gated_draft` | Complete | Affinity/request scheduling gives high G=2 acceptance but no draftable G=4 subset. |
| `14_intranode_semantic_pattern` | Complete | Semantic-Parallelism-like locality pattern is visible at G=2, but not an end-to-end speedup. |
| `15_replication_phi_envelope` | Complete | EPLB/cache replication makes G=4 viable (latency-bound, monotonic in phi); Qwen3 ~1.18x at R=2x iff inter-node A2A >= ~200us/coll. |
| `16_intranode_throughput` | Complete | Local draft halves MoE-FFN weight read at serving batch; real vLLM decode pins phi_moe~0.65-0.76 and T_draft/T_full~0.63; end-to-end throughput gain ~1.07x (b=0.85) to ~1.12x (b=0.9) at k=2, break-even by k=4. |
| `17_layerskip_local_draft` | Complete | Layer-skip is a bad trade: acceptance drops super-linearly vs linear cost saving, so layer-skip+local never beats pure local (~1.0x). Cheap-draft-via-skip lever fails. |
| `18_quantized_expert_throughput` | Complete | Serve-FP8 is ~2.2-3.4x but LOSSY. The lossless use is an FP8 draft + bf16 verify: FP8 is a near-perfect draft (acc 0.954 vs bf16) at 0.29x cost -> ~1.9x LOSSLESS throughput (the cheap draft layer-skip never was). Local routing is unnecessary under FP8. Cost: ~1.5x expert memory. FP8 gain peaks at moderate batch (~3.4x@B64), ~1x@B1. |
| `19_multinode_plan` | Plan | Novelty delineation: QuantSpec (ICML25) covers quantized-draft for KV/memory; SS-MoE covers expert-subset on-device. Ours = inter-node all-to-all amortization (local draft + full-EP verify), provably outside their scope. Specifies the 2-4 node go/no-go experiment. |
| `20_injection_emulation` | Complete | Real-IB loopback not viable here (fabric refuses NIC->NIC hairpin + self-loopback); emulate inter-node A2A via GPU-stream latency injection instead. Measured (Qwen3, attention-DP4+EP): ~145 exposed collectives/step, f=0.58-0.84 at 163-320us/coll. Break-even exposed latency is only ~11-18us/coll (3.3us for FP8-beta) -> lossless 1.1-3.3x across the plausible exposed-after-DBO range. Generalizes: GPT-OSS-20B (24L, MXFP4) -> N~70 (~3/layer, structural), f=0.55-0.72, speedup 1.4-2.3x -> win scales with MoE depth. Remaining gate: DBO overlap residual (needs 2-node IB), now with threshold d*~15-35us. |

## Timing Conclusion

Self-MoE-spec can only be useful in:

```text
low-batch multi-node EP decode
+ high exposed all-to-all after DeepEP/DBO
+ high local-only draft acceptance
```

Single-node EP is not enough. GPU6/7 local tests confirm this.

DeepEP and DBO should be treated as orthogonal components:

```text
local collective-free draft
+ DeepEP exact verification
+ DBO/phase-overlap at verification boundaries
```

In the current environment:

- DeepEP kernels are not installed.
- DBO cannot be measured on the target path.
- Forced same-host IB/GDRDMA fails below vLLM with `IBV_WC_RETRY_EXC_ERR`.
- Forced socket networking works but is not a practical multi-node IB proxy.

## Runtime Emulation Hook

Added research-only hooks:

```bash
VLLM_SELF_SPEC_EMULATE_A2A_DELAY_MS=<delay_ms>
VLLM_SELF_SPEC_LOG_A2A_COUNTS=1
VLLM_SELF_SPEC_A2A_COUNT_ACTIVE_FILE=<path>
```

These are for controlled local communication emulation. Defaults preserve normal
vLLM behavior.

Important caveat:

```text
VLLM_SELF_SPEC_EMULATE_A2A_DELAY_MS is per internal hook call,
not per decode step.
```

The current DP2+EP Qwen1.5-MoE path measured about:

```text
1536 active hook calls / worker / measured iteration
```

So delay values must be tiny and calibrated.

## Real Routing Coverage

### PowerMoE-3B

| EP size | Best simple `gamma` | Best simple top-k overlap | Full top-k local |
| ---: | ---: | ---: | ---: |
| 2 | ~0.59 | ~0.60 | ~0.5% |

### Qwen1.5-MoE-A2.7B

| EP size | Best simple `gamma` | Best simple top-k overlap | Full top-k local |
| ---: | ---: | ---: | ---: |
| 2 | ~0.62 | ~0.66 | ~13% |
| 4 | ~0.42 | ~0.46 | ~1% |
| 8 | ~0.32 | ~0.35 | <1% |

Hot expert replication helps but is still likely insufficient.

## Direct Acceptance Proxy

For Qwen1.5-MoE EP2, compared:

```text
full-router next-token distribution
vs.
local-masked-router next-token distribution
```

Result:

```text
best distribution overlap ~= 0.37
KL divergence: large
top-1 match: low / unstable
```

This is strong negative evidence for naive local-top-k drafting on this model.

Important caveat: this is still a **proxy**, not actual speculative acceptance.
It measures next-token distribution similarity after local router masking, but
it does not generate local draft sequences and verify them with the full model.
Actual acceptance rate `beta` remains unmeasured.

## Current Decision

Do **not** implement naive local-top-k drafting in vLLM yet.

The likely bottleneck is acceptance, not timing. Even if communication savings
exist in a multi-node setting, the local-only draft appears too divergent on the
tested real models. However, this conclusion is based on simple placements and
acceptance proxies, not optimized placement plus actual verification.

## Recommended Pivot

Important correction: the main Self-MoE-spec design should keep the model's
default number of routed experts and reroute them into local experts. For
Qwen1.5-MoE, this means:

```text
draft_expert_top_k = default expert_top_k = 4
```

Top-1/top-2 local drafts are separate cheaper-draft variants, not the main
method.

Phase 05 tested top-1/top-2 as a cheaper variant, and also measured the default
local top-4 case. The default local top-4 case still had low one-token sampled
acceptance (`~0.37-0.38` in the best tested placements). Phase 06 tested simple
selective policies and found they cannot isolate a high-acceptance subset.
Therefore, move to either stronger placement/policy for same-top-k local routing
or to a stronger compute-saving/learned component:

1. **Same-top-k local rerouting with stronger placement/policy**
2. **Layer-skip + local-routing hybrid**
3. **Trained/lightweight draft head**
4. **Negative-result framing:** quantify when expert locality is insufficient
   for lossless local MoE speculation.

## Phase 07 Result: Recent Models

Tested recent MoE checkpoints to check whether the Phase 03-06 negative result is
a model-choice artifact. Best sampled one-token acceptance (main method,
`draft_top_k = default`):

| Model | EP2 | EP4 | EP8 |
| --- | ---: | ---: | ---: |
| Qwen3-30B-A3B (128 exp) | 0.50 | 0.31 | 0.12 |
| GPT-OSS-20B (32 exp) | 0.70 | 0.58 | 0.52 |

Recent models beat Qwen1.5-MoE (~0.40 at EP2), and GPT-OSS-20B at EP2 (0.70) is
the project's best result. But the central finding is structural:

```text
acceptance is highest at low EP (smallest communication advantage)
and collapses at high EP (the only regime with a real communication advantage).
```

More EP shards -> fewer routed experts per shard -> lower local mass `gamma` ->
lower acceptance. None of the four tested checkpoints (all without a shared
expert) clears `beta >= 0.8`, and all are weakest in the high-EP multi-node
target regime. See `07_recent_model_acceptance/results_recent_models_acceptance.md`.

## Phase 08: Negative-Result Write-up (Complete)

The scoped negative result is written up in
`08_negative_result/negative_result_report.md`. Core claims: (1) local-only
drafting never reaches `beta >= 0.8` on the four tested no-shared-expert models;
(2) acceptance and exposed communication are anti-correlated along the EP axis;
(3) scope caveat: a shared-expert anchor is untested.

## Phase 09: Rebalancing Ceiling (Complete)

Tested whether locality-aware rebalancing/replication can lift high-EP
acceptance, by sweeping a per-device replicated draft-cache budget `M` (mass-
optimal per-layer set, EP-invariant) and measuring acceptance vs `M`. Result:

```text
acceptance rises monotonically with M (your hypothesis is directionally right),
but reaching beta >= 0.8 needs M* ~= 0.5-0.8 * E (half to four-fifths of all
experts) replicated on EVERY device.
```

The required replication factor over the plain-EP budget is
`R = M*/(E/N) ~= 0.5N..0.8N`, which grows **linearly with EP**: R ~= 4x..6x at
EP8, ~17x..25x at EP32. So at the cross-node scale where the communication
advantage is large, useful acceptance requires near-full per-device replication,
negating the memory rationale for EP. See
`09_rebalancing_ceiling/results_rebalancing_ceiling.md`.

| Budget (plain EP) | Qwen3-30B | GPT-OSS-20B |
| --- | ---: | ---: |
| M=E/8 (EP8) | 0.23 | 0.36 |
| M=E/4 (EP4) | 0.47 | 0.54 |
| M=E/2 (EP2) | 0.78 | 0.63 |
| beta>=0.8 needs | M~0.54E | M~0.79E |

## Phase 10: Verify-Warmed Draft Cache (Complete)

Tested whether exploiting the verify step's routing (temporal locality) lets a
small dynamic cache replace the large static one. A per-layer LRU expert cache
warmed by verified routing beats a static cache, but modestly:

- Dynamic vs per-request static coverage gap is only ~0.04-0.10.
- Most of the gain over Phase 09 is **per-request specialization**, not temporal
  recency (recency adds the smaller part).
- `beta >= 0.8` now needs `C ~= 0.31 E` (Qwen3) / `0.47 E` (GPT-OSS) -- about
  **half** the Phase 09 static `M*`, a ~1.7-2x improvement.
- But the cache is still replicated, so `R = C*N/E ~= 0.3N..0.5N` still grows
  linearly with EP (~2.5-4x at EP8, ~10-15x at EP32). The high-EP problem is
  softened ~2x, not removed.

Acceptance validation: Qwen3 C=32 -> 0.74, GPT-OSS C=16 -> 0.83. See
`10_temporal_locality_cache/results_temporal_locality.md`.

## Phase 11: Hierarchical-EP Draft Feasibility (Complete)

Idea (user): don't disable EP for the draft -- flatten it to **intra-node EP**
(NVLink all-to-all only), verify with full EP (inter-node IB). The draft then sees
`E/nodes` experts and avoids only the expensive inter-node hop.

- **Acceptance: feasible.** Realistic contiguous node shard is weak (Qwen3 0.33 at
  2 nodes, 0.12 at 4), but a small per-node high-coverage cache (~5-8 experts/GPU,
  cheap because the node aggregates G GPUs) lifts it to ~0.74-0.78 at 2-4 nodes.
- **Timing: borderline, hardware-bound.** Measured NVLink A2A ~30 us/collective;
  a decode step has `2 x num_layers` collectives (Qwen3: 96 -> ~2.9 ms intra-node
  A2A). The draft still pays full compute + intra-node A2A, so it wins only when
  the inter-node fraction `f_inter >= ~0.5` (needs ~200 us/coll IB latency + lean
  compute). At `f_inter ~ 0.3` it is marginal (0.88 at b=0.74). Forcing a slow
  NCCL transport crashes here, so inter-node is modeled; a real go/no-go needs a
  2-4 node IB testbed. See `11_intranode_ep_draft/results_feasibility.md`.

This is the most promising framing: it moves acceptance from fatal (~0.1) to
feasible (~0.74) at low cost. The open gate is purely whether inter-node
all-to-all is a large enough fraction of the decode step.

## Phase 12: Exact EP-Width Latency (Complete)

Real vLLM decode latency (Qwen3-30B-A3B dummy, single-node NVLink, GPUs 2-5)
replacing the Phase 11 analytical timing. In this vLLM `ep_size = dp_size*tp_size`,
so EP2xDP2 = two independent `TP2+EP` engines, EP4xDP1 = one `TP4+EP` engine.

| | B=2 | B=8 | B=16 | B=32 |
| --- | ---: | ---: | ---: | ---: |
| EP4xDP1 = L4(B) ms/tok | 5.35 | 7.16 | 8.53 | 9.87 |
| EP2xDP2 = L2(B/2) ms/tok | 5.01 | 7.11 | 9.05 | 11.46 |

EP2xDP2 is only marginally faster at low batch (<=7% at B<=4), ties at B=8, loses
at B>=16. At equal batch EP4 is always faster (more GPUs/compute). So on NVLink,
reducing EP barely helps (and hurts at scale): the intra-node all-to-all is too
cheap for EP-width to matter. The hierarchical draft therefore pays off **only**
on the inter-node all-to-all, which this single node cannot produce. See
`12_ep_width_latency/results_ep_width_latency.md`.

This is the measured confirmation of the Phase 11 gate: acceptance is favorable,
but the timing benefit is inter-node-bound and unverifiable without a 2-4 node IB
testbed.

## Phase 13/14: Semantic-Parallelism-like Affinity Scheduling

Tested a Speculative-MoE/Semantic-Parallelism-style idea: cluster co-activated
experts and assign each request to the group with best coverage, then measure
local activation rate (LAR), draftable high-coverage positions, and one-step
acceptance.

At `G=2`, the pattern matches the paper qualitatively:

| Model | Contiguous LAR | Affinity LAR | Remote-volume reduction | All-step acc | Draftable acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-30B-A3B | 0.511 | 0.827 | 64.6% | 0.858 | 0.869 |
| GPT-OSS-20B | 0.512 | 0.755 | 49.7% | 0.840 | 0.888 |

At `G=4`, affinity still improves LAR, but absolute LAR remains below 0.5 and
there is no `coverage >= 0.9` draftable subset. Therefore the paper-like
intra-node locality pattern is easy to show as a routing/volume result, but a
large single-node end-to-end vLLM speedup is unlikely because Phase 12 measured
NVLink EP communication as too cheap relative to decode compute. See
`14_intranode_semantic_pattern/results_intranode_semantic_pattern.md`.

## Final State

Pure single-GPU local-only MoE drafting is not viable in the high-EP target
regime. The chain of evidence:

- Acceptance is too low and **anti-correlated** with the EP regime that supplies
  the communication advantage (Phase 07).
- A static rebalancing fix needs an infeasible `R ~ N/2` replication (Phase 09).
- A verify-warmed / per-request cache halves that to `R ~ 0.3N..0.5N` (Phase 10) --
  plausible at moderate EP with affordable warm-cache memory, but still
  near-full replication at cross-node EP.
- Semantic-Parallelism-like affinity/request scheduling shows the desired
  intra-node locality pattern at G=2, but does not provide a high-coverage G=4
  draftable subset and does not overturn the single-node timing limit (Phase 14).

The negative-result write-up is `08_negative_result/negative_result_report.md`.
The single remaining untested lever is an **EP-invariant shared expert** (fixed
always-local mass independent of replication, the only mechanism that escapes the
`R ~ cN` scaling). If revisited, run the Phase 07 proxy + Phase 09/10 sweeps on a
shared-expert MoE (DeepSeek / Llama-4 / GLM). Otherwise the negative result stands,
now with the rebalancing and temporal-cache mitigations quantified and bounded.
