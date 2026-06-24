# Literature Review: Local-Expert Self-Speculative Decoding for MoE

Date: 2026-06-22

## Research Direction

This project explores a lossless speculative decoding method for expert-parallel
Mixture-of-Experts (MoE) serving:

1. Draft for `k` steps using the same MoE model, but restrict MoE routing to
   device-local experts so draft steps avoid expert-parallel all-to-all.
2. Verify the drafted tokens with normal full expert-parallel routing.
3. Use standard speculative decoding verification to preserve the target model
   output distribution.

The target regime is low-batch, decode-only, latency-sensitive, multi-node
expert-parallel serving where exposed all-to-all communication is on the
critical path. The key variables are local expert mass (`gamma`), acceptance
rate (`beta`), draft length (`k`), batch size, expert placement, and the
fraction of decode latency still exposed after DeepEP/DBO-style overlap.

## Go/No-Go Summary

| Question | Finding | Verdict |
| --- | --- | --- |
| Has this exact idea already been done? | No exact match found. The closest adjacent work is SS-MoE, which uses expert-subset self-speculation for on-device/offloaded MoE, not multi-node EP all-to-all elimination. | Go |
| Is routing locality strong enough to make acceptance plausible? | Existing work shows domain/task routing locality, prefill-decode correlation, and expert-placement benefits, but no paper directly measures local-only draft acceptance. | Conditional go |
| Can it add value with DeepEP/DBO and compare well to MTP/EAGLE/layer-skip? | There is a plausible but narrow regime: low batch, multi-node EP, decode-only, high exposed all-to-all, and high local expert mass. Incremental value is small when DBO already hides most communication or batch size is large. | Conditional go |

## Q1: Has Anyone Already Done Local-Expert Draft + Full MoE Verification?

No exact match was found for:

- local-expert-only draft routing,
- no draft-time EP all-to-all,
- full exact EP verification,
- multi-node expert-parallel serving,
- lossless speculative decoding semantics.

The closest work is **SS-MoE: Self-Speculative Decoding for On-device MoE
Acceleration**. SS-MoE uses a subset of routed experts as a draft path and
verifies with additional/full experts. However, its setting is on-device MoE
acceleration and expert caching/offloading, not distributed EP communication
avoidance. It is therefore adjacent but not duplicative.

Other nearby MoE speculative decoding papers attack different bottlenecks:

- **MoE-Spec** budgets experts during verification, reducing memory bandwidth
  by dropping low-utility experts. This is almost the inverse of this project:
  it restricts verification, while this project keeps verification exact and
  restricts only draft routing.
- **EVICT** adaptively truncates draft trees before verification to avoid
  activating too many unique experts. It is lossless and MoE-aware, but does
  not make draft steps collective-free.
- **Cascade** shows why ordinary speculative decoding can be harmful for MoE:
  speculative verification can activate 2-3x more expert weights and cause up
  to 1.5x slowdown. This is a strong motivation for avoiding expert expansion
  rather than an implementation of local-only drafting.
- **ELMoE-3D** uses MoE expert elasticity for self-speculative decoding, but in
  a hardware/on-premises serving context rather than distributed EP all-to-all
  avoidance.

### Q1 Verdict

**Go.** The idea appears novel enough if framed precisely as
communication-free local expert drafting plus exact full-EP verification. The
paper must clearly distinguish itself from SS-MoE, MoE-Spec, EVICT, and
Cascade.

## Q2: Does Expert Locality Make Acceptance Plausible?

The literature gives encouraging but incomplete evidence.

### Positive Evidence

**Scaling Multi-Node Mixture-of-Experts Inference Using Expert Activation
Patterns** profiles frontier MoE models including Llama 4 Maverick,
DeepSeek V3-671B, and Qwen3-230B-A22B. It reports:

- variable expert load imbalance,
- domain-specific expert activation,
- strong correlation between prefill and decode expert activations,
- workload-aware grouping and placement that can reduce all-to-all
  communication data by up to 20x.

This supports the idea that request grouping and expert placement can increase
local expert mass.

**Not All Models Suit Expert Offloading** studies local routing consistency
across 20 MoE LLMs and introduces segment-level expert-locality metrics:

- Segment Routing Best Performance (SRP),
- Segment Cache Best Hit Rate (SCH).

This is directly relevant because local-only drafting is effectively a
fixed-cache/local-shard constraint over a short decode segment. The paper also
warns that local routing consistency is model-dependent.

**MoETuner** exploits token routing dependency across layers: tokens routed to
a specific expert in one layer are likely to route to a limited set of experts
in the next layer. It uses expert placement to jointly reduce communication,
load imbalance, and tail latency, reporting 9.3% single-node and 17.5%
multi-node speedups.

**ExpertFlow** predicts expert usage paths and schedules tokens with similar
routes for expert caching. It targets offloading, but its success suggests MoE
routing has enough predictability to exploit.

### Missing Evidence

The key metric for this project is not directly reported by prior work:

```text
gamma = probability mass, top-k overlap, or routed-expert coverage
        available on the local EP shard under a chosen placement.
```

The first experiment must measure:

- per-token top-k overlap between full routing and local-only routing,
- local expert mass (`gamma`) per layer and per request,
- how `gamma` changes with random placement, locality-aware placement,
  hot-expert replication, and shared-expert handling,
- speculative acceptance rate (`beta`) for draft lengths `k = 1..N`,
- degradation across draft positions due to approximate KV/history.

### Q2 Verdict

**Conditional go.** The locality signal is real, but it is not yet the exact
acceptance signal required by this project. The first milestone should be a
measurement study, not a runtime implementation.

## Q3: Is There a Competitive Regime?

This method is a communication-amortization scheme, not primarily a
compute-reduction scheme. A local-only draft step still runs attention and local
expert compute; it mainly removes draft-time all-to-all.

It is most likely to help when:

- decode batch size is low,
- serving is latency-sensitive rather than throughput-saturated,
- expert parallelism crosses nodes,
- all-to-all remains exposed after DeepEP/DBO overlap,
- requests can be grouped by similar routing locality,
- local expert placement gives sufficiently high acceptance.

It is likely weak when:

- EP is single-node NVSwitch only,
- batch size is large enough that verification all-to-all becomes bandwidth
  dominated,
- prefill and decode are mixed in the same batch,
- DBO already hides most of the communication,
- MTP/EAGLE/layer-skip gives high acceptance with cheaper draft compute.

### System Positioning

| Method | Role | Why it matters |
| --- | --- | --- |
| DeepEP | Orthogonal EP communication backend | Use it for exact verification collectives; Self-MoE-spec reduces how often those collectives occur per accepted token. |
| vLLM DBO | Partially orthogonal overlap scheduler | Use it inside verification or at phase boundaries; avoid token-level draft/verify fusion. |
| MTP | Native trained multi-token prediction capability | Strong target-model-native comparison; not specifically designed for local EP communication avoidance. |
| EAGLE/EAGLE-2 | Trained draft heads and dynamic draft trees | Strong speculative decoding comparison, but not MoE communication-locality aware. |
| LayerSkip, Draft & Verify, SWIFT, CLaSp | Self-speculative layer skipping | Saves compute by skipping layers; does not directly remove MoE all-to-all. |
| Cascade | Utility-driven speculation for MoE | Shows naive SD can hurt MoE and provides a dynamic enable/disable comparison. |
| MoE-Spec/EVICT | MoE-aware verification cost reduction | Important MoE-specific SD comparisons; solve expert expansion differently. |

### Q3 Verdict

**Conditional go.** The credible win regime exists, but it is narrow:

```text
multi-node EP + low decode batch + high exposed all-to-all
+ high local expert mass + cycle-aligned scheduling.
```

The project should avoid claiming a general speculative decoding improvement.
It should claim a specific MoE systems result: converting expert locality and
placement quality into fewer all-to-all collectives per accepted token.

### Batch Size and Kernel Implication

Self-MoE-spec is not simply better when the batch is larger. It helps when
**exposed communication latency** is large enough to amortize, not when total
communication bytes are large. As batch size grows, the full verification step
must route `B * (k + 1)` tokens. Once verification becomes bandwidth-bound,
`T_verify(B, k)` approaches the cost of routing those tokens autoregressively,
so the communication-amortization advantage disappears while draft compute
remains as overhead.

The strongest expected regime is therefore:

- low-to-moderate decode batch size,
- multi-node EP rather than single-node NVSwitch-only EP,
- decode-only batches with prefill disaggregated,
- exposed all-to-all that DeepEP/DBO cannot fully hide,
- high local-only draft acceptance from placement/request locality.

Single-node EP is mainly a prototype and measurement environment. On one
NVSwitch/NVLink node, all-to-all is comparatively fast, so the exposed
communication fraction is likely small. If exposed communication is only
10-20% of decode latency, the ideal ceiling from removing draft-step
collectives is only about 1.1-1.25x before rejected drafts, scheduler overhead,
and extra local draft compute. Single-node experiments are still useful for
debugging local routing, measuring `gamma` and `beta`, and validating placement
simulation, but they are unlikely to be the strongest systems result.

This does **not** require disabling every advanced kernel. DeepEP low-latency
can still be the right communication backend. However, if DBO or another
overlap scheduler already hides most of the all-to-all critical path, the
remaining speedup ceiling becomes small. In practice, the method is most
interesting when DBO is unavailable, disabled, or ineffective at low batch.

### Success Criterion and Ablations

DeepEP and DBO should be treated as orthogonal components, not competitors. The
strongest positive result is not merely beating plain EP decode. The research
direction is compelling if adding Self-MoE-spec improves the best configured
communication stack in its intended regime:

- DeepEP low-latency verification without local draft,
- DeepEP low-latency verification with DBO or phase-overlap enabled,
- the same stack plus Self-MoE-spec local draft,
- MTP/EAGLE/layer-skip comparisons when the target model supports them,
- Cascade-style dynamic speculative decoding for MoE if available.

If adding Self-MoE-spec improves **DBO + DeepEP** at low-batch multi-node
decode, the central claim is strong: communication overlap alone is insufficient
because low-batch decode does not expose enough independent compute to hide
all-to-all, while local draft steps both reduce collective frequency and create
collective-free work.

The comparison must be fair:

- same model, hardware, parallelism, prompt/output distribution, and SLO,
- DBO and DeepEP tuned with their recommended decode settings,
- prefill separated from decode or reported separately,
- scheduler overhead, cycle-boundary waiting, and rejected draft work included,
- both TPOT latency and throughput reported.

The claim should be rejected if speedup appears only when DBO is disabled,
misconfigured, or evaluated outside the target low-batch decode regime.

### Orthogonality with DeepEP and DBO

DeepEP and Self-MoE-spec are mostly orthogonal. Local draft steps avoid
expert-parallel all-to-all entirely, while verification still needs exact full
routing. Therefore, the verify step should use the best available EP backend,
likely DeepEP low-latency for decode. In this framing, Self-MoE-spec reduces the
number of verification collectives per accepted token, and DeepEP minimizes the
cost of each remaining collective.

DBO is only partially orthogonal:

- **Helpful inside verification:** the full verify step can still use DBO-style
  microbatch overlap to reduce exposed all-to-all during exact routing.
- **Helpful as phase-offset scheduling:** one request group can verify while
  another group performs communication-free local draft steps. This preserves
  collective-free drafting and may provide more overlap material than ordinary
  low-batch DBO.
- **Harmful if used as token-level fusion:** mixing draft and verify requests in
  the same normal engine step reintroduces all-to-all into every draft step,
  destroying the main advantage.

The best system design is therefore not "Self-MoE-spec instead of DeepEP/DBO."
It is:

```text
local collective-free draft
+ DeepEP for exact verification collectives
+ DBO or DBO-like overlap only at verification/phase boundaries
```

This combination is the fair target to evaluate. If Self-MoE-spec only improves
DeepEP without DBO, the result is weaker. If it adds speedup on top of DeepEP
plus correctly applied DBO/phase-overlap, the contribution is much stronger.

## Comparison Table

| Work | Mechanism | Relation to this project | Citation |
| --- | --- | --- | --- |
| SS-MoE | Expert-subset self-speculation for on-device MoE acceleration | Closest adjacent work; not multi-node EP all-to-all elimination | [ACM DOI](https://dl.acm.org/doi/10.1145/3774904.3792218) |
| MoE-Spec | Verification-time expert budgeting | Restricts verify-side experts; this project restricts only draft routing and keeps verify exact | [arXiv:2602.16052](https://arxiv.org/abs/2602.16052) |
| EVICT | Lossless adaptive verification for MoE draft trees | Reduces verification expert expansion, not draft-time all-to-all | [arXiv:2605.00342](https://arxiv.org/abs/2605.00342) |
| Cascade | Utility-driven speculative decoding for MoE | Shows naive MoE SD can slow down due to expert expansion | [arXiv:2506.20675](https://arxiv.org/abs/2506.20675) |
| ELMoE-3D | Expert/bit elasticity for self-SD with hardware co-design | Adjacent expert-axis self-speculation, but not EP communication-focused | [arXiv:2604.14626](https://arxiv.org/abs/2604.14626) |
| SpecMoEOff | Uses SD to hide expert offloading latency | Offloading-focused, not distributed EP A2A elimination | [arXiv:2508.21706](https://arxiv.org/abs/2508.21706) |
| SpecMD | Studies speculative expert caching/prefetching | Supports routing predictability; offloading-focused | [arXiv:2602.03921](https://arxiv.org/abs/2602.03921) |
| ExpertFlow | Predictive expert caching and token scheduling | Shows route prediction and grouping can be useful | [arXiv:2410.17954](https://arxiv.org/abs/2410.17954) |
| Scaling Multi-Node MoE Inference Using Expert Activation Patterns | Profiles expert activation locality and placement | Strong evidence for domain/routing locality in multi-node MoE | [arXiv:2604.23150](https://arxiv.org/abs/2604.23150) |
| Not All Models Suit Expert Offloading | Defines local routing consistency metrics | Useful measurement framework for local-only drafting | [arXiv:2505.16056](https://arxiv.org/abs/2505.16056) |
| MoETuner | Expert placement balancing load, communication, and computation | Supports placement-locality co-design | [arXiv:2502.06643](https://arxiv.org/abs/2502.06643) |
| DeepEP | Low-latency/high-throughput EP communication kernels | Orthogonal verification communication backend | [GitHub](https://github.com/deepseek-ai/DeepEP) |
| vLLM DBO | Dual-batch overlap for DP+EP MoE | Partially orthogonal overlap mechanism for verification/phase boundaries | [vLLM docs](https://github.com/vllm-project/vllm/blob/main/docs/design/dbo.md) |
| Draft & Verify | Training-free layer-skip self-speculation | Self-SD baseline; compute-saving, not MoE-communication-specific | [arXiv:2309.08168](https://arxiv.org/abs/2309.08168) |
| LayerSkip | Early exit and self-speculative decoding | Strong self-SD baseline, but requires training recipe | [arXiv:2404.16710](https://arxiv.org/abs/2404.16710) |
| SWIFT | Training-free adaptive layer-skipping self-SD | Plug-and-play self-SD baseline | [OpenReview](https://openreview.net/forum?id=EKJhH5D5wA) |
| CLaSp | In-context layer skip for self-SD | Training-free layer-skip baseline | [arXiv:2505.24196](https://arxiv.org/abs/2505.24196) |
| EAGLE | Feature-level speculative sampling | Strong trained SD baseline; not MoE locality-aware | [arXiv:2401.15077](https://arxiv.org/abs/2401.15077) |
| EAGLE-2 | Dynamic draft trees | Strong SD baseline; EVICT builds on this style for MoE | [arXiv:2406.16858](https://arxiv.org/abs/2406.16858) |
| DeepSeek-V3 MTP | Native multi-token prediction objective | Strong MoE-native baseline; not specifically a local-EP draft method | [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) |

## Recommended First Experiments

Do not start with vLLM runtime changes. Start with a communication-only
advantage envelope. This first step isolates the core upside of
communication-disabled draft before measuring routing quality:

```text
speedup(beta, k) =
    E(beta, k) * T_base(B)
    / (k * T_draft_local(B) + T_verify(B, k))

E(beta, k) = 1 + beta + beta^2 + ... + beta^k
```

Measure or estimate:

- `T_base(B)`: normal EP decode step latency for batch size `B`,
- `T_draft_local(B)`: local-only draft step latency with no all-to-all,
- `T_verify(B, k)`: exact verification latency for `B * (k + 1)` tokens,
- `S_max`: speedup at `beta = 1`,
- `beta_min`: acceptance needed to break even or hit a target speedup.

Phase 01 implements this communication-envelope study in
`../01_communication_advantage`. Use its `speedup_envelope.py` to compute this
table from measured timings:

```bash
.venv/bin/python research/01_communication_advantage/speedup_envelope.py timings.csv \
    --target-speedup 1.3
```

For acceptance-rate sweeps, pass comma-separated beta values:

```bash
.venv/bin/python research/01_communication_advantage/speedup_envelope.py timings.csv \
    --acceptance-rates 0.5,0.6,0.7,0.8,0.9,1.0
```

The input CSV schema is:

```csv
reference_stack,batch_size,draft_length,t_base_ms,t_draft_local_ms,t_verify_ms
deepep_ll_verify,1,4,6.0,3.0,7.0
```

Only if `S_max` is meaningfully above the strongest reference stack should the
project move to Phase 02 measured timing experiments. After Phase 02 confirms
the measured timing envelope, the next phase should run an offline
routing/acceptance simulation:

1. Run the target MoE model and record gate logits/top-k experts per layer.
2. Simulate several expert placements:
   - random EP placement,
   - locality-aware placement,
   - hot-expert replication,
   - shared-expert always-local placement if the model has shared experts.
3. For each placement, measure:
   - local expert mass (`gamma`),
   - top-k overlap with full routing,
   - per-layer and per-token route divergence,
   - draft token acceptance rate (`beta`) under local-only routing,
   - acceptance decay across draft positions,
   - expected speedup after plugging in measured all-to-all latency.
4. Evaluate by ablation against:
   - plain autoregressive EP decode,
   - DeepEP low-latency verification without local draft,
   - DeepEP plus DBO or phase-overlap without local draft,
   - DeepEP plus DBO or phase-overlap with local draft,
   - MTP/EAGLE/layer-skip where supported,
   - Cascade-style dynamic enable/disable.

The go/no-go threshold should be stated in terms of measured `beta`, exposed
all-to-all fraction, and batch size. If local-only `beta` is too low under
realistic placement, the system implementation should not proceed.

## Final Positioning

The strongest paper framing is:

> Local-expert self-speculation turns MoE expert locality into fewer
> communication collectives per accepted token.

This is distinct from standard speculative decoding, which mainly saves compute,
and distinct from MoE communication work, which mainly optimizes or overlaps the
same all-to-all operations. The open question is whether local-only routing gives
enough draft fidelity to make the communication savings survive verification.
