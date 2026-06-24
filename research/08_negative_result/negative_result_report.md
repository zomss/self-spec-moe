# When Expert Locality Is Insufficient for Lossless Local-Draft MoE Speculation

Date: 2026-06-24

Status: negative result, scoped to MoE checkpoints **without a shared expert**.

## 1. Summary

We evaluated **Self-MoE-spec**: a lossless self-speculative decoding scheme for
expert-parallel (EP) MoE serving in which draft steps route only to
**device-local experts** (no expert-parallel all-to-all) and verification uses
exact full routing. The intended payoff is communication amortization: pay the
all-to-all collective once per verify cycle instead of once per token.

Across four MoE checkpoints spanning old/recent and fine/coarse expert
granularity, **local-only drafting does not reach a useful acceptance rate**, and
it is **weakest exactly in the high-EP multi-node regime** where the
communication advantage would be largest. The two conditions the method needs
are anti-correlated. We therefore do not recommend building a vLLM runtime path
for local-only MoE drafting on these architectures.

The result is scoped: every model tested lacks a shared expert. A large
always-local shared expert carries EP-invariant gate mass and is the one untested
mechanism that could change the conclusion (Section 8).

## 2. Method under test

```text
1. Draft for num_spec_tokens steps with the model's default expert top-k, but
   restrict MoE routing to device-local experts -> draft steps avoid EP all-to-all.
2. Verify drafted tokens with exact full-routing MoE.
3. Use standard speculative rejection sampling -> output distribution is lossless
   regardless of how badly local routing perturbs the draft.
```

Target regime: low-batch, decode-only, latency-sensitive, multi-node EP, where
exposed all-to-all is on the critical path.

## 3. Cost model and the regime

The draft is **not compute-cheap**: a local-routing draft step still runs full
attention and full top-k expert FLOPs (`c ~= 1`); it only removes the all-to-all.
So this is purely a communication-amortization scheme. With `f` = exposed
all-to-all fraction of a decode step and `E` = expected tokens per cycle:

```text
speedup ceiling (perfect acceptance) = 1 / (1 - f)
per-cycle break-even:  E > 1 + num_spec_tokens * c * (1 - f)
```

Two consequences frame everything below:

- The advantage exists only when `f` is large, i.e. **high-EP / cross-node** decode.
- Because `c ~= 1`, acceptance must be high to break even (an `c ~= 1` draft is a
  ~2x compute tax per output token if rejected).

## 4. Evaluation methodology

Two offline studies, no runtime build required:

- **Timing envelope** (Phases 01-02): normalized and measured `T_base`,
  `T_draft_local`, `T_verify` plugged into the speedup formula, treating DeepEP
  and DBO as orthogonal components.
- **Acceptance proxy** (Phases 03-07): run the target MoE, simulate device-local
  expert placement by masking the router to local experts, and measure the
  one-token speculative accept rate against the exact full-routing distribution.

Acceptance metrics: `gamma` (local gate mass), top-k overlap, expected acceptance
`sum_i min(p_i, q_i)`, Monte-Carlo sampled acceptance, `KL(p||q)`, and top-1
(greedy) match. Placements: contiguous / random / hot-replicated.

**Proxy limitations (these make the numbers optimistic):** it is one-token, not
multi-token (real `beta` compounds divergence lower across draft positions); it
does not run the full speculative loop with approximate draft KV; it uses a small
domain-bucketed prompt set.

## 5. Evidence: timing

- **Single-node EP is too weak.** At `f ~= 0.15`, the perfect-acceptance ceiling
  is only ~1.05-1.14x, and ~1.05x on top of a DBO-equipped stack.
- **Multi-node has analytical room but a high bar.** At `f = 0.6`, ~1.3x over a
  DeepEP-only verify stack needs `beta >= 0.8`; on top of DeepEP+DBO it needs
  `beta >= 0.98`.
- **The favorable regime could not be measured here.** Single-node NVLink EP is
  too fast; forced same-host IB/GDRDMA fails below vLLM with
  `IBV_WC_RETRY_EXC_ERR`; DeepEP kernels are not installed. A research-only
  per-collective delay-injection hook was added for controlled emulation, but it
  is not a substitute for a real multi-node fabric.

Net: timing room exists only at high EP and is hard to win on top of DBO.

## 6. Evidence: acceptance

Best sampled one-token acceptance, main method (`draft_top_k = default top-k`),
best placement:

| Model | Released | Experts / top-k | Shared | EP2 | EP4 | EP8 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| PowerMoE-3B | 2024 | 32 | no | weak (~0.4-class) | - | - |
| Qwen1.5-MoE-A2.7B | 2024 | 60 / 4 | small | 0.40 | weak | <0.1-class |
| Qwen3-30B-A3B | 2025 | 128 / 8 | no | 0.50 | 0.31 | 0.12 |
| GPT-OSS-20B | 2025 | 32 / 4 | no | 0.70 | 0.58 | 0.52 |

Required: `beta >= 0.8` (high `f`), `>= 0.9` (moderate / DBO-composed), `>= 0.98`
(1.3x on DBO at `f = 0.6`). No model, at any EP size, reaches the lowest bar.

Cheaper top-1 / top-2 local drafts (Phase 05) and confidence/locality gating
policies (Phase 06) did not rescue acceptance: the best gated subset on
Qwen1.5-MoE was still ~0.40 at a tiny ~5.6% selected fraction.

**Greedy caveat.** GPT-OSS reaches 0.70 sampled (temperature 1) but only 0.38
top-1 match; under low-temperature/greedy decode (the latency-critical case) its
effective acceptance is ~0.38. Qwen3's top-1 (0.56 at EP2) exceeds its sampled,
but is lower overall.

## 7. Core finding: acceptance and communication advantage are anti-correlated

The decisive structural observation:

```text
Local-draft acceptance is HIGHEST at low EP, where the exposed all-to-all
fraction f is SMALL and the communication ceiling 1/(1-f) is near 1.

Acceptance COLLAPSES at high EP, which is the ONLY regime where f is large
enough for the method to be worth doing.
```

Mechanism: increasing EP shards the routed experts across more devices, so each
device holds fewer experts, local gate mass `gamma` falls, top-k overlap falls,
and acceptance falls. Meanwhile higher EP (especially cross-node) is precisely
what raises `f`. The method's two preconditions move in opposite directions along
the EP axis.

GPT-OSS-20B degrades more gracefully than Qwen3-30B (0.52 vs 0.12 at EP8) only
because it is coarser (32 vs 128 experts), so each shard retains more mass. But
coarse models are also the least likely to be served at high cross-node EP, so
this does not recover a viable operating point.

This is a stronger statement than "acceptance is low": even if acceptance could
be pushed up, doing so requires low EP, which removes the reason to use the method
at all.

## 7b. Rebalancing cannot fix it affordably

The natural mitigation is locality-aware rebalancing/replication: give each device
more of the experts its tokens want. Phase 09 measured the **ceiling** of this
approach by sweeping a per-device replicated draft-expert cache of budget `M`
(mass-optimal per-layer set, identical on every device so it is EP-invariant) and
measuring acceptance vs `M`.

Acceptance rises monotonically with `M` (the rebalancing intuition is
directionally correct), but the budget required is prohibitive:

| Budget = plain-EP share `E/N` | Qwen3-30B | GPT-OSS-20B |
| --- | ---: | ---: |
| `M = E/8` (EP8) | 0.23 | 0.36 |
| `M = E/4` (EP4) | 0.47 | 0.54 |
| `M = E/2` (EP2) | 0.78 | 0.63 |
| `beta >= 0.8` needs | `M ~= 0.54 E` | `M ~= 0.79 E` |

Reaching `beta >= 0.8` requires replicating **half to four-fifths of all experts
on every device.** Expressed as a replication factor over the plain-EP per-device
budget:

```text
R = M* / (E/N) = N * (M*/E)  ~=  0.5 N .. 0.8 N
```

`R` grows **linearly with EP**: ~4-6x at EP8, ~17-25x at EP32. So at the
cross-node scale where exposed communication is large (the only place the method
helps), useful acceptance demands near-full per-device replication, which negates
the memory rationale for EP and shrinks the very all-to-all being amortized. Cost
and benefit both scale with `N`. This is the mass-optimal fixed cache under a
one-token proxy, so heuristic placement / multi-token / greedy decode are strictly
worse.

## 8. Scope and what would overturn this

The negative result is scoped to **routed-expert-only locality on models without a
shared expert.** The one mechanism not tested:

- A **large always-local shared expert** (DeepSeek / Llama-4 / GLM style)
  contributes the same gate mass on every device regardless of EP size. Because
  this mass is EP-invariant, it could provide an acceptance floor that does not
  collapse at high EP, directly countering the Section 7 anti-correlation.

The decisive follow-up experiment (if revisited) is to run the Phase 07 proxy on
a shared-expert MoE and check whether high-EP acceptance stays useful. If it also
collapses at high EP, the negative result generalizes; if it holds, the method is
alive specifically for shared-expert architectures, and the contribution becomes
"local-draft MoE speculation requires a shared-expert anchor."

Other threats to validity, all of which make the present numbers **optimistic**:
one-token proxy (multi-token would be lower), no real multi-node timing
measurement, and a small prompt set without request-grouping by routing locality.

## 9. Takeaway

For MoE checkpoints without a shared expert, expert locality is insufficient to
make local-only drafts survive exact verification at a useful rate, and the
locality that does exist is concentrated at low EP where the communication
advantage is negligible. Rebalancing/replication does raise acceptance but needs
a per-device replication factor `R ~ 0.5N..0.8N` to reach `beta >= 0.8`, which is
infeasible at the high EP the method targets. The reusable contributions are the
**acceptance-vs-EP characterization**, the **anti-correlation between local-draft
acceptance and exposed communication**, and the **rebalancing ceiling**
(`R ~ N/2` replication required), which together quantify when expert locality
can and cannot be converted into fewer all-to-all collectives per accepted token.

## References (in-repo)

- Timing envelope: `../01_communication_advantage/`, `../02_timing_envelope_experiment/`
- Acceptance, old models: `../03_local_routing_acceptance/`, `../04_expert_placement_acceptance/`,
  `../05_top1_local_draft/`, `../06_local_draft_policy/`
- Acceptance, recent models: `../07_recent_model_acceptance/results_recent_models_acceptance.md`
- Rebalancing ceiling: `../09_rebalancing_ceiling/results_rebalancing_ceiling.md`
- Consolidated status: `../status.md`
