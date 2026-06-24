# Results: Affinity Placement + Gated Local Draft

Date: 2026-06-24

Semantic-Parallelism-style affinity placement (per-layer co-activation clustering)
+ oracle request-to-group scheduling on no-shared-expert models. `num_groups` (G)
is the draft locality / node count. One-step acceptance proxy.

## Qwen3-30B-A3B (E=128, top-8)

| placement | G | LAR | cov p90 | draftable (>=0.9) | acc all | acc draftable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| affinity | 2 | **0.827** | 0.945 | 36.1% | **0.858** | **0.869** (n=37) |
| affinity | 4 | 0.468 | 0.604 | 0.0% | 0.453 | - |
| contiguous | 2 | 0.511 | 0.549 | 0.0% | - | - |
| contiguous | 4 | 0.269 | 0.305 | 0.0% | - | - |

## GPT-OSS-20B (E=32, top-4)

| placement | G | LAR | cov p90 | draftable (>=0.9) | acc all | acc draftable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| affinity | 2 | **0.755** | 0.927 | 19.9% | **0.840** | **0.888** (n=21) |
| affinity | 4 | 0.417 | 0.562 | 0.0% | 0.590 | - |
| contiguous | 2 | 0.512 | 0.562 | 0.0% | - | - |
| contiguous | 4 | 0.267 | 0.312 | 0.0% | - | - |

## Findings

**1. Affinity placement works, and strongly (the user's idea is validated).**
Co-activation clustering + request-to-group scheduling nearly doubles LAR over
contiguous at G=2 (Qwen3 0.83 vs 0.51, GPT-OSS 0.76 vs 0.51), matching the
Semantic-Parallelism premise (it reported ~25% -> ~46% LAR; the oracle assignment
here is more optimistic).

**2. At G=2 the acceptance side finally clears the bar -- the first positive in
the whole investigation.** With affinity placement, G=2 local-draft acceptance is
**0.86 (Qwen3) / 0.84 (GPT-OSS) across *all* steps**, and 0.87 / 0.89 on the
draftable subset (36% / 20% of steps at coverage >= 0.9). Both exceed `beta >= 0.8`.
Even ungated drafting works at G=2 because LAR is so high (0.76-0.83).

**3. But it does not extend to G=4.** At G=4, even affinity placement gives
LAR 0.42-0.47, *no* draftable subset (0%), and acceptance 0.45-0.59. With 4 groups
each holding E/4 experts, a request's routing across all layers cannot be
concentrated into one quarter of the experts.

**4. No step is ever fully local** (fully_local = 0% everywhere): across 24-48
layers some layer always routes outside the group, so drafting must target the
high-coverage subset, not a 100%-local one.

## Interpretation

The acceptance problem that blocked Phases 03-10 is **solvable at G=2** by
affinity co-scheduling: each node holds half the experts, clustered by
co-activation, and well-matched requests draft locally at ~0.85 acceptance,
losslessly verified. This is a real positive and the strongest acceptance result
in the project.

The catch is the familiar anti-correlation, pushed out by one notch:

```text
G=2 (node holds E/2 experts): acceptance works (0.85+), but only 1 inter-node hop
    -> limited communication advantage, and large per-node memory (half the pool).
G>=4 (more nodes, more inter-node A2A to amortize): acceptance collapses (0.45-0.59).
```

So affinity placement extends the feasible acceptance regime from "nowhere" to
**G=2 (2-way grouping)**, but not into the higher-G regime where the multi-node
communication advantage is largest.

## Conclusion

- **Acceptance is now solved for 2-way (2-node) intra-group drafting** on
  no-shared-expert models via affinity placement + request scheduling -- first
  time `beta >= 0.8` is reached in the project.
- It does **not** reach G>=4. The method's viable acceptance regime is the
  2-group case, which is also the low-communication-advantage end.
- Combined with Phase 12 (intra-node EP-width barely matters) and Phase 11
  (timing is inter-node-bound), the end-to-end win still requires that a 2-node
  split expose enough inter-node all-to-all to amortize -- measurable only on a
  real 2-node IB testbed.
