# Phase 90 results — hierarchical search (DRAFT)

## E1 arc (2026-07-23): can acceptance-importance be scored cheaply?
## ANSWER: only as a weak prior — three shortcuts refuted, one
## factor isolated, and the binding constraint identified.

CORRECTION (supersedes the first E1 write-up): the initial "8B"
validation compared Qwen3-8B scores against the 83 "dense" arms,
which are **Qwen2.5-7B** (28 layers) — invalid pairing. The true
Qwen3-8B record is 86/logs/beta_profile.log (full measured LOO
singles at 16k on-policy refs) + beta_greedy.log (greedy arms; sets
{2},{2,8},{2,8,10},{2,4,8,10}...). All results below use correct
pairings. The 32B rows of the first write-up were correctly paired
and stand.

### The validation chain (all vs committed measured records)

| screen candidate | test | result |
|---|---|---|
| angular distance (1 pass) | singles vs true 8B LOO@16k | **ANTI-correlated: rho -0.768** |
| margin-Taylor (1 fwd+bwd, off-policy C4) | singles vs LOO@16k | anti: rho -0.465 |
| micro-LOO @2k raw C4 (36 fwds) | singles vs LOO@16k | rho -0.243 (n.s.) |
| **micro-LOO on-policy** (86 refs, tail positions) | singles vs LOO@16k | **rho +0.344** (p=.054) |
| committed LOO@16k ITSELF (control) | within-size added-layer vs greedy arms | **rho -0.09: even perfect singles cannot rank set growth** |
| angular / Taylor / micro within-size | true-8B greedy arms | all ~0 (-0.03..-0.24) |
| 32B angular within-size | 32B greedy arms | -0.109 (first write-up, stands) |

### The three measured findings

1. **REF DISTRIBUTION dominates the sign**: the same mechanism flips
   from -0.24 (raw C4 text) to +0.34 (the target's own generations,
   same positions the committed harness scored). Acceptance
   importance is a property of the ON-POLICY distribution; off-policy
   refs actively invert the layer ranking. (Cheapest-8 overlap with
   the committed record: 6/8 on-policy vs ~0/8 off-policy.)
2. **NON-ADDITIVITY blocks singles->sets regardless**: the control
   row is decisive — the committed harness's own LOO singles fail to
   rank its own greedy added-layer outcomes (-0.09). No singles
   screen (perfect or proxy) replaces conditional measurement for
   composition; sub-additive/super-additive interactions (e.g. the
   adjacent-pair {5,6} collapse at Q3-8B off-policy) are first-order.
3. **RESOLUTION is the cost floor**: even on-policy, subsampled
   screens (576 positions) hit +0.34 because true per-layer damage
   differences are ~1-2 accept-% — below cheap-sample noise. The
   82-percentile-grade signal needs the full-ref protocol (~2-5 min
   per config), which is exactly the committed 77-harness.

### Consequence for C2's stage 2 (the honest hierarchy)

The measured screen hierarchy: one-pass geometry scores are DEAD
(anti-correlated); on-policy micro screens are valid only as WEAK
PRIORS for ordering candidates within a greedy round; ground truth
per round is the direct conditional beta at deployment-refs. This is
precisely the committed 83/86 recipe (singles as prior -> re-measure
top-POOL candidates per round) — which these experiments now justify
as near-optimal rather than naive: every cheaper shortcut was tried
and measured to fail, and the recipe's own design note
("interactions break the product; the singles ranking is only a
prior") is confirmed quantitatively. Perplexity-class importance
(compression literature) is not just uncorrelated but INVERTED on
off-policy text — draft construction must measure acceptance, on
policy, conditionally.

### Status of the pre-registered gates

P1 (proxy rank-correlation >= 0.7): REFUTED for all one-pass scores;
   best achievable screen (on-policy micro) = 0.34.
P2 (attention-mass concentration <= 25%): REFUTED (43% heads / 50%
   layers), and mass concentrates in exactly the layers acceptance
   skips — raw far attention is largely redundant.
P1b (band structure): SUPPORTED (67% of missing mass beyond 2048,
   explaining the measured win2048 flatline).
P3/P4 (hetero-window): E3 redesign required — per-layer window
   assignment must be searched by conditional on-policy beta (LOO
   seeding at ~36 x 2-5 min), not scores. Decision pending.

### Score artifacts (committed)
q38b_/q332b_ locality+importance CSVs (E1), q38b_acceptproxy.csv
(Taylor + micro-LOO), e1c.json (pair/ctx probes: ctx-invariance
CONFIRMED 2k vs 8k; adjacency super-additivity observed),
e1d_onpolicy.json (on-policy singles), rankcorr.json.
