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

## Related work survey (2026-07-23): the field converged on the same
## three properties our E1 arc measured

| method | selection mechanism | proxy class |
|---|---|---|
| Draft&Verify (orig. Self-SD) | Bayesian opt over layer sets, objective = MEASURED time/verified-token on calib data | direct measurement |
| SWIFT (ICLR'25) | on-the-fly random search + interval BayesOpt DURING serving; LLM-generated tokens as ground truth | direct, ON-POLICY, in-the-loop |
| CLaSp (ACL'25) | per-verify-step DP over layers, objective = hidden-state alignment vs LAST VERIFICATION's states | similarity proxy, but ON-POLICY + context-local, refreshed every verify |
| KNN-SSD | precomputed per-DOMAIN layer-set library + KNN input matching | pool + selector (== our per-regime pools) |
| KnapSpec (our h2h baseline) | knapsack: value = cosine sim per layer (ADDITIVE), weight = latency; corrected by r=5 recent-step empirical accept | offline-class additive surrogate + on-policy accept patch |
| SpecDec++ | trained acceptance-prediction head (per-token) for adaptive draft length | LEARNED proxy from measured acceptance |
| BanditSpec / Not-a-Bandit | online drafter selection, regret guarantees | bandit (external-drafter pools) |
| HAWQ line (quant) | Hessian-trace per-layer sensitivity | offline second-order score (task loss) |

Reading:
- NOBODY ships a working OFFLINE geometry score for acceptance. The
  successful selectors are (a) on-policy, (b) in-the-serving-loop /
  context-local, (c) measurement-anchored or learned from measured
  acceptance — the exact three properties E1 isolated (sign-flip
  on-policy; non-additivity; resolution). Our negative result matches
  the field's evolution away from offline scoring.
- KnapSpec's additive cosine knapsack is PRECISELY the surrogate
  class our control refutes (singles don't compose, rho -0.09) — a
  mechanistic explanation for why our measured-search configs beat
  their number (~1.57 vs 1.43 with their own lever included).
- Three upgrade candidates for our stage 2, in order of promise:
  E1e: CLaSp-style ON-POLICY context-local similarity DP — the
       similarity proxy re-based on live verification states; test
       against our committed greedy record before adoption.
  E1f: learned config->beta predictor (SpecDec++ class, per-config):
       train on our 100+ committed (config, beta) pairs; learning
       absorbs non-additivity that analytic scores assume away.
  Stage-3 theory: BanditSpec/Not-a-Bandit regret results are the
       citable foundation for our switch-cost-aware bandit.

## E1e (2026-07-23): CLaSp-class objective — REFUTED as a screen at
## the same grade as everything else; the ~0.35 ceiling looks
## intrinsic

Continuous on-policy hidden-state cosine (the literature's surviving
proxy class), 69 configs (36 singles + the 33 committed arms) on the
86 on-policy refs:

| test | result |
|---|---|
| singles vs committed LOO beta | rho 0.368 (p=.04) — same grade as binary micro-LOO (0.344) |
| arms raw | 0.941 — size confound (as always) |
| within-round, per size | +0.667 / +0.214 / -0.714 / -0.500 / -0.200 — no reliable ranking |
| pooled within-size | 0.924 — **INVALID: pooling whole-set (cos, beta) pairs across rounds reintroduces the size confound**; flagged so it is never cited |
| greedy-set overlap | 0-1 of k (like all other screens) |

Readings:
- The continuous-observable hypothesis FAILED: cos (continuous) and
  argmax agreement (binary) converge to the same ~0.34-0.37 singles
  correlation on identical refs/positions. The ceiling is not an
  observable-choice artifact — it is the intrinsic fidelity of
  subsampled on-policy signals at the 1-2 accept-% resolution the
  ranking requires. FOUR proxy classes now measured against the same
  committed record; all prior-grade, none screen-grade.
- Deeper sets show the cos objective DEGRADING (negative rho at
  sizes 4-6): as drafts diverge, hidden-state similarity stops
  tracking argmax agreement — a second mechanism (beyond
  non-additivity) capping similarity-based selection at depth. This
  is also a caution for CLaSp/KnapSpec-style objectives at
  aggressive skip budgets.
- SCOPE NOTE (fair to CLaSp): its deployment value is context-LOCAL
  re-optimization per verify window — a runtime adaptation mechanism,
  not an offline search screen. Our test evaluates the offline-screen
  role only. Per-context set re-optimization remains a legitimate
  stage-3 lever candidate (with citation), orthogonal to this verdict.
- Remaining untested route: E1f (learned config->beta predictor on
  the committed record). Everything analytic is now closed.

## E1f (2026-07-23): learned config->beta predictor — REFUTED; the
## proxy program is now CLOSED with a five-way falsification

Ridge + RBF-kernel-ridge on structural features (size, adjacency,
runs, position thirds) + per-layer score aggregates; leakage-strict
leave-one-ROUND-out eval; trained on all committed pairs (8B n=65,
32B n=93).

| model | within-round mean rho |
|---|---|
| 8B full-features (ridge / krr) | 0.081 / -0.146 |
| 8B additive product-of-singles baseline | 0.125 |
| 32B shared-features (ridge / krr) | 0.215 / -0.102 |
| 8B->32B TRANSFER (krr, clean split) | 0.429 (mixed signs; prior-grade curiosity) |

Root cause: learning non-additive interaction structure needs
training data of the same kind the predictor is meant to replace —
at 65-93 measured configs the model cannot beat the additive
baseline it was built to transcend. The one positive note (cross-
model transfer 0.43 on shared features) suggests weak universal
structure (position/adjacency effects partially transfer across
scale) — usable as a cold-start PRIOR for a new column's first
greedy round, nothing more.

### FINAL VERDICT — the proxy graveyard (all vs the same committed record)

| class | representative | best within-round fidelity |
|---|---|---|
| offline geometry | angular distance, raw attn mass | INVERTED (-0.77 singles) |
| gradient/first-order | margin-Taylor | inverted off-policy (-0.47) |
| subsampled objective (binary) | on-policy micro-LOO | 0.34 singles / ~0 rounds |
| subsampled objective (continuous) | CLaSp-class cos | 0.37 singles / ~0 rounds |
| learned predictor | ridge/krr on committed pairs | 0.08-0.22 CV; 0.43 transfer |

Acceptance-importance CANNOT be scored at screen grade by any tested
method: it must be measured — on-policy, conditionally, at full-ref
resolution. Proxies serve as round-ordering priors (0.3-0.4 grade)
and cross-model cold-start only. This is C2's central empirical
theorem, exhaustively supported, and it mechanistically explains the
h2h margin over surrogate-driven selection (KnapSpec's additive
cosine knapsack sits in the refuted class).
