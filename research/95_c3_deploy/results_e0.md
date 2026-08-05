# E0 results — the window-switching envelope on real data

Two seeds x {AR, w512, w2048} x 6 regimes x 2 arches, deployed C3 stack
(FULLCG + wholechain + scratchpad + shared-KV), GPUs 0-1, 2026-08-05.
Data: `data/e0_{arch}_w{win}_s{seed}.json`, scored by `scripts/score_e0.py`.

## Verdict

| arch | cross-seed gain, accept-binding | cross-seed gain, all regimes | gate |
|---|---|---|---|
| dense | **+2.78% / +2.65%** (both directions) | +0.96% / +0.79% | **PASS** on long-context workloads |
| llama | +0.08% / +0.00% | -4.62% / +0.00% | **FAIL** |

Cross-seed = pick each regime's window on one seed, score it on the other, so
selection and evaluation land on independent draws and the winner's curse is
removed. Raw same-seed envelopes (biased up, recorded but not used for the
decision): dense +1.75% / +1.42%, llama +1.27% / +0.00%.

**Correction on what a "seed" is here (2026-08-05).** `load_regime` accepts a
`seed` argument but **never uses it** — verified by hashing the prompt sets:
byte-identical for R4/R5/R5cot/R8/R1/R6 across seeds 0 and 1. Every regime but
R8 also runs at temperature 0.0. So seed 0 and seed 1 are **replicate boots on
an identical deterministic workload**, not independent content draws (only
R8's *sampling* differs, via the SamplingParams seed at T=1.0). The cross-seed
procedure is therefore a test-retest design over BOOT noise. That is still a
valid bias control — boot noise dominates here and selection remains
independent of evaluation — but it is not the content resampling C2's 2-sample
truth scoring performed, and must not be described as such.

**P1 (dense, registered [+1%, +4%], basis +2.04%): CONFIRMED.** +2.65-2.78%
on the accept-binding group, consistent in both directions, close to the
compile-cell prediction.

**P2 (llama, registered [+1%, +4%], basis +2.28%): REFUTED.** +0.00-0.08%.
The map predicted llama would be the *cleanest* test of the window axis (its
K-only arm adds nothing, so all its value had to come from the window). On
real serving it delivers nothing, for a diagnosable reason (below).

**The headline is workload-dependent, and both numbers are real.** For a
long-context-dominated workload the best fixed window is w2048 and switching
is worth +2.7%. For the mixed 6-regime workload w512 is already the right
default for 5 of 6 regimes, and switching adds only +0.8-1.0%. Quoting the
+2.7% without the workload qualifier would overstate it.

## Why llama fails: run-to-run variance swamps the effect

Between-boot swings **on an identical deterministic workload** (same prompts,
T=0.0), with the AR anchor stable to 0.2% (142.7 vs 142.4 tok/s):

| cell | S(seed 0) | S(seed 1) | swing | tau s0 -> s1 |
|---|---|---|---|---|
| llama R5 w2048 | 1.3952 | 0.8153 | **-41.6%** | 3.162 -> 2.317 |
| llama R5 w512 | 1.3406 | 0.9972 | **-25.6%** | 2.957 -> 3.151 |
| llama R5cot w2048 | 1.4950 | 1.3732 | -8.2% | 3.172 -> 2.875 |
| dense (worst of 8) | — | — | -4.0% | — |

**Cause NOT established — an earlier draft of this section over-claimed it.**
I argued the tau shifts prove different depth choices, via `tau <= K+1`. They
do not: in each pair only ONE value exceeds 3.0 and forces `K >= 4`, while the
other is consistent with either K2 or K4, so a constant K fits every pair.

What IS established: R5's prompts are identical across the two boots and its
sampling is greedy, so these swings are **pure system-level run-to-run
variance** — not content, and not demonstrably a policy choice. Candidate
causes not yet separated: kernel autotune nondeterminism, KV-cache sizing,
CUDA-graph capture differences.

So llama's ~0% switching gain is not "no window preference exists"; it is
"run-to-run variance at long context (-25% to -42% on identical inputs) is an
order of magnitude larger than the window effect it is supposed to exploit."
Dense is stable over the same comparison (worst swing -4.0%), which is why its
+2.7% resolves. **A 25-42% swing on a deterministic workload is a
reproducibility defect in its own right, and is logged as an open item.**

## The R8 gate failure, RESOLVED by measurement (supersedes the section below)

The diagnostic (`scripts/diag_r8_gate.sh`, 3960 logged decisions) refuted the
hypothesis this section advanced, and two of my inferences with it.

**The policy is not misbehaving.** Armed duty **4.6%** (K0 95.4%), live f
median **0.674** -- correctly below the cell break-even R=0.733. Band changes:
**9 in 3960 steps (0.2%)**, so the band-thrash hypothesis (H1) is REFUTED. The
f=1.000 spikes are new `generate()` calls arriving with no per-request
history; they decay below break-even in ~10 steps.

**The "armed ~94%" inference was wrong**, because it assumed OFF == AR. Direct
measurement with probes disabled (`VLLM_SELF_SPEC_ACCEPT_PROBE_BURST=0`):

| llama seed 0 | S vs AR | note |
|---|---|---|
| parked R8 | **0.9752** | accept 3.385 -> ~1% residual arming from new-request optimism |
| parked R4 | **1.0005** | accept None -- never armed |
| full policy R8 | 0.9350 | E0 |
| + GATE_DEBUG | 0.9058 | logging tax ~3% |

Decomposition of llama R8's 6.5% deficit:

    parked-engine overhead        2.5%
    probe / arming churn          4.0%   (0.975 parked -> 0.935 with probes)
    wrong steady-state decisions  ~0%    (correctly disarms 95.4% of steps)

**Finding 1 — "disarming is FREE" is not quite true.** The scheduler comments
"disarming is FREE (E0), so OFF's score is always 1.0". Phase 82's E0 measured
the TOGGLE as free (switch latency); it did not measure the parked engine's
throughput. Measured here: parked costs **2.5% at R8 (b16), ~0% at R4 (b8)**.
Small, but it means OFF's true score is cell-dependent and slightly below 1.0,
so the argmax scores every arm against a reference the engine cannot quite
deliver.

**Finding 2 — arm/disarm TRANSITIONS are not priced at all, and dominate.**
4.6% armed duty costs 4.0% throughput; the steady-state model predicts ~0.5%
(duty x per-step deficit). Transitions therefore cost several times their
occupancy. `S_K = (1+fK)/(KR+1)` prices only steady-state operation; the 2%
arming rent is a hysteresis constant, not a measured transition cost.

**This is not an argument for deleting probes.** At R4 the same probes HELP
(+2.1%: 1.0005 parked -> 1.0218 with probes) because arming wins there, and
phase 91 proved probes are load-bearing for drift detection (removing them
starved the refresh detector -- the exploration-for-detection finding). The
tension is real and now quantified: probes buy detection everywhere and cost
4% exactly where the config loses.

## (superseded) The f-vs-R margin analysis

Setting `S = (1 + fK)/(KR + 1) = 1` gives `fK = KR`, so **the policy arms iff
live f > the cell's R, independent of K**. Margins on llama are thin:

| | live f | cell R | margin |
|---|---|---|---|
| llama R8 (b8/c2000, w512) | 0.648 | 0.733 | **-0.085 (should disarm)** |
| dense R8 (same cell class) | 0.926-0.982 | ~0.73 | +0.20 (correctly arms, S=1.052) |

Measured llama R8 = 0.935 vs the always-armed prediction 0.931 -> armed
**~94%** of steps (w512) and ~56% (w2048), when it should have been ~0%.
Ruled out arithmetically: probe duty (3.1% -> S 0.998, cannot produce 0.935)
and the b8/b32 nearest-cell tie (S 0.931 vs 0.889, both disarm).

Mechanism: the pooled accept estimate is a mean over running requests in which
**unseen requests default to optimistic 1.0** (`scheduler.py:1324-1326`), and
a batch-band change **clears every per-request EMA** (`:1306-1308`, band =
`n_run.bit_length()`). With 16 requests the arming flip needs only

    (u*1.0 + (16-u)*0.648)/16 > 0.733   ->   u >= 4

— four of sixteen requests at the optimistic default is enough to arm a
configuration that loses 6.5%. R8's b16 sits *exactly* on a band boundary
(15 -> bit_length 4, 16 -> 5), so any fluctuation in the running count wipes
the accumulated evidence, and every genuinely new request enters at 1.0.

This does not overturn phase 91's optimistic-init doctrine (map-prior init
measured 0.4-1.2% worse because it starved genuine content winners). It
locates that doctrine's failure mode: **optimistic init is safe when the f-vs-R
margin is wide and unsafe when it is thin** — which is exactly the low-
acceptance regimes where a wrong call costs the most. Band-keyed resets on
power-of-2 batch sizes are a common deployment shape.

## Two constraints found by measurement

**Full-context drafting is not realizable in the deployed stack.**
`VLLM_SELF_SPEC_DRAFT_FULLCG` raises "requires
VLLM_SELF_SPEC_DRAFT_KV_WINDOW > 0 (the scratchpad materialises the
sinks+window key set)". Phase 94 met the same wall and ran `w-none` PLAIN,
disclosing the realization split at -0.5% mean; running it plain here would
compare across stacks inside the metric under test, so the switching set is
`{512, 2048}`.

**P8 (null control) refuted — the window is a COST floor, not only an accept
tradeoff.** At R6 (end ctx 318, where neither window binds semantically)
dense runs +5.0% faster on w512 with acceptance IDENTICAL (4.733 vs 4.734),
llama +12.0%. `_scratchpad_n_kept_blocks` gathers
`n_sink + ceil((window+K)/block_size)` blocks *every draft step regardless of
live context length*, so w2048 pays for KV it never reads. Consequences: no
window-inert regime exists, so no valid null control is available (bias
control fell entirely to cross-seed selection); regimes are regrouped by
mechanism. The finding strengthens the switching case in principle — at short
context a narrow window is strictly better, same accept and less cost — so the
window should track context length for cost reasons alone.

## Recommendation

**Do not build E1 (window multi-capture) yet.** The gate passes on dense only,
only for long-context-dominated workloads, and is worth +2.7% there / +0.9%
mixed. The same measurement surfaced a gate-safety defect costing **6.5% on a
single regime** and **8-42% of boot-to-boot decision noise on llama**. Both
trace to one cause: the live-f estimate is unreliable near the `f = R`
break-even. Fixing that is worth more than the switching gain it would also
unlock, and the switcher built on top of an unstable estimator would inherit
the instability.

Proposed order: **E1' (estimator repair) before E1 (multi-capture)** —
- seed new requests from the cell's `f_ref` instead of 1.0 when the cell has a
  profile (the map prior is available and phase 91 rejected it only as a
  *global* default, not as a per-cell seed for unseen requests);
- do not clear accumulated evidence on band change — decay it, or key the band
  on a hysteretic batch bucket instead of `bit_length`;
- require the arming margin to exceed the estimator's own standard error
  rather than a fixed 2%.

Each is testable against the same E0 harness, with llama R5/R8 as the
regression cells and dense as the no-regression control.
