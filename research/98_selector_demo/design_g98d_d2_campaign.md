# G98-D design — the D2 acceptance campaign

Source phase: this one. Objective: measure what the preregistration's D2
section registers — screen honesty (a), the knapsack identity against
count-matched controls (b), and u-axis resolution of `tau(w, g, u)` (c) —
using the cost model Round 2 just validated (47/48 covered, sound
elimination exercised 5/5 correct). Written 2026-08-14, immediately after
G98-C closed; nothing here is a measurement.

## What the prereg fixes in advance

* One unconditionally-armed **KMAX = 8** stream per (composition, regime);
  greedy, fixed batch shape, pinned realization.
* Per-position reached/accepted counters, binned by generated-suffix
  position; 2,000-4,000 armed steps per (composition, regime, u-bucket).
* Replication over **content seeds 4, 5** (fresh; cost used 2, 3), not
  boots; request-level bootstrap; identical prompt sets across compositions.
* Accounting closes as `E + C = A + H`; screen by the product bound
  (admit-only), confirm only portfolio finalists; controls frozen in
  `data/prereg/w98_d2_controls.json` (seeds 98001-3).
* u-buckets provisional at [0,256), [256,1024), [1024,3072), [3072,inf);
  final boundaries are a scored OUTPUT of the first run.
* Realization bridge: screening may run eager/piecewise; one paired boot
  quantifies the eager-vs-captured acceptance bridge; survivors clear
  `tau*` by more than the bridge width.

## Asset inventory (already built and tested)

| asset | state |
| --- | --- |
| accounting identity `E + C = A + H` | `w98_accounting.py` + tests |
| knapsack + count-matched controls | `w98_knapsack.py` + tests, seeds frozen |
| cost side of `tau*` | Round-2 fits (`d1p_fits.json`), validated |
| per-position counters | upstream `SpecDecodingStats.num_{accepted,draft}_tokens_per_pos` |
| host-load + measurement gates, lane, ratchet loop | carried from G98-C unchanged |
| prompt machinery | `generate_w98_prompt_manifest.py` |

## Build status (2026-08-15)

* **Gap 2 (KMAX-8) and gap 3 (per-request rows): DONE**, committed as the
  `w98-d2` boot scope. Details in the commit; the two design points worth
  keeping here: acceptance is a PREFIX, so per-position profiles need no
  per-position counters (accepted=a means 1..a accepted, a+1 rejected); and
  the per-request suffix length is genuinely required, because G98-C traces
  show **84% of armed steps span a range** of generated-suffix lengths, so
  the batch min/max cannot be inverted into a u-bin.
* **Gap 1 (seeds 4, 5): BLOCKED on a provenance decision** (below).
* Gaps 4, 5 (runner, bridge boot) not started.

## Gaps, in build order

1. **Seeds 4, 5 prompt manifest — needs a decision.** The committed bundle
   carries seeds 2, 3 only, and the frozen generator cannot be re-run
   unmodified on h104. What was established by re-fetching the five pinned
   datasets here:
   * **Content provenance is PROVEN.** All four `datasets` fingerprints
     match the frozen values exactly (`59ec1b7f9357c7a2`,
     `e3d8dec87297b9de`, `561ad8a9f5f90b8d`, `0d657f6528371f13`), and 4 of
     5 cache files are **byte-identical** to the pins on a different box.
   * Three mechanical blockers remain: the `aime` arrow cache differs in
     bytes while matching in fingerprint (arrow writer nondeterminism);
     `refs/main` is absent because the fetch pinned an explicit revision;
     and `regime_datasets.py` — which is itself hash-pinned — hard-codes
     the c4 glob at `/data/smcho/...`, a path that cannot exist here.
   * **Recommended resolution:** a D2-specific generator that keeps the
     strong checks (same dataset revisions, same fingerprints, same
     tokenizer snapshot, canonical encoding) and records its OWN file
     hashes and loader hash for h104 — the same relocation pattern already
     applied to the model checkpoints. This changes the verification basis
     from "byte-identical cache files" to "identical content fingerprints
     plus own pins", which is a preregistration-adjacent call and so is
     left to the researcher rather than taken silently.
2. **KMAX-8 unconditional arming.** No `KMAX` path exists in
   `koff_runtime.py`. Candidate route: `num_speculative_tokens = 8` with
   the existing per-batch schedule (`[[1, 32, 8]]`) plus a forced-ON boot
   scope so the K/OFF policy cannot disarm. The preregistration's own risk
   register (w98r2 section 7.3) says the capture-path weight-equality and
   action-identity assertions will fire: the P4 event contract hard-codes
   `expected_k in {0, 4}` ("P4 event K differs from its capture action")
   and same-boot binding identity. These must be generalised under a new
   boot scope (`w98-d2`), not weakened for the existing ones.
3. **Per-request, per-suffix-position accumulation.** Upstream counters
   are engine-global per scheduler step; D2 needs them per request and
   binned by generated-suffix position for the u-axis. Either extend the
   koff trace with per-step per-request accept counts (the trace already
   carries request rows) or add a D2 sidecar to the profiler.
4. **The G98-D runner**, in the G98-C mold: hash-bound authorization
   (create-only, versioned), both gates, resumable stage boots, singles ->
   screen (CPU, product bound) -> composed confirmation, with the
   commitment barrier between screen output and confirmation boots.
5. **Bridge boot** (eager vs captured acceptance) as its own recorded
   stage.

## Sizing

Singles per regime x 2 seeds, then finalists: on G98-C boot times (~2 min
warm), a singles sweep on the Round-2 axes is ~30-40 boots per seed;
confirmation adds the finalist set. Two to three h104 windows of the size
G98-C used, all gates carried over. Acceptance is near-deterministic under
greedy at fixed realization, so boots repeat only when a gate rejects.

## Decision criteria to leave the design stage

* Amendment 2 (instrument) approved or explicitly deferred — D2 itself is
  timing-free, so it may proceed regardless; only D3 waits on it.
* The KMAX-8 route validated by one smoke boot (assertions generalised,
  counters flowing, accounting closing) — the G98-A analogue for this gate.

## Expected next artifact

`w98d2_prereg` additions (seeds 4-5 manifest hashes, KMAX contract, bridge
protocol) + the G98-D smoke gate, before any scored acceptance number.

## G98-D0 smoke gate: PASSED (2026-08-15)

`run_w98_g98d0_smoke.py`, h104 GPU 7, both quant paths, non-scored. The
engine path is proven: the boot contract admits `w98-d2`, K=8 arms, and
`acceptance_rows` land with per-request suffix lengths.

| path | armed steps | rows | action | mean accepted / 8 |
| --- | --- | --- | --- | --- |
| target-matching | 22 | 187 | `w98-d2-kmax8` | **8.0000** |
| w4a16-quantized | 31 | 210 | `w98-d2-kmax8` | 6.9667 |

**The target-matching row is the instrument's own sanity check.** A draft
whose weights are identical to the target must, under greedy verification,
accept every drafted token — and it measures exactly 8.000 with a
per-position conditional of 1.0 at all eight positions. The acceptance
accounting is therefore correct by construction, not merely plausible. The
quantized path then reads 6.97/8 (87.1%), the first acceptance signal of
this phase.

**A mechanism worth recording for the u-axis.** Suffix spread appeared in
13 of the quantized boot's armed steps and in ZERO of the target-matching
boot's: when acceptance is perfect every request advances in lockstep, and
the batch diverges only as acceptance varies between requests. So batch
inhomogeneity is not an incidental nuisance — it is produced by exactly the
quantity D2 measures, and is guaranteed to be present wherever tau is
interesting. Per-request binning is required, not defensive.

Four defects were found and fixed by this gate before any scored number,
which is what it exists for:

1. the boot contract hard-required `num_speculative_tokens=4` for every
   scope;
2. step-0 work evidence was forwarded only when the action id was literally
   K4, so an armed K=8 step arrived with `None` and tripped the armed-action
   evidence check (`gpu_model_runner`, now keyed on "armed", identical for
   K=4 scopes);
3. my own boot check read the scheduler's DENSE batch-size -> K lookup,
   whose index 0 is an unused sentinel, and would have refused every D2
   boot; the runtime guarantee moved to the trace, where it is observable;
4. unarmed decode steps are structural at generated suffix 1 — the priming
   step after prefill has no previous token to draft from — so the contract
   permits exactly that class and refuses any unarmed step past it (2
   priming steps per boot observed, 0 violations).

## D2 calibration: two registered assumptions, measured (2026-08-15)

`probe_w98_d2_calibration.py`, non-scored, quantized path (target-matching
is degenerate here — it accepts everything, so it cannot show a difference
between boots or realizations).

### Boot determinism: CONFIRMED, and exactly

| comparison | requests identical | steps mismatched | mean accepted delta |
| --- | --- | --- | --- |
| captured boot A vs boot B | **17 / 17** | **0 / 210** | **0.000000** |

Every per-position conditional matches to six decimals. The measurement
contract's assumption — replicate over content seeds, not boots, "since
acceptance is near-deterministic under greedy at fixed batch and
realization" — is not merely near-deterministic here but bit-identical. Two
consequences for the campaign:

1. Repeat boots buy nothing; the sampling budget belongs entirely to
   content seeds and requests, as registered.
2. **D2 is immune to the clamp.** Acceptance is a function of weights and
   prompts, not of time, so a contaminated box cannot bias it. The G98-C
   measurement-shape gate must NOT be carried into D2 — it judges timing
   shape and would reject perfectly good acceptance data — and the D2
   campaign can run in windows the cost campaign would have refused.

### Realization bridge: NON-TRIVIAL, and now quantified

| arm | mean accepted / 8 | armed steps |
| --- | --- | --- |
| captured (deployed) | 6.9286 | 31 |
| eager | 7.3035 | 27 |

**Bridge width: +0.375 accepted tokens, eager over captured** (~5%
relative), with 19 of 201 compared steps (9.5%) disagreeing and only 11 of
17 requests bit-identical. Per-position, the gap concentrates at the head of
the chain: +0.019 at position 1 and +0.034 at position 2, ~0 thereafter.

The mechanism is numerical, not logical: capture can change kernel
selection and reduction order, perturbing logits enough to flip an argmax
occasionally, which changes the drafted token and therefore acceptance.
Greedy decoding does not make the two realizations identical — it only
makes each one deterministic.

This is exactly why the preregistration registered a bridge and required
survivors to clear `tau*` by more than its width. That width now has a
measured value instead of a placeholder: **screening in eager
OVERESTIMATES acceptance**, so an eager screen must be treated as
optimistic and the margin applied in the conservative direction. One paired
boot per the registered protocol; re-measure if the finalist set's
compositions differ materially from the calibration cell.

## Seeds 4-5 bundle: FROZEN (2026-08-15)

The provenance call was made in favour of re-pinning on h104. Generated by
`generate_w98d2_prompt_manifest.py` — Phase 97's generator reused verbatim,
overriding only output paths, content seeds, the regime loader's c4 cache
root, and the dataset file hashes.

* **369 records, 2,222,634 prompt tokens.** Bundle
  `data/prereg/w98d2_prompt_tokens.jsonl.gz`,
  sha256 `e0d18da3ea244a7d1e934956d899dfcd9c351f0db29669119a6180490b385244`.
* **Disjointness VERIFIED, not argued:** hashing every prompt in both
  bundles gives 352 cost prompts, 337 D2 prompts, **overlap 0**.
* Content identity preserved: same pinned revisions loaded through the same
  pinned wrapper, whose fingerprint check passed for all four Arrow
  datasets. The re-pin covers file hashes only.
* The loader is a copy whose sole delta is the c4 cache root
  (`w98d2_regime_datasets.py`); the canonical loader is hash-pinned by the
  seeds 2-3 freeze and was not touched.

Two constraints surfaced by actually running it:

1. **c4 needs more than one shard.** Seeds 4-5 draw further into the packed
   stream than 2-3 did, so shards 1 and 2 were fetched at the same pinned
   revision. The loader globs in sorted order, so extra shards only APPEND:
   every document index the earlier seeds resolved to is unchanged. All
   present shards are now pinned, and the D2 verifier keeps the frozen
   check that the glob resolves to EXACTLY the pinned set.
2. **R4/seed-5 has a short pool: 17 prompts, not 32** — recorded in the
   manifest's `short_pools`. R4 packs cnn_dailymail `test[:2000]` into ~177
   long prompts and seed 5 draws at offset 160. This is forced, not chosen:
   disjointness from the cost lattice requires offset >= 128, and 128 is the
   only offset that fits, so no pair of disjoint seeds can both be full.
   Enlarging the slice would change the `load_dataset` call and the pinned
   wrapper's fingerprint check would reject it — trading a verified content
   guarantee for pool uniformity. A boot consumes `batch` prompts (R4 batch
   = 8), so 17 is sufficient; the D2 generator therefore checks SUFFICIENCY
   against batch and records every shortfall rather than absorbing it.

## Analysis chain: validated end to end (2026-08-15)

`w98d2_stream.py` bridges w98-d2 traces to the frozen G98-0 accounting rows.
The bridge is a translation only — every estimate comes from
`w98_accounting`, which already carried u-bucket binning, `tau_at_depth`,
request-level bootstrap and paired-delta comparison. The one derivation is
the clip, `clipped = accepted + 1 - committed`, which inverts `row_emitted`
so a stream that closes here closes under the FROZEN identity rather than a
private one.

Run against the G98-D0 traces, the whole path — engine to trace to StepRow
to closure to bootstrapped tau — produces:

| path | H | D | A | C | E | E+C = A+H | tau (u-bucket 0) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| target-matching | 204 | 187 | 1496 | 85 | 1615 | yes | **9.0000** [9.0000, 9.0000] |
| w4a16-quantized | 227 | 210 | 1463 | 75 | 1615 | yes | 7.9485 [7.4476, 8.3716] |

Both readings are checks, not just outputs. `tau = 9.0000` is arithmetically
forced for a self-draft at KMAX=8 (eight accepted draft tokens plus the
target's own), and its bootstrap interval correctly collapses to a point
because the outcome is deterministic. Both streams emit exactly E = 1615,
as they must: same prompts, same token budget. The quantized path then
shows a real interval, which is what every scored D2 cell will look like.

Only bucket 0 is populated here because the smoke boots generated 96
tokens; the campaign's longer streams populate the upper u-buckets, whose
boundaries are a scored OUTPUT of the first run rather than an input.

## D2(b): a specification gap, not a build gap (2026-08-15)

D2(b) claims the knapsack set beats count-matched random and worst controls
at each surviving count. The controls are frozen
(`w98_d2_controls.json`, seeds 98001-3) and `w98_knapsack.py` implements
`knapsack_select`, `top_k_by_retention`, `worst_k_by_retention` and
`random_k` — but every one of those except `random_k` consumes a **per-layer
retention vector `r`**, and nothing in the preregistration registers how `r`
is obtained. It is not measured anywhere in the phase, and no artifact
carries it.

The natural estimator — leave-one-out, boot with exactly one layer skipped
and read the acceptance loss — is **forbidden by the frozen lattice**, whose
skip counts are `{0, 4, 8, 16}`; the boot contract refuses count 1 by
construction. So D2(b) cannot be run as registered without one of:

1. admitting count 1 under `w98-d2` for a NON-SCORED retention probe (the
   scope already exists and the counts it admits are its own constant, so
   this is additive and does not touch the cost campaign's contract); or
2. registering a different estimator for `r` — e.g. leave-one-out within a
   count-4 set, or a proxy that never touches acceptance — which is a
   preregistration decision because it changes what "the knapsack set"
   means; or
3. reporting D2(b) as NOT EXERCISED, the way Round 1 reported its
   elimination rule for want of a scored surface.

Note that the CONTROLS do not need `r` to be honest: `random_k` is seeded
and independent, and the comparison itself (`bootstrap_paired_delta`) is
already built and tested. Only the knapsack's own input is missing.

This is surfaced rather than resolved: inventing an estimator for `r` after
seeing the campaign's acceptance data is exactly the move preregistration
exists to prevent. Whichever route is taken should be recorded before any
D2(b) measurement is booted.
