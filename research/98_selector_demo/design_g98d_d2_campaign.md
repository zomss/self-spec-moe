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
