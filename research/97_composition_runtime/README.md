# Phase 97 — workload-aware composition runtime

Status: **active shared-KV-only design; the matched B0 value screen ran to
completion under V13 (432/432 captures, 3/3 blocks) but the frozen scorer
refused at the 2% cross-boot certification in 3 of 36 cells. The lane decision
is not the cause: median per-block offsets are +0.099%/+0.208%/-0.133%, the
solo block is not faster, and the two blocks that disagree most sit on the same
GPU. The cause is episode noise concentrated in block 1 (6/144 rounds below the
95% floor, versus 1 and 0). The current authority is the V14 block-1 restart,
which reuses blocks 2 and 3 unmodified and binds its result before it exists;
see `results_p4_b0_block_restart_v14.md`. The screen remains unscored. The
two-lane V13 package that produced the complete capture set is documented in
`results_p4_b0_run_authorization_v13.md`; its output
`run_b0_value_screen_v12` is preserved, complete, and reusable. V12 established the two-lane design
(`results_p4_b0_run_authorization_v12.md`) and was consumed by one pre-GPU
refusal: a relative `--output-dir` was not normalized inside the source
snapshot, so it emitted zero captures and loaded no model. V13 is V12 plus
that path repair and its regression tests. The history below is retained. V5
was executed once on GPU 4 and failed closed at
the first live OFF event with zero complete captures and no score. A separate
source-bound `114688 / 0.96` ingress diagnosis now identifies the exact
exclusion as `prefill_or_mixed_batch`: the first request entered width-one
decode while the other seven requests were still in full prefill. There was
no preemption, recomputation, or invalid-spec-token exclusion. V5 and the
diagnosis authorization are consumed. A separately authorized in-process
atomic-ingress proof then queued all eight requests before execution and
passed the exact two-step contract: one complete eight-request prefill followed
by eight width-one OFF decodes with no exclusion, preemption, or recomputation.
Its V1 package failed before weight load because an empty integer environment
setting could not be parsed; the source-bound V2 repair passed. Both proof
authorizations are consumed. The source-bound V6 review then authorized one
GPU-4, nine-boot/432-capture screen. Its single attempt initialized the first
boot and queued all eight requests atomically, but failed closed because vLLM
appended an eight-hex internal suffix to every request id while the scored
recorder required the unsuffixed frozen ids. V6 is consumed; its
`run_b0_value_screen_v5` output has zero complete captures and no score and is
immutable. A shared fail-closed canonicalizer now accepts only exact frozen
ids or the vLLM-generated `-<8 lowercase hex>` form, rejects unknown,
malformed, reordered, and colliding identities, and emits frozen ids to the
recorder. The atomic proof analyzer uses the same helper. The source-bound V7
review then registered the consumed `run_b0_value_screen_v6` attempt described
below.
Separately, the source-bound real chunked-prefill diagnosis at budget 8192 and
utilization 0.90 passed: eight natural mixed steps forced q=1/OFF with no draft
dispatch, followed by 15 pure decode steps. Both diagnoses retained one
target-owned 36-layer KV cache and neither repairs nor scores the screen. The
design freezes a scored, equal-weight
W3 research objective, exact OFF/K4/w512 controls, same-event target-step
accounting, and the robust decision rule.
Those weights are not
production-workload evidence.
The existing W14/D bundle is invalid for scored value because 313/1,680 rounds
have `D>H`; Prometheus interval deltas are diagnostic-only. The w512 action
remains acceptance-only with no cost credit, and a dominance check stops P4 if
it cannot improve acceptance over K4. The exact 384-prompt token bundle is
frozen at Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218`. The strict CPU same-event adapter
and paired-block scorer are frozen. Legacy P3 aggregates fail the adapter and
remain diagnostic-only. The synchronous live K4/OFF boundary emits explicit
per-request captures; the boot-static w512 surrogate passes exact page-mask,
shared-KV, and true-slot equivalence; and the pre-run B0 bound specified a
21,682-block shared-KV floor against 21,000 required. V9 later proved that
this floor did not cover full-prefill transient activations. Capture-runner
conformance now clears all six registered implementation checks, including
trusted boot-static w512 labeling, 48-cell same-boot rotation, stable logical
weight identity with within-boot pointer proof, and launch/capture resource
guards. The runner closes to nine boots and 432 cells and has a CPU-only
preparation path. The prior authorization remains a historical HOLD and is
rejected against the changed source hashes. The authorized V2 attempt reached
GPU 4 but stopped during the first engine's sampler profile because the
installed Ninja executable was absent from the child `PATH`; it emitted zero
captures and no score. That output is immutable and cannot be resumed. The V3
review bound the failure, repaired and preflighted virtualenv executable
discovery, and authorized one fresh GPU-4 attempt. That V3 attempt passed the
Ninja preflight but stopped before capture because FlashInfer's cached sampler
requires `libcudart.so.13` and the child loader could not resolve it. V3 and its
zero-capture V2 output are now consumed. V4 forces and preflights the
PyTorch-native sampler before output creation, preserves the existing Ninja
preflight, and binds the retry to the create-only
`run_b0_value_screen_v3` path. That attempt cleared sampler, shared-KV,
capacity, compilation, and graph initialization, then stopped in the first OFF
cell when the 8,160-token effective scheduler budget produced a mixed
prefill/decode event. The strict pure-decode recorder rejected it. V4 is
consumed with zero complete captures and no score; its output is immutable.
The measurement-only repair keeps chunked prefill enabled and expands the
scheduler budget to 114,688 tokens, covering the 112,908-token maximum frozen
microbatch plus speculative reserve. The first one-initialization GPU-4 probe
at 0.90 utilization executed zero requests but measured only 19,928 shared-KV
blocks. A separate source-bound repair probe changed only measurement GPU
memory utilization to 0.96 and passed with 22,113 blocks. Both probe
authorizations are consumed. V5 bound the passing result, source closure, GPU
4, nine boots, 432 cells, and the create-only `run_b0_value_screen_v4` path.
Its one attempt initialized with 22,090 live shared-KV blocks but rejected the
first event as score-ineligible. V5 and its zero-capture output are consumed.
The later non-scored full-prefill diagnosis flushed a passive trace before the
same recorder rejection and proved the event contained one decode plus seven
prefills. Its only exclusion was `prefill_or_mixed_batch`; the oversized token
budget did not create atomic request admission. That diagnosis authorization
is also consumed. The separate one-boot serving package validated the
preserved `8192 / 0.90` boundary with an append-only non-scored trace; it
observed eight expected mixed transitions and 15 later pure-decode steps. The
subsequent in-process proof preserved the exact `114688 / 0.96` workload,
target-owned 36-layer KV cache, and 291 target/draft weight aliases. It observed
all eight prefills in step 0 and all eight q=1/OFF decodes in step 1. This
clears only the atomic-ingress gate. The later V6 attempt preserved that
queue-all behavior but exposed a distinct request-id normalization mismatch
between the proof and scored recorder. The tested repair preserves internal
request-id randomization and canonicalizes only at the frozen evidence
boundary. V7 then reached the first capture but exposed the one-token
prefill-sample versus decode-only work offset. V8 repaired and tested that
offset, but its launch failed closed before GPU preflight because the final
parent/child pathname dispatchers still listed only V6/V7. V9 bound the shared
dispatcher repair and was executed exactly once. It completed eight R4 OFF
captures, then failed on CUDA OOM in the first 112,304-token R5 full-prefill.
The actual CUDA graph pool was 4.90 GiB versus a 0.50 GiB estimate, and the
compiled MLP could not allocate a 2.57 GiB activation with 1.45 GiB free. V9
is consumed; its output is immutable and unscored. The checked transient bound
rejects another `114688 / 0.96` full-microbatch authorization: even an
optimistic graph correction leaves only 20,092 KV blocks, below the 21,000
hard requirement. The bounded chunked-prefill capture-cohort barrier now
passes its CPU-only state-machine proof at budget 8192: early prefills are held
without releasing target KV, all members enter the first measured pure-decode
event together, and one unmeasured plus 512 measured tokens close exactly for
OFF, K4, and w512. The mechanism is wired into the synchronous live scheduler
behind an exact measurement-only marker, and execution-path CPU regressions
pass. The separate V1 non-scored GPU-4 probe authorization was invoked once,
but the parent failed after preflight and output-directory creation because it
did not normalize the registered relative output path before calling
`relative_to(REPO_ROOT)`. No child or GPU model ran, and no probe result or
score exists. V1 and its output path are consumed. A tested path repair and
fresh V2 authorization passed CPU review. Its later one-shot GPU-4 execution
completed engine boot and the resource precheck, then rejected the first
cohort request because the live guard compared the configured 8,192-token
bound with the expected reserve-adjusted 8,160-token scheduler budget. V2 and
its output are consumed; no cohort, probe result, or score exists. The repair
now checks configured 8,192, effective 8,160, and 32 sequence slots
independently, and its real draft-model normalization regressions pass. A
fresh source-bound V3 package authorized one non-scored GPU-4 probe. Its exact
one-shot execution passed boot, resource, and repaired budget gates, then
failed closed on the first pure-prefill scheduler event because the runner
applied K4's width-one decode invariant to a prefill-side draft query width of
4,081. V3 and its output are consumed; no cohort completed and no probe result
or score exists. The phase-aware repair retains width one for pure decode,
permits only positive-width K4 on explicitly armed pure prefill, and passes
paired fail-closed and live-cohort CPU regressions. The source-bound V4 probe
was then executed exactly once. All three R4/R5/R5cot barriers completed, but
the probe-only final validator compared valid randomized internal request IDs
directly with frozen IDs instead of canonicalizing them. V4 and its output are
consumed; no `probe_result.json` or score exists. The shared canonicalization
repair and direct fail-closed regressions now pass. The separate source-bound
V5 package was then executed exactly once on GPU 4. Its target-matching K4
barriers for R4/R5/R5cot completed, the final canonical-ID check passed, and
the formal non-scored probe result was emitted with zero quality violations.
V5 is consumed. This is K4-only GPU evidence; OFF, w512, and all-action
rollover coverage remain CPU-proven. The separate source-bound V10 package was
executed once on GPU 4 at `8192 / 8160 / 0.90`. It completed all 48 OFF
captures and 24 K4 captures through R5cot, then failed closed on the first
K4/R8 pure-prefill dispatch. Its 72 complete captures and empty R8 placeholder
are hash-bound, immutable, unscored, and ineligible for reuse. A fresh
GPU-0/GPU-1 diagnosis reproduced the same failure both for isolated R8 and
after the exact R5cot tail. The proposal ran and returned `[16, 4]`, but R8's
2,116 variable-width draft step-0 tokens have no uniform scalar query width.
The runtime evidence check incorrectly treats that nullable uniform-width fact
as required positive-prefill evidence. The observed R5cot finished IDs are not
causal. Both diagnosis packages are consumed; the first stopped in its
observer and the corrected second closed the diagnosis. The runtime now keeps
uniform step-0 width exact and nullable while recording explicit step-0 token
count and batch size. Its CPU regressions pass. A fresh source-bound GPU-0/GPU-1
validation then completed both the full isolated-R8 cohort and the exact
R5cot-tail-to-R8 sequence without runtime or shutdown exceptions. Its parent
aggregate rejected only because it expected two armed-prefill observations;
chunked R5cot validly emitted eight before the one exact R8 observation. The
immutable external audit closes both GPU cases and classifies the parent error
as observer cardinality, but the package remains consumed and unscored. No V11
or value-screen retry was authorized by that package. A separate source-bound
V11 review now binds the consumed V10 boundary, the immutable case-level audit,
the repaired runtime sources, and the fresh create-only
`run_b0_value_screen_v10` output. It authorizes exactly one GPU-4,
nine-boot/432-capture value-screen attempt and remains unexecuted. V10 captures
and repair-validation outputs remain ineligible for reuse.
The `114688 / 0.96` settings remain rejected and are not a serving default.
P4a, action admission, and performance claims remain unauthorized. Exact
post-capture resource evidence remains a later admission gate. The executable
live registry remains P3's K4/OFF pair.
Skip, co-resident B1, and integrated composition remain gated.**

Source: Phase 96's uncertainty-aware selector (`w14_plan.md`), the Phase 82
runtime-switching measurements, the Phase 93/94 composition work, and the
2026-08-08 design discussion. Phase 96 remains responsible for validating the
selector's cost/acceptance split. Phase 97 is the engineering phase that makes
a selected composition executable under real HBM and graph constraints.

This README is the active design of record. Where it conflicts with
`design_review.md`, `results_boot_proxy.md`, or
`results_warmup_diagnostics.md`, this scope amendment wins. Those documents
remain unchanged historical evidence for the now-closed private-KV branch.

## Objective

Build and validate a hierarchical deployment system in which:

1. the user-provided target realization is fixed, while an RL trainer may
   externally advance its weight version at synchronization boundaries;
2. the target and every self-spec draft action use the same target-owned KV
   tensors, regardless of the draft-weight realization;
3. a workload-aware planner admits either the target-matching draft path alone
   or that path plus one resident quantized draft-weight realization;
4. the admitted boot class exposes a small runtime pool over draft-weight
   path, window, layer-skip set, proposal length `K`, and `OFF`; and
5. exact composition measurements and passive serving telemetry select only
   actions that preserve their registered mechanism and do not regress against
   the best proper subset or `OFF` beyond the declared noise tolerance.

The intended result is not universal hot switching and not a promise that
isolated lever gains add. It is a resource-feasible outer boot decision with
safe online recourse among a small, prevalidated action graph.

## Scope decision

### In scope

- target-matching draft weights;
- at most one optional quantized draft-weight realization per boot class;
- one target-owned shared KV cache for target verification and every draft
  action;
- preselected window and layer-skip actions;
- proposal length `K` and `OFF`;
- ordinary-inference initialization and RL actor-to-rollout weight refresh;
- matched singleton, pair, and triple composition validation; and
- workload-conditioned HBM, graph, KV-capacity, and service-value gates.

### Out of scope

- draft-only KV quantization or a draft-only KV dtype;
- a private native or quantized draft-KV pool;
- a quantized mirror of the target KV cache;
- CPU KV quantization, transfer, readiness, lag, or backfill protocols;
- target quantization as a selector lever;
- more than one optional quantized draft-weight format per boot; and
- allocating a new weight format or capturing a new graph on the serving
  critical path.

The historical four-class boot matrix answered useful diagnostic questions,
but only its shared-KV rows remain active evidence. No further private-KV or
KV-mirror profiling is part of Phase 97.

## Relationship to Phase 96

- Phase 96/D0 and D remain unchanged. Runtime engineering must not alter their
  registered actions, graph strata, scorer, or held-out data.
- Phase 96/E supplies evidence about complete draft-weight, window, skip, and
  `K` configurations.
- Phase 96/F supplies the value and resource gate for the resident pool.
- The checked P4 F audit is `not_approved`, not a negative-value result. The
  workload objective/weights, exact action value, valid D/E results, portfolio
  result, exact resource evidence, and transition overhead are unresolved.
- Phase 97 now freezes a research-only equal-weight W3 objective for the
  matched B0 screen. This resolves the screen's objective ambiguity but does
  not create measured production workload evidence or authorize a run.
- The new same-event contract declares the existing W14/D bundle invalid for
  scored value. It does not modify, reinterpret, or rescore Phase 96 data.
- Phase 97 may develop schemas, CPU resource tooling, synthetic tests, and
  correctness checks while Phase 96 validation is incomplete. It must not
  claim a useful runtime pool or launch an integrated scored evaluation before
  the relevant evidence is frozen.
- Any post-engineering action has a new realization and must be remeasured. A
  boot-alone cost measurement cannot silently price a co-resident or
  multi-capture runtime.

## Agreed direction

1. **Target quantization is fixed input.** Target architecture, checkpoint
   lineage, quantization, KV dtype, parallel layout, hardware, and serving
   stack are supplied by the user. The selector does not change them.
2. **Shared target KV is an invariant, not a lever.** Every eligible
   self-spec action reads and writes the same target-owned KV tensors.
   `kv_path` is therefore fixed to `shared_target` in the boot manifest and
   absent from the runtime action space.
3. **Draft-weight capability is a boot decision.** A boot may admit the
   target-matching path alone or that path plus one optional quantized draft
   realization. Enabling the optional path does not force its use.
4. **Workload capacity gates the optional weights.** A boot containing the
   quantized draft is feasible only if its persistent HBM, graph memory, and
   refresh peak leave enough shared target-KV capacity for the declared
   workload envelope.
5. **Window and layer skip are runtime candidates only inside a preselected
   pool.** The runtime never invents skip sets or captures graphs on the
   serving critical path.
6. **`K` and `OFF` remain runtime actions.** `K` is priced separately because
   target verification, acceptance depth, and graph dispatch change with
   proposal width.
7. **One action applies to a whole draft dispatch initially.** Per-request
   actions would partition the microbatch and create a different execution
   state and cost model.
8. **Every proper subset remains available when resources allow.** A full
   quantization × window × skip composition is never forced merely because all
   capabilities were enabled at boot.
9. **Boot changes use routing or a draining restart.** Bypassing a resident
   quantized path does not reclaim its weights, graphs, or refresh workspace.

## System objects

Use separate names for fixed inputs, boot decisions, runtime actions, and
mutable state:

```text
e = fixed environment
    (target realization, target quant, target KV dtype, TP/PP, GPU, software)

w = declared workload envelope
    (serving or rollout, concurrency, prompt/output distributions,
     live-KV demand, actor-sync cadence, SLOs)

h = boot class
    (shared-target-KV invariant, admitted draft-weight paths and storage,
     refresh workspace, W_max, K_max, resident action/graph pool)

a in P_h union {OFF} = runtime action
    (draft-weight path, window, skip set, K, graph id)

r_t = mutable runtime state
    (target/draft weight versions, current action, counters, profiler state)
```

The Phase 96 composition field named `quant` means
`draft_weight_quant`; it never authorizes a target-quantization or KV-cache
change.

The runtime feasible set is a graph, not a Cartesian product. Each action has
a stable id, required resident objects, legal transitions, graph descriptor,
and measured transition cost. Startup validation materializes only the
registered combinations.

## Shared target-KV contract

### Ownership and writes

- The target owns the cache allocation and all committed historical KV.
- Every active draft attention layer aliases its target twin's KV tensors.
- During a draft chain, the draft may write provisional KV for newly proposed
  slots into those same tensors.
- The following target verification overwrites the relevant slots with
  target-exact KV.
- Request allocation, append, rollback, and teardown use the normal target KV
  lifecycle. There is no second pool and no cache-path transition.

A quantized draft therefore uses quantized draft weights for its proposal
computation while consuming target-produced committed history. This is a valid
approximate proposer as long as architecture, layer mapping, rotary layout,
head layout, cache dtype, and KV cache specification match. Acceptance belongs
to that exact hybrid realization and must be measured; it is never borrowed
from a private-cache realization.

### Fail-closed boot validation

Every Phase 97 boot manifest must fail when any of the following is true:

- shared KV is disabled;
- a draft-only KV dtype is requested;
- any active draft attention layer lacks a target twin;
- a draft and target twin have different KV cache specifications;
- their bound KV tensors are not identical storage;
- a skipped/windowed action changes cache ownership or true slot mapping; or
- the KV allocator reports a draft-private pool.

The repository-wide environment default may remain opt-in for unrelated
experiments, but Phase 97 manifests must request and validate shared KV
explicitly.

### Layer-skip switching

The target fills committed history at every target layer. A draft layer that
was skipped in one action can therefore be re-enabled at the next target-step
boundary without historical backfill.

The current implementation is still boot-static: it replaces skipped modules
with passthrough modules and deletes their forward-context registrations.
Runtime switching must instead:

1. retain the union of all potentially active draft decoder layers;
2. retain and validate every draft-layer-to-target-layer KV alias;
3. capture one executable graph per admitted skip set, or demonstrate an
   equivalent true-work-saving dispatch; and
4. switch graph/action ids only at a target-step boundary.

### Window switching

Windowing changes the draft's read view, not cache ownership:

- **masked down-selection** changes attention masking or visible sequence
  lengths inside a maximum-window realization. It can change acceptance but
  must not be credited with lower KV gather cost unless measured;
- **cost-true switching** dispatches a separately captured per-window graph
  whose KV read/gather work actually scales with the selected window.

The action registry must distinguish these grades. A separately booted
`w512` cost cannot price masked `w512` inside a `w2048` graph. `w-off` is a
different paged-attention realization, not an infinite member of a FULLCG
window pool.

### Switch boundary

Draft-weight, window, skip, and `K` changes occur only between complete target
steps, never inside a `K`-token draft chain. Cache ownership never changes.

## Boot classes and lifecycle

The active boot family contains two capability classes:

| class | resident draft-weight paths | KV path | runtime recourse |
| --- | --- | --- | --- |
| `B0` | target-matching only | shared target KV | window, skip, `K`, `OFF` |
| `B1` | target-matching plus one quantized draft | shared target KV | choose either weight path, window, skip, `K`, `OFF` |

`B1` is the desired co-resident capability, not yet an implemented fact. The
existing W4A8-plus-shared-KV proxy proves that the quantized path can boot and
use shared KV; it does not prove that target-matching and quantized draft
modules and graphs can coexist or switch at runtime.

| object | fixed or reserved at boot | ordinary inference | RL rollout |
| --- | --- | --- | --- |
| target realization | architecture, quantization, KV dtype, layout | weights remain static | trainer publishes a new target/actor version |
| target-matching draft path | alias/direct realization and graphs | static | follows the published target version |
| optional quantized draft | format, kernel, resident buffers, graphs, refresh workspace | initialize once | refresh after each published actor sync |
| shared target KV | alias map and target cache capacity | normal target request lifecycle | same lifecycle |
| action graph pool | admitted window/skip/`K` descriptors | immutable | immutable |

For RL rollout, use a versioned refresh protocol:

```text
target synchronization for version v completes
    -> mark quantized-weight actions ineligible
    -> requantize/copy into boot-reserved buffers and workspace
    -> atomically publish draft_weight_version = v
    -> make matching quantized-weight actions eligible
```

No draft dispatch may observe a partially refreshed weight set. A
double-buffered implementation must reserve both copies at boot. Any stale
draft interval is a separate registered policy, not an implicit part of
weight quantization.

## Workload-conditioned resource gate

Use measured rather than nominal allocations:

```text
M_required(e, w, h, P_h) =
    M_target_fixed(e)
  + M_target_matching_draft_extra(e, h)
  + M_quantized_draft_weights(e, h)
  + M_weight_refresh_peak(h)
  + M_graphs_and_workspaces(h, P_h)
  + M_shared_target_KV(e, w)
  + M_safety
```

A boot class is feasible only when:

```text
M_required <= M_usable_HBM
available_shared_target_KV_blocks(e, h, P_h)
    >= required_live_KV_blocks(w)
M_weight_refresh_host(h) <= M_pinned_host_budget
```

There is exactly one KV block-capacity test. It captures allocator
granularity, fragmentation, graph private pools, and implementation overhead
that byte arithmetic can miss. Weight-refresh workspace is measured at peak
live allocation rather than inferred from checkpoint size.

### Workload descriptor

The workload envelope must declare:

- distribution and hard admission bound for concurrent requests;
- prompt-length and requested-output-length distributions;
- high-percentile sum of live committed KV tokens;
- prefix-cache assumptions and expected hit behavior;
- ordinary-inference versus RL-rollout mode;
- actor-to-rollout synchronization cadence for RL;
- latency/goodput objective and preemption/recompute tolerance; and
- whether a gateway can route workloads to specialized replicas.

Use a high-percentile or explicit admission envelope, not the mean workload.
The percentile and safety margin must be frozen before scoring the planner.

### Two resource gates

1. **Coarse gate:** analytic bytes and prior shared-KV boot ledgers exclude
   only clearly impossible classes. Near-boundary classes remain eligible for
   an exact boot measurement.
2. **Exact gate:** after implementation, record peak allocated/reserved HBM,
   graph-pool memory, refresh workspace, pinned host memory, actual shared-KV
   blocks, maximum admitted live KV, and preemption/recompute behavior on a
   frozen workload trace.

Resource infeasibility is a valid hard elimination. Uncertain performance is
not; feasible uncertain actions proceed to registered measurement.

### Outer objective

The boot planner optimizes service value, not isolated-request speedup:

```text
maximize    robust SLO-qualified service goodput(e, w, h, P_h)
subject to  HBM, shared-KV capacity, graph, correctness,
            refresh, and tail-latency constraints
```

Phase 96's `S_dec` and interval tie-sets remain the inner action signal. The
outer score additionally prices resident capacity, queueing, preemption,
probe duty, transition overhead, graph memory, and RL weight refresh.

## Runtime action registry

Each executable action is a validated descriptor:

```text
action_id
boot_class_id
draft_weight_path_id
realization
window_mode and window_value
skip_set
K
target_graph_descriptor
draft_graph_descriptor
requires_current_draft_weight
legal_switch_predecessors
measured_transition_cost
```

The boot manifest separately records:

```text
kv_path = shared_target
shared_kv_binding_id
target_kv_cache_spec
```

The policy references only `action_id`; it must not independently combine
lever fields. Startup validation rejects an action whose resident weight path,
graph, target configuration, or shared-KV binding does not match the manifest.

`OFF` is always represented explicitly. If `B1` is admitted, at least one
target-matching action and `OFF` remain available while the quantized weights
are unavailable or suboptimal.

## Composition contract

### Why isolated gains are not additive

Layer skip removes complete layer work, including weight and attention work
that quantization or windowing might otherwise save. Quantization and
windowing can shift the bottleneck, and activation quantization, graph
dispatch, or scratchpad gathering can introduce overhead. Consequently, no
design requirement may multiply isolated speedups or demand that every lever
retain its full isolated saving.

The valid requirement is conditional non-regression: an advertised lever must
engage its intended mechanism in the exact composition, and the completed
composition must not underperform the best proper subset or `OFF` outside the
registered tolerance.

### Matched ablations

For a lever set `L` drawn from `{draft-weight-q, window, skip}`, define:

```text
S_dec_star(L, x) = max over admitted K of S_dec(L, K, x)
```

where `x` is one frozen execution cell and workload regime. Every scored
composition must be compared with all registered proper subsets `J` of `L`
and with `OFF` (`S_dec = 1`).

For a one-lever marginal check, `C` and `C minus lever` must match on:

- target and weight versions;
- boot class and shared-KV binding;
- hardware and parallel layout;
- workload trace, batch, context, and generated-suffix state;
- `K`;
- kernel backend;
- graph grade and warmup policy; and
- measurement currency.

Only the lever under test may change. Acceptance is measured for the exact
action; it is never transferred or reconstructed by multiplying singleton
acceptance.

### Mechanism-retention gate

Every cost-reducing lever in a composition must pass both checks:

1. engagement evidence proves that the registered quantized kernel, reduced
   window work, or skipped-layer graph actually executed; and
2. its matched conditional draft-cost marginal is non-regressive within the
   registered cost noise.

Define the observed conditional saving:

```text
Delta_R_obs(lever | C) = R(C minus lever) - R(C)
R(C) <= R(C minus lever) * (1 + epsilon_cost)
```

Here `R` is matched draft-step cost in target-step units, so lower is better.
A composition that claims to retain a lever's expected impact must also compare
that observation with an overlap-aware conditional prediction:

```text
rho_retained(lever | C)
    = Delta_R_obs(lever | C) / Delta_R_expected(lever | C)
LCB(rho_retained(lever | C)) >= rho_min
```

`Delta_R_expected` must account for work already removed by the other levers
and for the predicted bottleneck after composition. It cannot be copied from
the lever's isolated saving. P0 freezes `rho_min` per lever family from
pre-composition evidence, before scored composition results are inspected. If
the expected conditional saving is below the cost-noise floor, the design
makes no retention claim and prefers the simpler subset unless end-to-end
evidence establishes a meaningful benefit.

A masked window that retains maximum-window gathering may be registered as an
acceptance-only action, but it cannot claim the window's expected memory-read
saving.

### End-to-end non-regression gate

Use Phase 96's measured 1.0% noise floor initially:

```text
epsilon_nr = 0.01
S_ref(L, x) = max(1, max over proper subsets J of S_dec_star(J, x))
S_dec_star(L, x)
    >= S_ref(L, x) * (1 - epsilon_nr)
```

If this fails, the full composition is not promoted for that cell; the
selector uses the better subset or `OFF`. A tie does not prove composition
value, and the simpler incumbent is preferred unless keeping the tied action
has a registered exploration purpose.

Claim a positive composition benefit only when the lower confidence bound of
the relative improvement over the best proper subset or `OFF` is at least
2.0%:

```text
LCB((S_dec_star(L, x) - S_ref(L, x)) / S_ref(L, x)) >= 0.02
```

These are inner decode-value gates. Boot-class promotion must additionally
pass the resource-adjusted service objective, including the HBM and shared-KV
capacity cost of a resident quantized draft even when runtime actions bypass
it.

## Runtime profiling and selection

Runtime profiling should reuse ordinary unperturbed serving measurements. A
sync-bracketed region profiler remains diagnostic and cannot decide an action.

For each resident action, retain:

- target request-steps `H`, armed steps `D_arm`, accepted tokens `A`, clipped
  terminal emissions `C`, and committed emissions `E`;
- decode-time deltas and target-step time;
- exact scheduled-KV trace, active-request count, and generated-suffix range;
- target/draft graph descriptors and selected kernel realization;
- target/draft weight versions, refresh time, and quant-path ineligibility;
- shared target-KV blocks, admitted capacity, preemptions, and recomputes; and
- switch, probe, and fallback events.

The selector loop is:

1. mask actions whose weight version, resident objects, or graph transition is
   ineligible;
2. query the Phase 96 cost interval `q_a(x)` at the current execution state;
3. preserve uncertified but feasible cost strata;
4. probe unevaluated or stale resident actions at low duty;
5. form `tau_eff = E/H` and `[tau_lo/q_hi, tau_hi/q_lo]`;
6. construct the interval-dominance epsilon tie-set including `OFF`;
7. apply the matched-subset non-regression mask;
8. prefer the incumbent or simpler subset while evidence remains tied; and
9. periodically re-probe because acceptance changes with content and suffix
   position.

Acceptance evidence is never transferred between window, skip, or
draft-weight actions. Initially one action applies to the whole draft
microbatch. Cohort partitioning is a separate future realization.

## Component flow

```text
fixed target e + workload envelope w
                |
                v
        boot/resource planner -----> boot manifest h + action pool P_h
                                              |
                                              v
actor weight publication ----------> version/eligibility manager
                                              |
                                              v
live counters ---------------------> runtime selector -----> action_id
        ^                                                       |
        |                                                       v
        +-------------------------- graph/action dispatcher <---+
                                      |
                                      v
                         one target-owned shared KV cache
```

The planner never admits a boot class that violates the workload envelope.
The selector never chooses outside the manifest or bypasses weight-version,
graph, subset-value, or shared-KV validation.

## Feature packages

| ID | feature | deliverable | dependency |
| --- | --- | --- | --- |
| F0 | fixed environment and workload schemas | versioned manifests and fail-closed validation | none |
| F1 | shared-KV invariant | one target-owned allocation, alias/spec checks, no private draft pool | F0 |
| F2 | boot resource ledger and planner | measured HBM, graphs, refresh peak, shared-KV blocks, robust class filter | F0, F1 |
| F3 | runtime action registry | validated weight/window/skip/`K` action ids | F0, F1 |
| F4 | passive runtime profiler | action counters, intervals, probes, and tie-set policy | Phase 96 D/E, F3 |
| F5 | runtime window switching | masked grade first; cost-true graph only if valued | Phase 96 F, F3 |
| F6 | runtime layer-skip switching | union registration, per-set graphs, boundary switching | Phase 96 F, F1, F3 |
| F7 | draft-weight capability and refresh | `B0`/`B1`, graph-safe selection, RL version publication | Phase 96 E/F, F2, F3 |
| F8 | composition validation | matched ablations, mechanism retention, subset/`OFF` gate | F4-F7 |
| F9 | integrated workload evaluation | resource-adjusted oracle, selector regret, serving SLOs | selected F4-F8 |

## Work plan

### P0 — scope and schema freeze

**Cost:** document and CPU work only.

- Accept this shared-KV-only scope amendment.
- Update environment, workload, boot-class, action, and memory-ledger schemas.
- Remove KV-path, readiness, private-pool, mirror, and backfill fields.
- Freeze `epsilon_cost`, per-lever `rho_min`, `epsilon_nr`, the +2%
  composition-claim threshold, the workload percentile, and resource safety
  margin.

No engine edit or scored GPU run begins before P0 closes.

### P1 — shared-KV invariant and manifest tests

**Cost:** CPU, synthetic, and existing smoke-test reuse.

**Status:** schema/validator layer complete; strict live preflight is wired for
the minimal target-matching B0. Window/skip union validation remains pending.

- Make Phase 97 manifests require shared KV and reject a draft-only KV dtype.
- Validate layer-to-layer cache specs and tensor-storage identity.
- Validate that skip and window actions preserve cache ownership and true slot
  mapping.
- Assert that KV accounting contains no draft-private allocation.
- Add negative tests for every fail-closed condition.

### P2 — boot resource planner

**Cost:** CPU tooling first; exact non-scored boots only after registration.

**Status:** CPU/evidence layer complete. The checked-in engineering envelope
requires 21,000 shared-KV blocks. Exact minimal B0 has 24,529 and is admitted;
the W4A8-only B1 proxy has an optimistic ceiling of 21,928 but is rejected
because it is not co-resident evidence and its weight, graph/workspace, and RL
refresh quantities are unknown. See `results_resource_preflight.md`.

- Represent only `B0` and `B1`.
- Reuse the existing baseline/shared and W4A8/shared ledgers as lower-bound
  evidence.
- Model target, extra draft weights, refresh peak, graphs/workspaces, and one
  shared target-KV pool.
- Replay frozen workload traces through the block-capacity model.
- After `B1` exists, remeasure its co-resident post-capture capacity; do not
  infer it from the current weight-q-only proxy.

### P3 — registry, K/OFF, and profiler plumbing

**Cost:** CPU and synthetic tests; reuse the existing K/OFF path.

**Status:** complete for minimal B0. Exact B0 exposes only `off` and
`target-matching-k4`; the synthetic replay, live contract tests, focused
scheduler test, and P3b GPU matrix pass. The frozen correctness contract is
target-local. When admission makes a step mixed, the scheduler aborts the
complete pending K4 dispatch, rewrites existing decodes to q=1/OFF before
runner input construction, records discarded width and row count, and re-arms
K4 only after a later pure-decode OFF step. The matched controls retain exact
geometry and top candidate sets; a one-BF16-step history-dependent argmax flip
remains diagnostic rather than overriding each target forward's own argmax.
See `results_p3_koff_runtime.md`, `results_p3_live_engine_wiring.md`,
`results_p3_live_gpu_smoke.md`,
`results_p3_mixed_query_width_diagnosis.md`, and
`results_p3b_mixed_abort.md`. Performance is not claimed. The P3 correctness
blocker for P4 is cleared.

- Add the action registry and boot-manifest validation.
- Validate action-specific accounting, intervals, tie-sets, weight-version
  masks, staleness, probes, and fallback using synthetic traces.
- Carry separate verified and next action ids through scheduler and worker.
- Validate live target/draft weight and KV aliases and the canonical target
  slot-buffer identity.
- Emit passive K/OFF records from synchronous and asynchronous engine paths.
- Run a registered live K4-to-OFF-to-K4 correctness smoke before adding new
  graph mechanisms.
- Abort pending K4 rows at mixed admission and prove q=1/OFF geometry,
  fail-closed provenance, zero discarded-draft accounting, and later K4
  re-entry against shape-matched OFF and fixed-K4 controls.

### P4 — runtime window MVP

**Cost:** engineering plus registered fixed-length confirmation measurements.

**Entry status:** one action is preregistered in
`results_p4_window_entry.md`: `target-matching-w512-masked-k4`. It is a
same-graph, acceptance-only masked action with no cost credit. The entry
validator preserves all 36 base aliases and canonical true-slot identity at
the schema/synthetic layer. The conformant runner now supplies a boot-static
capture realization with live alias and identity validation, but the action is
authorized only for the registered value measurement and is not resource
admitted. Phase 96/F value is `not_approved`; runtime switching, measured
matched controls and value, transition cost, and exact post-capture capacity
remain pending. The follow-up in
`results_p4_b0_value_screen_preregistration.md`
freezes the research objective, equal weights, same-event accounting, exact
three-action comparison, and an acceptance-dominance short circuit. The CPU
capture adapter and scorer contract are now frozen in
`results_p4_b0_adapter_scorer.md`. The additive synchronous wiring result in
`results_p4_live_recorder_wiring.md` clears the implementation blocker without
altering that frozen contract. The additive proof in
`results_p4_w512_acceptance_equivalence.md` clears mask equivalence over all
163,830 registered K4 mask cases while forbidding latency transfer. The
additive bound in `results_p4_b0_resource_bound.md` conservatively debits
2,847 blocks from measured minimal B0 and clears resource readiness with a
682-block floor above the engineering envelope. It does not authorize a GPU
command. The separate review in `results_p4_b0_run_authorization.md` pins the
intended GPU-4-only, nine-boot/432-cell run and remains a historical HOLD. The
additive implementation in `results_p4_b0_capture_runner_conformance.md` now
clears all six live conformance blockers, but changed source hashes invalidate
the old package for execution. The fresh review in
`results_p4_b0_run_authorization_v2.md` validates the current source closure
and authorized the first exact GPU-4 attempt. That attempt is preserved in
`results_p4_b0_value_screen_attempt_v1.md` as an unscored zero-capture launcher
failure. The additive V3 review in `results_p4_b0_run_authorization_v3.md`
bound that failure and PATH/Ninja repair. Its create-new retry is preserved in
`results_p4_b0_value_screen_attempt_v2.md` as another unscored zero-capture
environment failure, this time at FlashInfer sampler loading. V3 is consumed
and its output is immutable. The additive V4 review in
`results_p4_b0_run_authorization_v4.md` forces and preflights the
PyTorch-native sampler before output creation and authorizes exactly one fresh
GPU-4 attempt at the registered `run_b0_value_screen_v3` path. That attempt is
preserved in `results_p4_b0_value_screen_attempt_v3.md` as a zero-complete-
capture mixed-ingress failure. V4 is consumed. The full-prefill runner repair
then passed its CPU checks, but the first exact GPU-4 capacity probe in
`results_p4_b0_full_prefill_resource_probe.md` measured 19,928 blocks and
failed the 21,682-block launch floor without executing a request. The separate
0.96-utilization probe in
`results_p4_b0_full_prefill_resource_repair_probe.md` changed only the
measurement memory utilization and passed with 22,113 blocks. Both probe
authorizations are consumed. The source-bound V5 review in
`results_p4_b0_run_authorization_v5.md` authorized exactly one fresh GPU-4
screen at `run_b0_value_screen_v4`. That attempt is preserved in
`results_p4_b0_value_screen_attempt_v4.md`: the first live OFF event was
score-ineligible, but the rejected event did not persist its exclusion reason.
V5 is consumed with zero captures and no score. The separately authorized
real-serving `8192 / 0.90` diagnosis passed and is recorded in
`results_p4_b0_serving_chunked_prefill_diagnosis.md`. It validates safe mixed
q=1/OFF behavior, not the screen, P4a, or admission. See
`results_p4_f_value_gate.md`. The later source-bound diagnosis and in-process
proof are recorded in `results_p4_b0_full_prefill_ingress_diagnosis.md` and
`results_p4_b0_atomic_ingress_proof.md`. The proof passes the measurement-only
ingress gate. The completed review in
`results_p4_b0_run_authorization_v6.md` binds that mechanism into the scored
runner and authorizes exactly one fresh GPU-4 execution at the create-only
`run_b0_value_screen_v5` path. That attempt is preserved in
`results_p4_b0_value_screen_attempt_v5.md`: the first scheduler event was
rejected because randomized internal request ids did not exactly match the
frozen external prompt ids. V6 is consumed with zero captures and no score.
V7 bound the request-ID repair and reached the first capture cell, where it
exposed the one-token prefill-sample/decode-work offset. Its incomplete,
unscored attempt is preserved in `results_p4_b0_value_screen_attempt_v6.md`.
V8 bound the tested decode-work repair, but its exact launch failed at the
stale reviewed-path dispatcher before GPU work or output creation. That
zero-capture refusal is preserved in
`results_p4_b0_value_screen_attempt_v7.md`. V9 bound one shared
parent/child dispatcher, direct execution-path regressions, and a fresh
`run_b0_value_screen_v8` output; see
`results_p4_b0_run_authorization_v9.md`. V9 was executed exactly once and is
preserved in `results_p4_b0_value_screen_attempt_v8.md`. It completed eight
R4 captures, then OOMed during the first R5 full-prefill before completing the
first physical boot or scoring. The additive diagnosis in
`results_p4_b0_full_prefill_transient_bound.md` rejects a same-geometry V10
and hands off to the CPU-only chunked-prefill cohort-barrier proof in
`results_p4_b0_chunked_prefill_cohort_barrier_design_and_cpu_proof.md`. That
proof passes three action rollovers, ten fail-closed cases, and whole-cohort
abort handling. Live scheduler wiring and its CPU execution-path tests pass.
The separately source-bound V1 non-scored GPU-4 probe then failed before child
launch on a relative output-path normalization omission; its authorization and
create-new path are consumed. The tested dispatcher repair and fresh V2
package passed CPU review. Its later one-shot execution booted the engine,
observed 24,527 shared-KV blocks, and then failed before the first scheduled
request because the live admission guard confused the configured 8,192-token
bound with the designed 8,160-token effective scheduler budget. V2 is
consumed without a cohort or result. The configured/effective guard and
evidence repair now pass real draft-model normalization, fail-closed drift,
and complete scheduler regressions. The separate V3 package was then executed
once. It passed those repaired gates but failed on the first chunked-prefill
model step when a decode-only K4 query-width check was applied despite
`pure_decode=false`. V3 is consumed without a complete cohort or result.
The phase-aware repair and refreshed CPU proof pass. The separate V4 package
was executed once. It completed all three GPU cohort barriers and then failed
the probe-only final request-ID comparison because valid randomized internal
IDs were not canonicalized. V4 is consumed without a probe result or score.
The request-ID repair, direct regressions, full CPU audit, and separate V5
authorization passed. V5 was later executed exactly once and passed all three
target-matching K4 R4/R5/R5cot cohort barriers plus final canonical-ID
validation. Its non-scored result is immutable and does not claim GPU coverage
for OFF, w512, or the full matrix. The source-bound V10 review passed and its
one GPU-4 attempt is preserved in
`results_p4_b0_value_screen_attempt_v9.md`. It completed 72 captures before
the first K4/R8 cell failed. The source-bound comparison in
`results_p4_b0_r5cot_r8_diagnosis.md` proves an R8 variable-width prefill
evidence bug rather than required R5cot history contamination. V10 and both
diagnosis packages are consumed. The subsequent variable-prefill repair keeps
width one strict for pure decode and adds explicit token-count/batch-size work
evidence. Its isolated-R8 and exact transition GPU cases both pass; the parent
aggregate alone rejects an incorrect armed-prefill count assumption. The
source-bound attempt and external immutable audit are documented in
`results_p4_b0_variable_prefill_repair_validation.md`. No score, V11, or retry
authority exists.

The implementation bullets below are conditional on a new passing value
decision.

- Implement scheduler-to-proposer window action ids.
- Validate masked down-selection under one maximum-window graph.
- Label it acceptance-only unless lower read/gather work is demonstrated.
- Add per-window graphs only if cost-true switching clears value and graph-HBM
  gates.

`w-off` remains outside the FULLCG window pool unless separately registered.

### P5 — runtime layer-skip MVP

**Cost:** conditional engineering plus registered measurements.

- Retain the full layer/module and shared-KV alias union.
- Capture only skip sets selected by the portfolio gate.
- Prove target-step-boundary switching and token correctness.
- Measure graph HBM, switch latency, draft cost, and acceptance per set.

### P6 — draft-weight capability, refresh, and selection

**Cost:** co-residency engineering plus registered Phase 96/E/F measurements.

- Implement `B0` and `B1`.
- Build graph-safe paths that select or bypass the resident quantized draft
  without runtime allocation.
- Keep shared target KV identical for both paths.
- For RL, refresh after target publication and atomically gate eligibility.
- For ordinary inference, verify that no refresh fires after initialization.
- Demonstrate fallback to target-matching or `OFF`.
- Measure exact co-resident HBM and shared-KV capacity.

### P7 — matched composition validation

**Cost:** set only after P4-P6 pass their engineering gates.

- Freeze one model, boot, kernel, graph grade, workload trace, and measurement
  currency.
- Measure target-matching baseline, each singleton, required pairs, and each
  shortlisted triple.
- Run same-`K` marginal mechanism checks and optimize `K` separately for the
  proper-subset-and-`OFF` `S_dec_star` comparison.
- Measure overlap-aware conditional-retention ratios for every advertised
  lever.
- Reject cells that fail mechanism retention or the 1% non-regression gate.
- Claim composition value only above the +2% lower-confidence-bound gate.

This is a focused matched-ablation matrix, not another broad profiler sweep.

### P8 — integrated validation

**Cost:** set only after P2 and P7 freeze the boot class and action pool.

- Compare static best, runtime oracle, selector, and AR on frozen traces.
- Report decode speed, service goodput, TTFT/TPOT, p95/p99 latency,
  preemptions, probe duty, switch cost, quantized-weight selection duty,
  refresh unavailability, shared-KV capacity, tie-set size, and regret.
- Repeat the exact memory and graph ledger after integration.

## Provisional gates

Thresholds marked here must be frozen in P0 before scored measurements.

### G0 — scope and schema

- Target quantization appears only in the fixed environment.
- `kv_path=shared_target` appears once in the boot manifest and never as an
  action axis.
- Draft-only KV dtype, private KV, mirrors, and KV conversion fields are
  rejected.
- Every action resolves to one validated boot/action entry.
- Phase 96 artifacts and graph strata remain unchanged.

### G1 — resource feasibility

- The boot fits usable HBM with the declared margin.
- Weight-refresh peak and pinned-host use fit their budgets.
- The single shared target-KV pool covers the frozen workload envelope.
- Preemption/recompute and tail latency satisfy the declared SLO.
- A failing class is removed before runtime profiling.

### G2 — correctness and shared-KV identity

- Every accepted draft equals its target verifier argmax; every replacement
  and bonus token comes from that same target forward.
- Transition correctness uses a same-boot fixed-action control with the same
  registered query shape. Cross-boot sequential AR identity is mandatory only
  if a query-width-invariant target realization is also registered.
- A mixed admission either verifies pending drafts under a registered mixed
  contract or aborts them and runs existing decode requests at q=1/OFF; the
  choice and discarded provenance are explicit.
- Target-step accounting closes exactly.
- Every active draft layer aliases its target twin's KV storage and spec.
- Skip/window switching preserves true slot mapping.
- No draft-private KV allocation appears.
- A quantized action observes one fully published target/draft version.

### G3 — mechanism retention

- Kernel and graph evidence proves the advertised lever executed.
- Every advertised cost-reducing lever passes its matched conditional
  `R(C) <= R(C minus lever) * (1 + epsilon_cost)` check.
- Every material expected conditional saving retains its pre-registered
  fraction `rho_min`; isolated savings are never used as the denominator.
- Acceptance-only masked windows do not claim cost-true window savings.

### G4 — runtime transition cost

- Per-step p95 switch latency is below one target decode step.
- Amortized switch plus probe overhead is below 0.5%, matching Phase 96/F.
- RL refresh time and quantized-path unavailability are included in rollout
  value.

### G5 — selector safety

- Zero false or unresolved Round-1 eliminations in the validated domain.
- The predicted epsilon-optimal tie-set contains the measured best or incurs
  at most 1.5% simple regret.
- Action-specific acceptance is never used as counterfactual evidence for
  another composition.

### G6 — composition non-regression

- `epsilon_nr` is initially 1.0%.
- Every promoted lever set satisfies the proper-subset-and-`OFF`
  `S_dec_star` gate in its declared cell.
- A composition is called beneficial only when its relative improvement has a
  lower confidence bound of at least +2.0%.
- A failing composition is masked; its best subset remains available.

### G7 — resource-adjusted value and recourse

- `B1` retains target-matching and `OFF` recourse.
- Quantized-to-baseline fallback succeeds at a target-step boundary.
- Re-entry requires a current quantized weight version.
- Bypassing quantized weights receives no HBM or graph-memory credit.
- The selected pool yields at least +2% mean resource-adjusted value over the
  best feasible static boot/action, or one workload class at +5% with the mean
  nonnegative, unless P0 registers a stricter serving-SLO rule.

## Open design questions

1. What exact workload contract selects `B0` or `B1`: declared maxima,
   high-percentile traces, or an admission-control envelope?
2. Is the outer objective aggregate goodput, SLO-qualified goodput, cost per
   token, or a lexicographic latency/throughput objective?
3. Should later B1 work retain P4's greater-of-5%-or-4-GiB HBM safety reserve,
   or register a different measured fragmentation allowance?
4. Should a gateway route short- and long-context workloads to specialized
   `B0`/`B1` replicas?
5. Is one action per engine step sufficient, or is cohort partitioning worth
   its extra dispatch and reduced batching?
6. Which window actions require cost-true graphs rather than masked
   acceptance-only switching?
7. How many window and skip graphs fit before capture HBM or startup time
   erases their value?
8. Which skip sets enter the pool: Phase 96 winners, an epsilon-tie-set union,
   or a diversity-constrained subset?
9. Can target-matching and one quantized weight path coexist without module
   and graph duplication erasing weight-quantization value?
10. During RL refresh, should drafting use target-matching self-spec or `OFF`,
    and is in-place graph-stable refresh sufficient?
11. Which weight formats can refresh in-place while retaining the same shared
    target-KV specification and graph contract?
12. Which boot memory and graph metrics are already observable in vLLM, and
    which require new instrumentation?
13. What signal triggers routing or a draining boot-class change when the
    workload distribution shifts?

## Assumptions to challenge

- Shared target KV is valid for every admitted quantized draft-weight
  realization with matching architecture and cache specification.
- A target-matching path can coexist cheaply enough to make the optional
  quantized path bypassable.
- A small prevalidated window/skip/weight action graph captures most runtime
  oracle value.
- Every advertised lever can retain non-regressive conditional draft cost,
  even though isolated savings are not additive.
- One boot choice per engine is sufficient; fleet routing handles strongly
  different memory regimes.
- Passive target-step telemetry is cheap enough for continuous use.
- Resource-adjusted service value may reverse a positive per-request
  `S_dec` result.

## Phase-local artifacts

All Phase 97 documents, scripts, schemas, logs, and data stay under
`research/97_composition_runtime/`.

```text
README.md                         # active shared-KV-only design of record
design_review.md                  # historical four-class diagnostic design
results_boot_proxy.md             # historical boot-resource results
results_warmup_diagnostics.md     # historical private-KV localization
data/boot_resource_ledger.json    # historical four-class ledger
data/warmup_diagnostic_ledger.json
scripts/inventory_boot.py         # existing diagnostic inventory
scripts/run_*.sh                  # historical authorized diagnostic runners
schemas/environment.schema.json   # implemented fixed-environment schema
schemas/workload.schema.json      # implemented workload-envelope schema
schemas/boot_class.schema.json    # implemented shared-KV B0/B1 schema
schemas/boot_candidate.schema.json # exact/projection evidence schema
schemas/action.schema.json        # implemented action-registry schema
schemas/shared_kv_runtime.schema.json
schemas/p4_window_entry.schema.json
schemas/p4_f_value.schema.json
schemas/p4_target_step_event.schema.json
schemas/p4_same_event_accounting.schema.json
schemas/p4_b0_value_screen.schema.json
schemas/p4_prompt_manifest.schema.json
schemas/p4_b0_same_event_capture.schema.json
schemas/p4_b0_adapted_round.schema.json
schemas/p4_b0_runner_scorer.schema.json
schemas/p4_b0_capture_runner_conformance.schema.json
schemas/p4_b0_chunked_prefill_probe_authorization_v3.schema.json
schemas/p4_b0_chunked_prefill_probe_authorization_v4.schema.json
schemas/p4_b0_chunked_prefill_probe_authorization_v5.schema.json
schemas/p4_b0_run_authorization_v2.schema.json
schemas/p4_b0_run_authorization_v3.schema.json
schemas/p4_b0_run_authorization_v4.schema.json
schemas/p4_b0_run_authorization_v5.schema.json
schemas/p4_b0_run_authorization_v6.schema.json
schemas/p4_b0_run_authorization_v7.schema.json
schemas/p4_b0_run_authorization_v10.schema.json
schemas/p4_b0_full_prefill_resource_probe.schema.json
schemas/p4_b0_full_prefill_resource_repair_probe.schema.json
scripts/validate_shared_kv.py     # implemented fail-closed validator
scripts/validate_p4_window_entry.py # P4 preregistration and gate validator
scripts/validate_p4_f_value.py    # Phase 96/F evidence and authority audit
scripts/validate_p4_same_event_accounting.py # per-scheduler-event counters
scripts/validate_p4_b0_value_screen.py # matched research-value preregistration
scripts/generate_p4_prompt_manifest.py # offline exact-token freeze
scripts/validate_p4_prompt_manifest.py # token/provenance validator
scripts/adapt_p4_b0_same_event.py # strict explicit-event capture adapter
scripts/score_p4_b0.py            # W3 episode + paired-block scorer
scripts/run_p4_b0_value_screen.py # fail-closed 9-boot/432-cell runner
scripts/validate_p4_b0_capture_runner_conformance.py
scripts/validate_p4_b0_chunked_prefill_probe_authorization_v3.py
scripts/validate_p4_b0_chunked_prefill_probe_authorization_v4.py
scripts/validate_p4_b0_chunked_prefill_probe_authorization_v5.py
scripts/validate_p4_b0_run_authorization_v2.py
scripts/validate_p4_b0_run_authorization_v3.py
scripts/validate_p4_b0_run_authorization_v4.py
scripts/validate_p4_b0_run_authorization_v5.py
scripts/validate_p4_b0_run_authorization_v6.py
scripts/validate_p4_b0_run_authorization_v7.py
scripts/validate_p4_b0_run_authorization_v10.py
scripts/validate_p4_b0_full_prefill_transient_bound.py
scripts/probe_p4_b0_full_prefill_resource.py # consumed zero-request GPU probe
scripts/probe_p4_b0_full_prefill_resource_repair.py # consumed repair probe
scripts/plan_boot_class.py        # implemented CPU resource planner
tests/test_shared_kv_invariants.py
tests/test_boot_resource_preflight.py
tests/test_koff_runtime.py
tests/test_p4_window_entry.py
tests/test_p4_f_value.py
tests/test_p4_same_event_accounting.py
tests/test_p4_b0_value_screen.py
tests/test_p4_prompt_manifest.py
tests/test_p4_b0_adapter_scorer.py
tests/test_p4_live_recorder.py
tests/test_p4_b0_capture_runner.py
tests/test_p4_b0_capture_runner_conformance.py
tests/test_p4_b0_run_authorization_v2.py
tests/test_p4_b0_run_authorization_v3.py
tests/test_p4_b0_run_authorization_v4.py
tests/test_p4_b0_run_authorization_v5.py
tests/test_p4_b0_run_authorization_v6.py
tests/test_p4_b0_run_authorization_v7.py
tests/test_p4_b0_run_authorization_v10.py
tests/test_p4_b0_full_prefill_transient_bound.py
tests/test_p4_b0_full_prefill_resource_probe.py
tests/test_p4_b0_full_prefill_resource_repair_probe.py
tests/test_p4_b0_full_prefill_ingress_diagnosis.py
tests/test_p4_b0_atomic_ingress_proof.py
tests/test_p4_b0_chunked_prefill_cohort_barrier.py
tests/test_p4_b0_chunked_prefill_probe_authorization.py
tests/test_p4_w512_equivalence.py
tests/test_p4_b0_resource_bound.py
results_shared_kv_invariants.md   # 19-test CPU result
results_resource_preflight.md     # P2 decision and 32-test result
results_p3_koff_runtime.md        # P3 decision and 50-test cumulative result
results_p3_mixed_query_width_diagnosis.md
results_p3b_mixed_abort.md        # repaired mixed-boundary GPU PASS
results_p4_window_entry.md        # w512 masked entry; not admitted
results_p4_f_value_gate.md        # F not approved; value unresolved
results_p4_b0_value_screen_preregistration.md # frozen base screen contract
results_p4_b0_adapter_scorer.md # frozen CPU adapter/scorer pre-wiring snapshot
results_p4_live_recorder_wiring.md # synchronous explicit-event wiring PASS
results_p4_w512_acceptance_equivalence.md # acceptance-only mask proof PASS
results_p4_b0_resource_bound.md # conservative B0 resource readiness PASS
results_p4_b0_capture_runner_conformance.md # six-check CPU conformance PASS
results_p4_b0_run_authorization_v2.md # consumed source-bound GPU-4 approval
results_p4_b0_value_screen_attempt_v1.md # zero-capture V2 failure
results_p4_b0_run_authorization_v3.md # consumed retry approval
results_p4_b0_value_screen_attempt_v2.md # zero-capture V3 failure
results_p4_b0_run_authorization_v4.md # consumed native-sampler retry approval
results_p4_b0_value_screen_attempt_v3.md # zero-capture V4 mixed-ingress failure
results_p4_b0_full_prefill_resource_probe.md # failed 19,928-block probe
results_p4_b0_full_prefill_resource_repair_probe.md # passing 22,113-block probe
results_p4_b0_run_authorization_v5.md # consumed measurement-only GPU-4 approval
results_p4_b0_value_screen_attempt_v4.md # zero-capture V5 eligibility failure
results_p4_b0_run_authorization_v6.md # consumed in-process GPU-4 approval
results_p4_b0_value_screen_attempt_v5.md # zero-capture request-id failure
results_p4_b0_run_authorization_v7.md # consumed request-id repair approval
results_p4_b0_value_screen_attempt_v6.md # incomplete decode-work failure
results_p4_b0_run_authorization_v8.md # consumed decode-work repair approval
results_p4_b0_value_screen_attempt_v7.md # pre-GPU V8 dispatcher refusal
results_p4_b0_run_authorization_v9.md # consumed dispatcher-repair approval
results_p4_b0_run_authorization_v10.md # consumed bounded-cohort approval
results_p4_b0_serving_chunked_prefill_diagnosis.md # 8192/0.90 mixed PASS
results_p4_b0_full_prefill_ingress_diagnosis.md # exact mixed-ingress cause
results_p4_b0_atomic_ingress_proof.md # in-process two-step PASS
results_p4_b0_value_screen_attempt_v8.md # eight R4 captures, R5 OOM
results_p4_b0_value_screen_attempt_v9.md # 72 captures, first K4/R8 failure
results_p4_b0_r5cot_r8_diagnosis.md # isolated R8 proves evidence mismatch
results_p4_b0_variable_prefill_repair_validation.md # both GPU cases PASS; parent observer rejection
results_p4_b0_full_prefill_transient_bound.md # same-geometry V10 rejected
results_p4_b0_chunked_prefill_cohort_barrier_design_and_cpu_proof.md # CPU PASS
results_p4_b0_chunked_prefill_cohort_barrier_live_wiring_and_gpu_probe_authorization.md # live CPU PASS; consumed V1 authority
results_p4_b0_chunked_prefill_probe_attempt_v1.md # pre-child relative-path failure
results_p4_b0_chunked_prefill_probe_authorization_v2.md # repaired V2 approval
results_p4_b0_chunked_prefill_probe_attempt_v2.md # post-boot budget-guard failure
results_p4_b0_chunked_prefill_probe_authorization_v3.md # repaired V3 approval
results_p4_b0_chunked_prefill_probe_attempt_v3.md # prefill query-width failure
results_p4_b0_chunked_prefill_probe_authorization_v4.md # phase-aware V4 approval
results_p4_b0_chunked_prefill_probe_attempt_v4.md # post-cohort ID-normalization failure
results_p4_b0_chunked_prefill_probe_authorization_v5.md # consumed repaired V5 approval
results_p4_b0_chunked_prefill_probe_attempt_v5.md # three-cohort non-scored PASS
data/preflight/*.json             # environment, workload, candidates, plan
data/p3/*.json                    # exact B0 package, trace, and replay result
data/p4/*.json                    # P4 entry, rejected proxy, validation result
data/p4/p4_live_recorder_wiring.json # additive current wiring readiness
data/p4/p4_w512_acceptance_equivalence.json # additive mask readiness
data/p4/p4_b0_conservative_resource_bound.json # additive resource readiness
data/p4/p4_b0_capture_runner_conformance.json # source-bound conformance pass
data/p4/p4_b0_run_authorization_v2.json # consumed exact GPU-4 approval
data/p4/p4_b0_run_authorization_v2_validation.json
data/p4/p4_b0_run_authorization_v3.json # consumed repaired retry approval
data/p4/p4_b0_run_authorization_v3_validation.json
data/p4/p4_b0_run_authorization_v4.json # consumed native-sampler approval
data/p4/p4_b0_run_authorization_v4_validation.json
data/p4/p4_b0_run_authorization_v5.json # consumed full-prefill retry approval
data/p4/p4_b0_run_authorization_v5_validation.json
data/p4/p4_b0_run_authorization_v6.json # consumed one-shot GPU-4 approval
data/p4/p4_b0_run_authorization_v6_validation.json # pre-run review validation
data/p4/p4_b0_run_authorization_v7.json # consumed one-shot GPU-4 approval
data/p4/p4_b0_run_authorization_v7_validation.json # pre-run validation PASS
data/p4/p4_b0_run_authorization_v8.json # consumed one-shot GPU-4 approval
data/p4/p4_b0_run_authorization_v9.json # consumed one-shot GPU-4 approval
data/p4/p4_b0_run_authorization_v10.json # fresh one-shot GPU-4 approval
data/p4/p4_b0_run_authorization_v10_validation.json # CPU PASS
data/p4/p4_b0_chunked_prefill_probe_authorization.json # consumed V1 probe approval
data/p4/p4_b0_chunked_prefill_probe_authorization_validation.json # historical pre-run PASS
data/p4/run_b0_chunked_prefill_probe_v1/failure.json # immutable zero-model failure
data/p4/p4_b0_chunked_prefill_probe_authorization_v2.json # consumed V2 approval
data/p4/p4_b0_chunked_prefill_probe_authorization_v2_validation.json # historical pre-run PASS
data/p4/run_b0_chunked_prefill_probe_v2/failure.json # immutable post-boot failure
data/p4/p4_b0_chunked_prefill_probe_authorization_v3.json # consumed V3 approval
data/p4/p4_b0_chunked_prefill_probe_authorization_v3_validation.json # historical pre-run PASS
data/p4/run_b0_chunked_prefill_probe_v3/failure.json # immutable post-boot failure
data/p4/p4_b0_chunked_prefill_probe_authorization_v4.json # consumed V4 approval
data/p4/p4_b0_chunked_prefill_probe_authorization_v4_validation.json # historical pre-run PASS
data/p4/run_b0_chunked_prefill_probe_v4/failure.json # immutable post-cohort failure
data/p4/p4_b0_chunked_prefill_probe_authorization_v5.json # consumed V5 approval
data/p4/p4_b0_chunked_prefill_probe_authorization_v5_validation.json # historical pre-run PASS
data/p4/run_b0_chunked_prefill_probe_v5/probe_result.json # immutable non-scored PASS
data/p4/p4_b0_serving_chunked_prefill_diagnosis_authorization.json # consumed
data/p4/p4_b0_full_prefill_ingress_diagnosis_authorization.json # consumed
data/p4/p4_b0_atomic_ingress_proof_authorization.json # consumed V1
data/p4/p4_b0_atomic_ingress_proof_authorization_v2.json # consumed PASS
data/p4/p4_b0_atomic_ingress_proof_v1_failure_diagnosis.json # immutable
data/p4/p4_b0_full_prefill_resource_probe_authorization.json # consumed
data/p4/p4_b0_full_prefill_resource_probe.json # immutable capacity failure
data/p4/p4_b0_full_prefill_resource_repair_probe_authorization.json # consumed
data/p4/p4_b0_full_prefill_resource_repair_probe.json # immutable capacity pass
data/p4/p4_b0_full_prefill_transient_bound.json # current geometry rejection
data/p4/p4_b0_full_prefill_transient_bound_validation.json # CPU PASS
data/p4/p4_b0_chunked_prefill_cohort_barrier_design_and_cpu_proof.json # CPU PASS
data/p4/p4_b0_chunked_prefill_cohort_barrier_validation.json # CPU PASS
data/p4/run_b0_value_screen_v1/failure.json # immutable V2 failure record
data/p4/run_b0_value_screen_v2/failure.json # immutable V3 failure record
data/p4/run_b0_value_screen_v3/failure.json # immutable V4 failure record
data/p4/run_b0_value_screen_v4/failure.json # immutable V5 failure record
data/p4/run_b0_value_screen_v5/failure.json # immutable V6 failure record
data/p4/run_b0_value_screen_v6/failure.json # immutable V7 failure record
data/p4/run_b0_value_screen_v7/failure.json # immutable V8 refusal record
data/p4/run_b0_value_screen_v8/failure.json # immutable V9 R5 OOM record
data/p4/run_b0_serving_chunked_prefill_diagnosis_v1/diagnosis.json # PASS
data/p4/run_b0_full_prefill_ingress_diagnosis_v1/diagnosis.json # PASS
data/p4/run_b0_atomic_ingress_proof_v1/failure.json # immutable V1 failure
data/p4/run_b0_atomic_ingress_proof_v2/proof.json # atomic ingress PASS
data/p4/p4_b0_prompt_tokens.jsonl.gz # exact 384-record token bundle
schemas/passive_koff_trace.schema.json
schemas/p4_w512_equivalence.schema.json
schemas/p4_b0_resource_bound.schema.json
scripts/replay_koff_trace.py      # strict passive K4/OFF replay
scripts/verify_p3b_mixed_abort.py # target-local live repair gate
scripts/run_p4_b0_serving_chunked_prefill_diagnosis.py # consumed diagnostic
scripts/run_p4_b0_full_prefill_ingress_diagnosis.py # consumed diagnostic
scripts/run_p4_b0_atomic_ingress_proof.py # consumed in-process proof
scripts/validate_p4_b0_chunked_prefill_cohort_barrier.py # CPU state proof
scripts/validate_p4_w512_equivalence.py # acceptance-equivalence gate
scripts/validate_p4_b0_resource_bound.py # conservative resource gate
```

Historical private-KV runners and ledgers are retained for provenance. They
are not active commands and do not authorize follow-up GPU work.

## Planned commands

CPU and synthetic entry points use the repository environment:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s research/97_composition_runtime/tests \
  -p 'test_*.py' -v
.venv/bin/python research/97_composition_runtime/scripts/plan_boot_class.py --help
.venv/bin/python research/97_composition_runtime/scripts/plan_boot_class.py \
  --environment research/97_composition_runtime/data/preflight/environment_qwen3_8b_h100_tp1.json \
  --workload research/97_composition_runtime/data/preflight/workload_rl_capacity_v1.json \
  --candidate research/97_composition_runtime/data/preflight/candidate_b0_measured.json \
  --candidate research/97_composition_runtime/data/preflight/candidate_b1_projected.json
.venv/bin/python research/97_composition_runtime/scripts/replay_koff_trace.py \
  --environment research/97_composition_runtime/data/preflight/environment_qwen3_8b_h100_tp1.json \
  --workload research/97_composition_runtime/data/preflight/workload_rl_capacity_v1.json \
  --candidate research/97_composition_runtime/data/preflight/candidate_b0_measured.json \
  --boot research/97_composition_runtime/data/p3/boot_b0_minimal_k4.json \
  --actions research/97_composition_runtime/data/p3/actions_b0_minimal_k4.json \
  --runtime research/97_composition_runtime/data/p3/runtime_snapshot_b0_synthetic.json \
  --trace research/97_composition_runtime/data/p3/passive_koff_trace_synthetic.json
.venv/bin/python \
  research/97_composition_runtime/scripts/validate_p4_window_entry.py \
  --entry \
  research/97_composition_runtime/data/p4/p4_window_entry_w512_masked.json
.venv/bin/python \
  research/97_composition_runtime/scripts/validate_p4_f_value.py \
  --decision \
  research/97_composition_runtime/data/p4/p4_f_value_decision.json
.venv/bin/python \
  research/97_composition_runtime/scripts/validate_p4_same_event_accounting.py \
  --contract \
  research/97_composition_runtime/data/p4/p4_same_event_accounting_contract.json
.venv/bin/python \
  research/97_composition_runtime/scripts/validate_p4_prompt_manifest.py \
  --manifest \
  research/97_composition_runtime/data/p4/p4_b0_prompt_manifest.json
.venv/bin/python \
  research/97_composition_runtime/scripts/validate_p4_b0_value_screen.py \
  --preregistration \
  research/97_composition_runtime/data/p4/p4_b0_value_screen_prereg.json
.venv/bin/python \
  research/97_composition_runtime/scripts/score_p4_b0.py \
  --contract \
  research/97_composition_runtime/data/p4/p4_b0_runner_scorer_contract.json
.venv/bin/python \
  research/97_composition_runtime/scripts/validate_p4_w512_equivalence.py \
  --proof \
  research/97_composition_runtime/data/p4/p4_w512_acceptance_equivalence.json
.venv/bin/python \
  research/97_composition_runtime/scripts/validate_p4_b0_resource_bound.py \
  --bound \
  research/97_composition_runtime/data/p4/p4_b0_conservative_resource_bound.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research/97_composition_runtime/scripts/\
validate_p4_b0_chunked_prefill_cohort_barrier.py \
  --artifact \
  research/97_composition_runtime/data/p4/\
p4_b0_chunked_prefill_cohort_barrier_design_and_cpu_proof.json
```

The historical command below consumed V6 and must not be rerun, resumed, or
pointed at another output:

```bash
.venv/bin/python \
  research/97_composition_runtime/scripts/run_p4_b0_value_screen.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_run_authorization_v6.json \
  --output-dir \
  research/97_composition_runtime/data/p4/run_b0_value_screen_v5
```

It stopped at the first scheduler event with zero complete captures and no
score because randomized internal request ids did not exactly match the
frozen prompt ids. See `results_p4_b0_value_screen_attempt_v5.md`. No GPU-5
fallback, retry, partial resume, P4a authority, action admission, or
performance claim is permitted.

V7, V8, and V9 are consumed and must not be rerun. The historical command
below executed V9 exactly once:

```bash
.venv/bin/python \
  research/97_composition_runtime/scripts/run_p4_b0_value_screen.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_run_authorization_v9.json \
  --output-dir \
  research/97_composition_runtime/data/p4/run_b0_value_screen_v8
```

It completed eight R4 captures and stopped on CUDA OOM in the first R5
full-prefill. `run_b0_value_screen_v8` is immutable: do not resume it, score
it, overwrite it, point V9 at a different output, or use GPU 5. See
`results_p4_b0_value_screen_attempt_v8.md`.

The historical V10 command below was executed exactly once and must not be
rerun, resumed, or pointed at another output:

```bash
.venv/bin/python \
  research/97_composition_runtime/scripts/run_p4_b0_value_screen.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_run_authorization_v10.json \
  --output-dir \
  research/97_composition_runtime/data/p4/run_b0_value_screen_v9
```

V10 used only the bounded `8192 / 8160 / 0.90` cohort-barrier path. It completed
72 captures and stopped on the first K4/R8 dispatch without adaptation or
score. `run_b0_value_screen_v9` is immutable and none of its captures may be
reused. See `results_p4_b0_run_authorization_v10.md` and
`results_p4_b0_value_screen_attempt_v9.md`.

The historical GPU-0/GPU-1 diagnosis V2 command is also consumed. Its isolated
R8 and exact R5cot-to-R8 cases both reproduced the registered failure, proving
that the preceding finished IDs are not required. See
`results_p4_b0_r5cot_r8_diagnosis.md`. The earlier V1 diagnosis output is
separately preserved as an inconclusive observer failure. Neither package
authorizes a value-screen retry.

The later source-bound GPU-0/GPU-1 repair-validation command is also consumed
and must not be rerun, resumed, or pointed at another output. Both immutable
case results completed: isolated R8 produced 610 measured K4 events, and the
exact R5cot-tail-to-R8 case produced 1,273. The parent aggregate then rejected
its own assumption that one logical cohort yields one armed-prefill
observation; real chunked R5cot produced eight. The external immutable audit
finds exactly one repaired R8 signature in each case. See
`results_p4_b0_variable_prefill_repair_validation.md`. This closes the GPU
repair evidence only; it did not itself authorize V11 or a value-screen retry.
The separate V11 review in `results_p4_b0_run_authorization_v11.md` now passes
without GPU execution.

The following V11 command was registered and has been consumed by one
interrupted attempt:

```bash
.venv/bin/python \
  research/97_composition_runtime/scripts/run_p4_b0_value_screen.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_run_authorization_v11.json \
  --output-dir \
  research/97_composition_runtime/data/p4/run_b0_value_screen_v10
```

The V12 command below was consumed by one pre-GPU refusal that emitted zero
captures; its output is preserved and never reused:

```bash
.venv/bin/python \
  research/97_composition_runtime/scripts/run_p4_b0_value_screen_v12.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_run_authorization_v12.json \
  --output-dir \
  research/97_composition_runtime/data/p4/run_b0_value_screen_v11
```

The current authority is the repaired V13 package:

```bash
.venv/bin/python \
  research/97_composition_runtime/scripts/run_p4_b0_value_screen_v12.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_run_authorization_v13.json \
  --output-dir \
  research/97_composition_runtime/data/p4/run_b0_value_screen_v12
```

V12 adds these phase-local artifacts:

```text
results_p4_b0_run_authorization_v12.md
data/p4/p4_b0_run_authorization_v12.json
data/p4/p4_b0_run_authorization_v12_validation.json
schemas/p4_b0_run_authorization_v12.schema.json
scripts/run_p4_b0_value_screen_v12.py
scripts/validate_p4_b0_run_authorization_v12.py
tests/test_p4_b0_run_authorization_v12.py
tests/test_p4_b0_value_screen_v12.py
```

It permits one GPU-4 parent launch, nine sequential boots, and scoring only
after all 432 captures complete. The output must be absent at launch. V10
capture reuse, repair-output reuse, partial resume, retry, fallback GPU, P4a,
action admission, and production-value claims remain forbidden.

The historical command below consumed V5 and must not be rerun, resumed, or
pointed at another output:

```bash
.venv/bin/python \
  research/97_composition_runtime/scripts/run_p4_b0_value_screen.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_run_authorization_v5.json \
  --output-dir \
  research/97_composition_runtime/data/p4/run_b0_value_screen_v4
```

This failed with zero complete captures and no score. The measurement-only
`114688 / 0.96` configuration is not a serving default.

The historical command below separately consumed the one-boot real-serving
diagnosis authorization and must not be rerun or pointed at another output:

```bash
.venv/bin/python \
  research/97_composition_runtime/scripts/run_p4_b0_serving_chunked_prefill_diagnosis.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_serving_chunked_prefill_diagnosis_authorization.json \
  --output-dir \
  research/97_composition_runtime/data/p4/run_b0_serving_chunked_prefill_diagnosis_v1
```

That non-scored diagnosis passed the `8192 / 0.90` mixed-boundary contract. It
does not authorize a value-screen retry.

The historical command below consumed the full-prefill ingress diagnosis and
must not be rerun, resumed, or pointed at another output:

```bash
.venv/bin/python \
  research/97_composition_runtime/scripts/run_p4_b0_full_prefill_ingress_diagnosis.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_full_prefill_ingress_diagnosis_authorization.json \
  --output-dir \
  research/97_composition_runtime/data/p4/run_b0_full_prefill_ingress_diagnosis_v1
```

That non-scored diagnosis identified `prefill_or_mixed_batch` as the sole V5
event exclusion. It emitted no complete capture, adapted round, or score and
does not authorize a value-screen retry.

The historical command below consumed the passing V2 atomic-ingress proof and
must not be rerun, resumed, or pointed at another output:

```bash
.venv/bin/python \
  research/97_composition_runtime/scripts/run_p4_b0_atomic_ingress_proof.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_atomic_ingress_proof_authorization_v2.json \
  --output-dir \
  research/97_composition_runtime/data/p4/run_b0_atomic_ingress_proof_v2
```

The earlier V1 proof authorization and output are also consumed and immutable.
V1 stopped before model-weight load because its disabled integer capture
setting was encoded as an empty string. V2 passed the non-scored two-step
contract and permitted only a fresh V6 authorization review. That review has
now passed; the proof itself remains non-scored and consumed.

The historical command below consumed V4 and must not be rerun, resumed, or
pointed at another output:

```bash
.venv/bin/python \
  research/97_composition_runtime/scripts/run_p4_b0_value_screen.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_run_authorization_v4.json \
  --output-dir \
  research/97_composition_runtime/data/p4/run_b0_value_screen_v3
```

The historical zero-request command below consumed the full-prefill resource
probe authorization and must not be rerun or pointed at another output:

```bash
.venv/bin/python \
  research/97_composition_runtime/scripts/probe_p4_b0_full_prefill_resource.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_full_prefill_resource_probe_authorization.json \
  --output \
  research/97_composition_runtime/data/p4/p4_b0_full_prefill_resource_probe.json
```

The historical repair command below also consumed its zero-request probe
authorization and must not be rerun or pointed at another output:

```bash
.venv/bin/python \
  research/97_composition_runtime/scripts/probe_p4_b0_full_prefill_resource_repair.py \
  --authorization \
  research/97_composition_runtime/data/p4/p4_b0_full_prefill_resource_repair_probe_authorization.json \
  --output \
  research/97_composition_runtime/data/p4/p4_b0_full_prefill_resource_repair_probe.json
```

The planner performs the frozen live-KV trace replay directly. The checked-in
workload remains a non-scored engineering assumption and must be replaced by
a measured trace before service-value evaluation.

Any other future GPU runner must be separately registered, use Phase 96's
equal-work rules, record its registration hash, and change only the declared
matched-ablation field.

## Decision criteria

Phase 97 may enter runtime engineering only when:

1. the fixed environment, workload, `B0`/`B1`, and action boundaries are
   accepted;
2. shared target KV is fail-closed in the manifest and absent from the action
   axes;
3. the workload memory envelope and outer service objective are frozen;
4. weight-version publication and ordinary-inference versus RL lifecycles are
   accepted;
5. the initial graph/action pool and optional-weight bypass path are chosen;
6. `epsilon_cost`, each `rho_min`, the 1% non-regression tolerance, the +2%
   benefit threshold, and every resource threshold have frozen measurement
   sources; and
7. Phase 96/F supplies a pool whose robust resource-adjusted value justifies
   the requested runtime mechanisms.

## Expected next artifact

The matched B0 value screen is preregistered in
`results_p4_b0_value_screen_preregistration.md`. Its frozen snapshot remains
blocked and every authorization in it is false. The additive recorder,
equivalence, and conservative-resource artifacts clear the evidence gaps
without rewriting that snapshot. The historical review in
`results_p4_b0_run_authorization.md` records a HOLD against the prior source.
`results_p4_b0_capture_runner_conformance.md` now clears all six registered
implementation checks under CPU tests, while preserving that HOLD and
rejecting it as execution authority.

The V2 review in `results_p4_b0_run_authorization_v2.md` bound the changed
runtime, scheduler, worker, runner, tests, and conformance hashes. Its exact
GPU-4 attempt stopped before capture because `.venv/bin` was absent from the
child `PATH`; `results_p4_b0_value_screen_attempt_v1.md` records zero captures,
no score, and immutable failed output. V2 is consumed.

The fresh review in `results_p4_b0_run_authorization_v3.md` bound that failure,
the launcher repair, its tests, and a new output path. Its one attempt passed
the Ninja preflight but failed before capture because the cached FlashInfer
sampler's `libcudart.so.13` dependency was not resolvable. The immutable record
in `results_p4_b0_value_screen_attempt_v2.md` reports zero captures and no
score. V3 and both attempted output paths are consumed.

The V4 review in `results_p4_b0_run_authorization_v4.md` binds the native
sampler repair, runner and test hashes, GPU 4, and the fresh create-only
`run_b0_value_screen_v3` path. Its isolated preflight proves
`TopKTopPSampler.forward_native` before output creation, while retaining the
virtualenv/Ninja preflight. Its attempt passed those gates and engine
initialization but stopped at the first cell's mixed prefill/decode transition.
`results_p4_b0_value_screen_attempt_v3.md` records zero complete captures, no
score, consumed V4 authority, and immutable output.

The manifest-derived full-prefill budget covers the 112,908-token maximum
frozen microbatch plus speculative slot reserve while retaining chunked
prefill, but the live ingress diagnosis disproves that budget coverage alone
preserves the pure-decode invariant. Its first consumed one-initialization
probe is recorded in
`results_p4_b0_full_prefill_resource_probe.md`: GPU 4 exposed only 19,928
shared-KV blocks, below both the 21,682-block launch floor and the 21,000-block
live-workload requirement. The separate repair result in
`results_p4_b0_full_prefill_resource_repair_probe.md` changed only measurement
GPU memory utilization to 0.96 and passed with 22,113 blocks, 431 above the
launch floor. It executed zero requests and does not validate serving.

The V5 review in `results_p4_b0_run_authorization_v5.md` bound the passing
resource result, source closure, GPU 4, nine boots, 432 cells, and a fresh
create-only `run_b0_value_screen_v4` path. Its one attempt passed engine
initialization but stopped at the first score-ineligible live OFF event. The
rejected event was not serialized in that attempt.
`results_p4_b0_value_screen_attempt_v4.md` records zero complete captures, no
score, consumed V5 authority, and immutable output.

The separately authorized full-prefill ingress diagnosis reused that exact
first OFF boot, cell, workload, and `114688 / 0.96` envelope while flushing a
passive trace before the strict recorder. Its two records prove staggered
multiprocess admission: p000 prefilled alone, then entered width-one decode
while p001-p007 scheduled their full prefills. The sole event exclusion was
`prefill_or_mixed_batch`; there was no preemption, recomputation, or
invalid-spec-token exclusion. See
`results_p4_b0_full_prefill_ingress_diagnosis.md`. That one-boot authority is
consumed and the output is non-scored.

The subsequent atomic-ingress package first consumed V1 before model load on
an empty-string integer environment error, then used a fresh source-bound V2
authorization for the one permitted repair attempt. V2 selected
`InprocClient`, queued all eight requests before the first `engine.step()`, and
produced exactly one eight-request pure-prefill step followed by one
eight-request q=1/OFF pure-decode step. The decode had no exclusion,
preemption, recomputation, or draft dispatch; all 36 shared-KV and 291
weight-alias invariants held. See
`results_p4_b0_atomic_ingress_proof.md`. Both proof authorizations and outputs
are consumed and immutable. The source-bound review in
`results_p4_b0_run_authorization_v6.md` now binds the value-screen runner to
this in-process queue-all path and passes. Its one execution is recorded in
`results_p4_b0_value_screen_attempt_v5.md`. All eight requests were queued,
but their randomized internal ids failed the recorder's exact frozen-id check
before the first event was persisted. V6 is consumed with zero complete
captures and no score. The shared strict canonicalizer and its recorder/proof
regressions now pass, and the fresh source-bound review is recorded in
`results_p4_b0_run_authorization_v7.md`. V7 binds the current sources and the
new create-only `run_b0_value_screen_v6` path without disabling vLLM request
randomization. Its execution is preserved in
`results_p4_b0_value_screen_attempt_v6.md`: all request-ID and shared-KV gates
passed, but one token sampled during prefill left the decode-only recorder one
commit short per request. V7 is consumed with one incomplete capture and no
score. V8 retained the decode-only `S_dec` currency while requesting one
additional frontend token; OFF, K4, and W512 rollover tests passed. Its exact
launch nevertheless failed before GPU preflight because the final parent and
child dispatchers omitted V8. The zero-GPU refusal is preserved in
`results_p4_b0_value_screen_attempt_v7.md`, and V8 is consumed. V9 replaces
both branches with one reviewed authorization-to-output resolver, directly
tests parent and child execution paths, and binds the fresh create-only
`run_b0_value_screen_v8` path. Its one execution completed all eight R4 OFF
captures, then OOMed in the first 112,304-token R5 full-prefill. The immutable
partial result and consumed authority are recorded in
`results_p4_b0_value_screen_attempt_v8.md`.

The follow-up bound in `results_p4_b0_full_prefill_transient_bound.md`
corrects the 0.50 GiB graph estimate against the 4.90 GiB actual pool and
derives R5/R5cot compiled-MLP live-set lower bounds. It rejects another
`114688 / 0.96` full-microbatch launch. The CPU-only design and fail-closed
proof in
`results_p4_b0_chunked_prefill_cohort_barrier_design_and_cpu_proof.md` now
passes at the bounded 8,192-token scheduler budget. It proves early-release
prevention, exact identity and one-plus-512 accounting, whole-cohort abort,
deadlock refusal, and isolated OFF/K4/w512 rollover. Its executable state
machine is now connected to synchronous scheduler dispatch through the exact
measurement-only request contract documented in
`results_p4_b0_chunked_prefill_cohort_barrier_live_wiring_and_gpu_probe_authorization.md`.
Execution-path CPU tests prove held-request skipping, per-request final-prefill
arming, preservation of early K4 drafts, atomic pure-decode release, complete
cohort cancellation, incomplete-shutdown refusal, and ordinary-serving bypass.
The separate source-bound V1 package authorized one non-scored K4 boot on
physical GPU 4 over R4/R5/R5cot at `8192 / 0.90`. Its exact invocation passed
sampler, EngineCore, and GPU identity/idle preflights, then created the output
directory and failed before preparation or child launch because the relative
output path was not normalized. The immutable refusal is recorded in
`results_p4_b0_chunked_prefill_probe_attempt_v1.md`. V1 is consumed; no model,
cohort, probe result, or score exists. Another invocation requires a tested
path repair, a fresh source-bound authorization, and a create-only V2 output.
Those CPU gates passed in
`results_p4_b0_chunked_prefill_probe_authorization_v2.md`. The later exact V2
execution booted successfully and passed its resource check, then the first
request exposed a guard mismatch: 8,192 is the configured batch bound, while
8,160 is the expected effective scheduler budget after the 32 speculative
slots are reserved. V2 is consumed with no completed cohort, result, or score;
see `results_p4_b0_chunked_prefill_probe_attempt_v2.md`. The required
real-config regression, narrow guard/evidence repair, and fresh source-bound
authorization now pass in
`results_p4_b0_chunked_prefill_probe_authorization_v3.md`. V3 authorized one
non-scored GPU-4 probe. Its later exact one-shot execution passed engine,
resource, and repaired budget gates, then rejected the first pure-prefill
event because the runner evidence builder treated its 4,081-token draft query
width as a width-one decode. V3 and its output are consumed with no complete
cohort, result, or score; see
`results_p4_b0_chunked_prefill_probe_attempt_v3.md`. A narrow phase-aware
guard repair, paired CPU regressions, and refreshed active proof now pass. The
fresh source-bound V4 authorization is documented in
`results_p4_b0_chunked_prefill_probe_authorization_v4.md`. Its later one-shot
execution completed all three cohort barriers, with valid width-one K4 decode
events and zero quality violations, then failed the probe-only final identity
check because randomized internal request IDs were compared directly with
frozen IDs. V4 is consumed without a result or score; see
`results_p4_b0_chunked_prefill_probe_attempt_v4.md`. The shared request-ID
canonicalization repair, positive and fail-closed regressions, and complete
CPU audit passed. The separate source-bound V5 package in
`results_p4_b0_chunked_prefill_probe_authorization_v5.md` was then consumed by
one exact GPU-4 execution. All three target-matching K4 R4/R5/R5cot cohort
barriers and the canonical-ID result check passed; see
`results_p4_b0_chunked_prefill_probe_attempt_v5.md`. The result is non-scored,
does not cover OFF or w512 on GPU, and grants no downstream authority. The
separate source-bound V10 package was consumed by one GPU-4 value-screen
attempt. It completed 72 captures and failed at the first K4/R8 dispatch. The
subsequent GPU-0/GPU-1 comparison reproduces the failure with R8 alone and
after the exact R5cot tail, so the active blocker is variable-width prefill
evidence rather than transition history. See
`results_p4_b0_value_screen_attempt_v9.md` and
`results_p4_b0_r5cot_r8_diagnosis.md`. The subsequent exact evidence-schema
repair and GPU validation are recorded in
`results_p4_b0_variable_prefill_repair_validation.md`: both GPU cases pass,
while the consumed parent aggregate rejects only its incorrect observation
count. The separate source-bound V11 review passed and registered only the
fresh `run_b0_value_screen_v10` output; no existing output is reusable. V11
is documented in `results_p4_b0_run_authorization_v11.md` and has since been
consumed by the interrupted attempt described below.

V11 was then executed once on GPU 4. It completed the first boot's 48
captures and was interrupted when another user reserved that GPU; the
immutable record classifies the stop as `external_resource_reassignment`
with no runtime fault, and forbids retry, resume, fallback, and scoring.
Two contention probes then tried and failed to produce a GPU-0/GPU-1
contention bound, both in the launcher: V1 registered a rendezvous range
overlapping the host ephemeral range, and V2 completed `serial-gpu0` before
refusing `serial-gpu1` on a capture-adapter hash that a concurrent staging
pass had rewritten inside the run window. Both probes are consumed and no
contention bound exists.

The source-bound V12 review therefore removes the contention probe from the
critical path and authorizes a block-parallel two-lane screen instead, on
Phase 96's registered block-per-GPU precedent (read-only, measured lane
spread 0.45%/0.49%). Blocks are never split: lane-a takes blocks 1 and 3 on
GPU 0, lane-b takes block 2 on GPU 1. The transfer rests on properties of
the frozen matrix and scorer rather than new assumptions: acceptance is a
pure count ratio, committed tokens per action are equal, the Latin square
gives every action an identical lane mixture, and the residual block-by-action
interaction is exactly what the frozen paired complete-boot-block bootstrap
resamples. The 2% cross-boot certification is retained as a fail-closed lane
tripwire. V12 also closes the V2 failure mode structurally, with a verified
source snapshot that children resolve against, a byte-for-byte executing-code
guard, a worktree-quiescence preflight, and the capture block as the restart
unit. V12 passes CPU validation, registers the fresh create-only
`run_b0_value_screen_v11` output, and remains unexecuted. See
`results_p4_b0_run_authorization_v12.md`.

The separate source-bound diagnosis retained the real serving boundary at
budget 8192 and utilization 0.90. Its 24-record non-scored trace contains one
pure prefill, eight natural mixed prefill/decode steps forced to q=1/OFF, and
15 later pure-decode steps. See
`results_p4_b0_serving_chunked_prefill_diagnosis.md`. The measurement-only
`114688 / 0.96` repair is not a serving default, and the serving PASS does not
repair capture ingress. P4a, action admission, and performance claims remain
false.

After a future complete authorized screen, the scorer first applies its
acceptance-dominance short circuit. If that fires, P4 stops and K4/OFF remain
the executable pool. A later value pass would still require explicit P4a
authority, exact live aliases, target-local controls, transition overhead, and
post-engineering resource remeasurement before admission. P5 skip and B1
remain blocked; private-KV profiling stays cancelled.
