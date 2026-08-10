# Phase 97 P3 diagnosis — mixed prefill/decode query width

Follow-up: the recommended abort policy was implemented and passed in
`results_p3b_mixed_abort.md`. The diagnosis below remains the historical basis
for the target-local correctness contract.

Status: **DIAGNOSED, NOT REPAIRED. The live `[q5 decode, q10 prefill]`
packing and target-owned slot mappings are correct. The failed cross-process
greedy-identity check is an execution-shape numerical failure near BF16 ties,
not a stale index, overlapping slot, shared-KV alias, or K/OFF transport
failure. P4 remains blocked pending a correctness-contract decision and the
corresponding boundary policy.**

Date: 2026-08-09.

## Scope

This was a correctness-only diagnosis of the first mismatch in
`results_p3_live_gpu_smoke.md`. It did not profile latency, throughput, or
kernels and did not test window, skip, quantization, or private-KV features.

The diagnostic trace extends an explicitly enabled K/OFF JSONL stream with
the exact runner request order, query widths, query starts, input ids,
positions, sequence lengths, target slot values, verifier-logit indices,
draft ids, and top target logits. The added device-to-host copies occur only
when `VLLM_SELF_SPEC_KOFF_TRACE` names a create-new diagnostic artifact.

## Exact failing geometry

At engine step 4 the target runner received:

```text
request order          [long, short]
scheduled widths       [5, 10]
draft widths           [4, 0]
query_start_loc        [0, 5, 15]
positions              [35,36,37,38,39, 0,1,2,3,4,5,6,7,8,9]
target slots           [51,52,53,54,55, 64,65,66,67,68,69,70,71,72,73]
selected logit rows    [0,1,2,3,4,14]
draft-target rows      [0,1,2,3]
bonus rows             [4,5]
```

The long input row is the last committed token followed by its four drafts:

```text
[13, 22406, 11, 10339, 279]
```

The short row is exactly its ten-token prompt. Long and short positions are
contiguous within each request, their target slots are disjoint, the short
sample uses only its prompt-tail row 14, and the long verifier uses rows 0–4.
This is the intended ragged speculative-decoding layout.

The target argmax rows were:

```text
[22406, 11, 10339, 279, 6672, 576]
```

They match all four scheduled long drafts, so accepting four drafts and one
bonus was internally correct for that target forward.

## Controls

All controls used the same Qwen3-8B checkpoint, BF16 target/KV, native greedy
sampler, V1 runner, GPUs 4–5, and no profiler.

| control | relevant target shape | result at long token 12 |
| --- | --- | --- |
| registered mixed K4-to-OFF | `[5,10]`, `NONE` | `22406`; logits `25.875` vs `25.75` for `5005` |
| K4 with no short admission | `[5]`, `FULL` | same `22406` and same top logits |
| always-OFF shared-KV boot | q=1, `FULL` at position 35 | same `22406` and same top logits |
| repeated paired run with top-10 logprobs | switched K4 and independent AR | both chose `5005`; both candidates had logprob `-1.4283871651` |
| batch-invariant switched versus independent AR | q=5 versus separately booted q=1 | still diverged at token 12, in the opposite direction |
| batch-invariant q=5 versus q=1 inside speculative boots | same checkpoint and speculative boot class | both chose `5005` at token 12; later diverged at token 15 by one BF16 logit step |

The no-admission control rules out request admission and the K4-to-OFF
selection itself. The always-OFF control performs no draft dispatch and rules
out provisional shared-draft KV as a necessary cause. Both reproduce the
same token under the speculative boot.

The batch-invariant within-boot comparison exposes the remaining numerical
shape effect directly. At long position 38, the q=5 verifier produced:

```text
token 279: 28.625
token 264: 28.500
```

The q=1 path produced a tie:

```text
token 279: 28.625
token 264: 28.625
```

The q=5 path selected `279`; q=1 selected the lower-id tied token `264`.
Thus even vLLM batch-invariant mode did not make causal q=5 verification and
sequential q=1 decode bit-identical. It also did not make independently
compiled speculative and non-speculative boots identical.

## Diagnosis

The original mismatch happened at the first near-degenerate argmax reached by
the test and therefore coincided with the mixed boundary. That correlation
was not evidence of malformed mixed-batch packing.

The actual failure is the test oracle's assumption that these are one exact
numerical realization:

```text
BF16 target, q=5 causal verification, speculative boot
    ==
BF16 target, sequential q=1 decode, separately compiled AR boot
```

They are mathematically equivalent but need not be bit-identical. Query
width, causal/decode attention path, padding/graph realization, and separate
compilation can alter reductions by one BF16 step. A zero- or one-step top-logit
margin can then flip greedy output and all subsequent tokens. The observed
evidence localizes the issue to target execution-shape numerics; it does not
localize a particular layer or kernel, and no further kernel profiling is
authorized or needed for the current research decision.

## Correctness-contract decision

Do not repair the verified packing or allocate private KV. Instead choose one
of two explicit contracts before P4:

1. **Target-local speculative correctness (recommended).** Require every
   accepted draft to equal the argmax of the target verifier row that accepts
   it, and every replacement/bonus token to come from that same target
   forward. Compare transitions against a same-boot fixed-action control with
   the same query shape. Treat independently booted sequential AR identity as
   diagnostic near numerical ties, not as the sole pass gate.
2. **Strict sequential-AR identity.** Supply a target implementation that is
   query-width invariant across q=1 and q=K+1, or force q=1 target execution.
   Current `VLLM_BATCH_INVARIANT=1` is insufficient, and forcing q=1 would
   remove the normal parallel verifier realization and requires a new value
   assessment.

Independently of that global choice, the safest mixed-boundary policy is to
abort pending K4 drafts when a newly admitted prefill makes the step mixed,
run the existing decode requests at q=1/OFF, record the discarded K4
provenance explicitly, and re-arm K4 only at a later pure-decode target-step
boundary. This removes the q=5-to-q=1 transition ambiguity at rare admission
boundaries without changing steady-state shared-KV ownership.

## Repair gate

P4 stays blocked until the next change does all of the following:

1. registers whether mixed admission verifies or aborts pending drafts;
2. if aborting, proves mixed decode widths are q=1 and no discarded draft is
   accepted or counted;
3. compares the boundary against a same-boot OFF control and stable-K4 steps
   against a same-boot fixed-K4 control;
4. retains exact target/draft weight aliases, all 36 KV aliases, disjoint true
   slot values, q=1 OFF-to-K4 bootstrap, and H/D/A/C/E closure; and
5. keeps cross-boot AR output as a reported diagnostic with top-logit margins
   rather than misclassifying a one-step BF16 flip as a slot-mapping failure.

## Artifacts

- `data/p3/mixed_width_diag_20260809_trace.jsonl`: exact failing geometry and
  raw top logits;
- `data/p3/mixed_width_diag_20260809_spec.json`: reproduced switched output;
- `data/p3/mixed_width_control_no_admission_*`: stable-K4/no-admission control;
- `data/p3/mixed_width_control_off_admission_*`: q=1 shared-KV OFF control;
- `data/p3/mixed_width_20260809_{spec,ar}.json`: paired top-logprob control;
- `data/p3/mixed_width_batch_invariant_*`: cross-boot and within-boot
  batch-invariant controls; and
- corresponding `logs/mixed_width_*.log`: boot and completion logs.
