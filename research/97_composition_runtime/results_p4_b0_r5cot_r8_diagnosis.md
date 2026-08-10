# Phase 97 P4 — K4 R5cot-to-R8 transition diagnosis

Status: **GPU DIAGNOSIS COMPLETE. R8 variable-width pure prefill reproduces
the failure in isolation; preceding R5cot finished IDs are not required. The
result is non-scored and grants no retry or downstream authority.**

Date: 2026-08-10.

## Outcome

Two source-bound cases ran concurrently with identical K4, shared-KV, model,
prompt, and `8192 / 8160 / 0.90` engine contracts:

| Case | GPU | Prior work | R8 result |
|---|---:|---|---|
| isolated R8 | 0 | none; `finished_req_ids=[]` | registered K/OFF failure |
| R5cot tail to R8 | 1 | exact prompts 24–31; 618 complete K4 events | same failure |

Both cases had 24,527 target-owned KV blocks, 291 target/draft weight aliases,
36 shared target-KV layer aliases, no private draft KV, and zero preemption or
recomputation. Both released their GPUs after preserving their results.

The V1 diagnosis package is separately preserved as inconclusive. Its
read-only observer attempted to iterate `CachedRequestData` before the first
workload model step. V2 changed only that observer to read `.req_ids`, passed
its CPU regression, and used fresh create-only outputs.

## Decisive evidence

The first R8 target step is a valid armed pure-prefill step in both cases:

```text
scheduled target prompt tokens                              2,100
new R8 requests                                                 16
capture_cohort_arm                                            true
decode_req_ids                                                  []
next action                                    target-matching-k4
proposal called                                                 true
draft output shape                                           [16, 4]
draft step-0 runtime mode                                      NONE
draft chain runtime mode                                  PIECEWISE
```

The draft's step-0 input contains the 2,100 prompt tokens plus one newly
sampled token for each of the 16 requests:

```text
draft step-0 tokens                                          2,116
draft batch size                                                16
2,116 mod 16                                                     4
maximum per-request query width                                344
uniform query width                                           none
```

The proposer resets `_last_step0_query_width` to `None` and assigns it only
when `num_tokens % batch_size == 0`. R8 has unequal prompt lengths, so there is
no truthful uniform scalar. The proposal nevertheless executes and returns
four draft tokens for every request. `make_runner_evidence()` then treats the
missing uniform scalar as if the non-decode draft had not executed and raises
the registered invariant.

The transition case independently closes the exact R5cot tail first. Its last
event contains requests 30 and 31, exactly matching the V10 boundary. Those
two finished IDs remain on the following scheduler output, but the isolated
case has no finished IDs and reaches identical R8 draft inputs, output shape,
runtime modes, and exception. Therefore stale finished IDs are observable but
not causal.

## Root cause and repair boundary

The bug is an evidence-schema mismatch: one field currently means both
"uniform decode query width" and "some positive prefill draft work." Those
are different facts.

The repair should preserve strict width-one validation for pure decode while
recording non-decode work separately, for example:

- keep a nullable uniform query width for exact decode checks;
- add step-0 token count and batch size, or an explicit positive-prefill-work
  fact, for variable-width armed prefill; and
- require the existing cohort arm, empty decode set, proposal-called flag,
  `[batch, K]` output, and execution modes without inventing an average query
  width.

Add both isolated-R8 and R5cot-to-R8 execution-path regressions before any new
value-screen authorization. Do not patch V10 in place or reuse its 72
captures.

## Artifacts

- V2 authorization:
  `data/p4/p4_b0_r5cot_r8_diagnosis_authorization_v2.json`
  (`8fceecacae885465320a5f586101151b3b36b84927d72e1ab70739574e7ce220`);
- aggregate diagnosis:
  `data/p4/run_b0_r5cot_r8_diagnosis_v2/diagnosis.json`
  (`17e04cf5ec8d43786a61d5414c860e21c7bdfc3d242506db6de4842fdea92a6b`);
- isolated-R8 result:
  `data/p4/run_b0_r5cot_r8_diagnosis_v2/isolated-r8/case_result.json`
  (`7f849ced4e9bb8ddde250b9d6e1a6246eda211da9fca107f06a53076496dbf41`);
- transition result:
  `data/p4/run_b0_r5cot_r8_diagnosis_v2/r5cot-to-r8/case_result.json`
  (`31a4d8b328eec528090f73f852486ba1c6d5f2b0d417136e62a5ee2b36a033a2`);
- transition K/OFF trace:
  `data/p4/run_b0_r5cot_r8_diagnosis_v2/r5cot-to-r8/koff_trace.jsonl`
  (`75bb84a1445f4da260922567761cbf5f59473f8f39359dc0ac50cf4123d9a857`);
  and
- consumed V1 harness failure:
  `data/p4/run_b0_r5cot_r8_diagnosis_v1/failure.json`
  (`62b0a48b39688ec7bcbbe069acc8932e4d24aa2b041f8702e76a3c9cca0b18bf`).
