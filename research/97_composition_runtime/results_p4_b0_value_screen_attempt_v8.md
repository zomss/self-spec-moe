# Phase 97 P4 — B0 value-screen attempt V8

Status: **STOPPED on CUDA OOM in the first R5 full-prefill after eight complete
R4 captures. V9 is consumed. The partial output is not scoreable and must not
be retried, resumed, or moved to another GPU.**

Date: 2026-08-09.

## Result

The exact V9 command ran once on physical GPU 4. Source validation,
virtualenv/Ninja, native-sampler, in-process EngineCore, GPU identity/idle,
shared-KV, weight-alias, KV-floor, compilation, and graph-capture checks all
passed. The first OFF boot then completed every registered R4 cell:

```text
failed boot                              p4-b0-b1-p1-off
completed physical boots                                0
complete R4 captures                                     8
complete events                                     16,384
complete committed tokens                          131,072
incomplete captures                                       0
empty R5 placeholders                                     1
adapted rounds                                            0
score emitted                                         false
```

The first R5 cell queued its eight-request, 112,304-token prompt microbatch.
The compiled Qwen3 MLP failed while allocating the bfloat16
`(112304, 12288)` SiLU/multiply output:

```text
requested allocation                              2.57 GiB
allocator-reported free memory                     1.45 GiB
estimated CUDA graph pool                          0.50 GiB
actual CUDA graph pool                             4.90 GiB
shared-KV capacity                               22,090 blocks
registered launch floor                          21,682 blocks
```

The immutable record is
`data/p4/run_b0_value_screen_v8/failure.json`. It binds all eight complete
capture hashes, the empty R5 placeholder, preparation, first plan, first boot
spec, and V9 authorization. No `adapted_rounds.jsonl` or `score.json` exists.

## Root cause

The resource gate checked the number of target-owned KV blocks after graph
capture, but it did not reserve prompt-length-scaled full-prefill activations.
The old conservative bound also allowed 2 GiB for graph/workspace memory,
while this launch reported a 4.90 GiB actual graph pool against a 0.50 GiB
estimate.

At the failed activation, the compiled MLP has the 24,576-wide gate/up output
and 12,288-wide SiLU/multiply output live together with three 4,096-wide rows.
For R5 this lower-bounds the live activation set at 10.282 GiB; R5cot raises it
to 10.337 GiB. The allocator reached the R5 SiLU allocation with insufficient
headroom.

This is not a private-draft-KV failure. The attempt retained one target-owned,
36-layer bfloat16 KV pool, zero private draft KV pools, and 291 target/draft
weight aliases. It is also not evidence that the R4 action is valuable: the
frozen scorer requires all 432 captures and never ran.

## Fail-closed disposition

V9 authorized one attempt, and that attempt is consumed. No fallback GPU,
retry, partial resume, adaptation, or scoring occurred. No runner or GPU
compute process remained after failure. All V8 output files are immutable.

The eight complete captures are retained only as execution evidence. They
cannot be combined with a later run because the matrix contract requires a
fresh, complete, same-authorization output and forbids partial resume.

## Required repair

Do not issue V10 with the same `114688 / 0.96` full-microbatch geometry. The
separate transient bound in
`results_p4_b0_full_prefill_transient_bound.md` rejects it even under
optimistic report rounding.

The preferred next design is bounded chunked prefill plus a capture-cohort
decode barrier: queue the full frozen microbatch, allow prompt chunks to
finish without releasing early requests into measured decode, then release
the whole cohort into its first measured pure-decode step. That design needs
CPU state-machine tests and a separately authorized, non-scored one-boot GPU
memory/ingress probe before any V10 review.

## Artifacts

- V9 authorization:
  `data/p4/p4_b0_run_authorization_v9.json`
  (`1542763ea55e6cda9aaf29fda357bc3818a508eecd34d22d869c22aa13ea1ee4`);
- failure record:
  `data/p4/run_b0_value_screen_v8/failure.json`
  (`605e10c42bef250269c346141599dd0af08613f2bd3b8d675ee2fcff187b9986`);
- preparation:
  `data/p4/run_b0_value_screen_v8/preparation.json`
  (`4eb2f700c170b2d2ddfca1409e2948d50baee16960e3a69f0abefcf7c92e5ee1`);
- first plan:
  `data/p4/run_b0_value_screen_v8/plans/p4-b0-b1-p1-off.json`
  (`7ce7fe4a83bab45d5893c92500a3eba94fc0c947a059d324ff34093f1529b2da`);
  and
- first boot spec:
  `data/p4/run_b0_value_screen_v8/boot_specs/p4-b0-b1-p1-off.json`
  (`e22a1729d9aac8057f1d4f9e2dccf93dd3802584ebb9dd679a5d81f27e4ea6cf`).
