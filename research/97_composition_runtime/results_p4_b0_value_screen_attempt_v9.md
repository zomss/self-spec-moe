# Phase 97 P4 — B0 value-screen attempt V9

Status: **STOPPED at the first K4/R8 pure-prefill dispatch after 72 complete
captures. V10 is consumed. The partial output is unscored and must not be
retried, resumed, overwritten, or reused.**

Date: 2026-08-10.

## Result

The exact V10 command ran once on physical GPU 4 with bounded chunked prefill
at `8192 / 8160 / 0.90`. The OFF boot completed all 48 cells. The K4 boot
completed all 24 R4, R5, and R5cot cells, then stopped on the first R8 cell:

```text
complete captures                                      72 / 432
OFF captures                                             48 / 48
K4 captures before R8                                    24 / 48
complete target events                                   454,556
complete committed tokens                              2,949,120
empty failed placeholders                                      1
adapted rounds                                                  0
score emitted                                                false
```

The last complete event was K4/R5cot seed 1 round 4 at engine step 27,699.
Step 27,700 admitted the 16 frozen R8 seed-0 prompts, scheduled all 2,100
prompt tokens, armed the exact capture cohort, and selected K4. The worker
then failed closed with:

```text
KOffRuntimeError: target-matching-k4 non-decode dispatch requires a positive
draft step-0 query width and pure-prefill cohort arming
```

Shutdown correctly rejected the incomplete R8 cohort. No adapted round or
score was written. GPU 4 and all runner processes were released.

## Preservation

`capture_manifest.json` binds every one of the 72 complete captures, the
empty R8 placeholder, all nine plans, all nine boot specs, preparation, and
the V10 authorization. `failure.json` records the observed exception chain,
the exact failed scheduler boundary, and the consumed/no-score disposition.

The original terminal transcript was not retained, so that fact is explicit
in the failure record. The independently hash-bound capture boundary ends at
engine step 27,699 and the separate GPU diagnosis reproduces the exception.

## Diagnosis handoff

The source-bound GPU-0/GPU-1 comparison in
`results_p4_b0_r5cot_r8_diagnosis.md` proves this is an R8 variable-width
prefill evidence bug, not a required R5cot-history dependency. That diagnosis
does not repair V10 or authorize a retry.

## Artifacts

- V10 authorization:
  `data/p4/p4_b0_run_authorization_v10.json`
  (`d772e983718cd8e4ccb16c506da1a7a0ef15908221b59c0ea98b014b5d37bc65`);
- capture manifest:
  `data/p4/run_b0_value_screen_v9/capture_manifest.json`
  (`4b3cb6d8420483b875bc2bfafccd804681b9b847447e448f135533549b77225e`);
- failure record:
  `data/p4/run_b0_value_screen_v9/failure.json`
  (`78b21f7b6110a2e383f6ce7d6ba2e26a121a30722ed89dddac61f1afa93a852f`);
  and
- immutable output: `data/p4/run_b0_value_screen_v9`.
