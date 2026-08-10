# Phase 97 P4 — B0 value-screen attempt V7

Status: **STOPPED at the parent authorization-path gate before GPU preflight,
output creation, capture, or scoring. V8 is consumed and cannot be retried.**

Date: 2026-08-09.

## Result

The exact registered V8 invocation exited with code 2:

```text
P4 runner refused: execution must use a reviewed authorization path
```

```text
GPU preflight started                                  false
GPU model executed                                     false
completed boots                                            0
complete / incomplete captures                          0 / 0
adapted rounds                                              0
score emitted                                           false
runner output created                                   false
```

The immutable machine record is
`data/p4/run_b0_value_screen_v7/failure.json`. That directory was created only
after the stopped attempt to preserve the failure record; the runner created
no measurement output.

## Root cause

The package resolver and execution-authority validator both admitted V8. The
final parent `execute_run` pathname dispatcher and its child-mode counterpart
still admitted only the V6 and V7 authorization paths. The V8 non-execution
validator built the matrix directly and therefore did not traverse either
launch-only branch, allowing a false-positive run-ready result.

This is not a GPU, driver, sampler, EngineCore, shared-KV, resource-capacity,
request-ID, decode-work, or model failure.

## Fail-closed disposition

V8 authorized one attempt. That attempt is consumed. No fallback GPU, retry,
partial resume, adaptation, or scoring occurred. No runner or GPU compute
process remained after the refusal.

## Repair

V9 uses one exact authorization-to-output resolver from both parent and child
execution paths. Regression tests traverse both paths with GPU/model work
mocked and prove rejection of unknown authorizations and cross-paired output
paths. The V9 validator also traverses the shared resolver before declaring
the package launchable.

The V9 authorization and evidence are recorded in
`results_p4_b0_run_authorization_v9.md`.
