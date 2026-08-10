# Phase 97 P4 — chunked-prefill cohort probe attempt V1

Date: 2026-08-09

Status: **STOPPED before child launch or GPU model execution; the V1 probe
authorization is consumed and cannot be retried**

## Outcome

The exact registered command was invoked once. Native-sampler, in-process
EngineCore, and physical-GPU-4 identity/idle preflights passed. The parent then
created the reviewed output directory and exited with code 1 before emitting
`preparation.json`, opening a child log, or spawning the GPU child.

No model was loaded, no physical boot completed, no cohort or target event was
recorded, and no probe result or score exists. Post-failure GPU 4 inspection
reported the registered UUID, zero MiB in use, and zero compute processes.

## Root cause

The authorization validator compared `output_dir.resolve()` with the reviewed
path, but did not return or propagate that normalized value. The registered
CLI intentionally supplied the repository-relative path. `execute_parent`
therefore received a relative `Path`, created it successfully, and then tried:

```python
output_dir.relative_to(REPO_ROOT)
```

Because `REPO_ROOT` is absolute, `pathlib` raised `ValueError`. The failure was
after create-new directory consumption but before `subprocess.run`, so this is
a parent path-normalization/coverage failure—not a GPU, driver, model,
resource-fit, shared-KV, or cohort-barrier result.

The CPU execution test missed this because it called `execute_parent`
directly with an absolute `TemporaryDirectory` path instead of traversing
`main()` with the exact registered relative argv.

## Fail-closed disposition

The V1 package authorized one invocation. That invocation and its create-new
output path are consumed. No retry, resume, absolute-path workaround, fallback
GPU, scoring, or V10 issuance occurred. The machine-readable record is
`data/p4/run_b0_chunked_prefill_probe_v1/failure.json`
(`b377820fe89ebebfffd278fb3f2f9b7150fc0b6f3de41c9599d34e9767d3032a`);
the directory must be preserved without overwrite.

## Handoff

The parent/child path repair and exact registered-relative-argv regression now
pass. A fresh source-bound non-scored V2 probe package with a create-only V2
output is recorded in
`results_p4_b0_chunked_prefill_probe_authorization_v2.md`. V1 remains consumed
and immutable; it was not retried or reused.
