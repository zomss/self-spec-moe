# Phase 97 P4 — conservative boot-static w512 B0 resource bound

Status: **PASS for pre-engineering resource readiness. The measured minimal
B0 base has a conservative lower bound of 21,682 shared-KV blocks after all
registered debits, versus 21,000 required. The remaining floor is 682 blocks,
or 10,912 tokens. This artifact does not authorize GPU measurement, P4a,
action admission, or a performance claim.**

Date: 2026-08-09.

## Decision

The sole remaining readiness item,
`conservative_resource_bound_missing`, is cleared. The readiness chain now
has no missing evidence artifacts, but authority remains a separate decision:

```text
same_event_recorder_unwired             cleared previously
w512_mask_equivalence_unproven          cleared previously
conservative_resource_bound_missing     cleared here
GPU measurement authorization           false
P4a engineering authorization           false
```

The next artifact is a separate run-ready/authorization package. This bound
contains no GPU command and cannot self-authorize one.

## Capacity arithmetic

The source measurement is the exact minimal-B0 K4 boot, not the older w512
projection:

```text
measured minimal-B0 blocks                         24,529
workload hard live-KV tokens                      320,000
workload token safety margin                       16,000
required blocks = ceil(336,000 / 16)               21,000
apparent pre-bound headroom                         3,529
```

Four independent reserves are converted to whole KV blocks with
`ceil(reserved_bytes / 2,359,296)` before summation. This intentionally loses
any cross-category sub-block packing credit.

| reserve | bytes | block debit | basis |
| --- | ---: | ---: | --- |
| persistent w512 buffers | 1,048,576 | 1 | 676,096-byte static upper bound, rounded to 1 MiB |
| graph capture and temporary workspace | 2,147,483,648 | 911 | 3x the 0.54 GiB baseline full graph pool, including 0.01 GiB report allowance, rounded to 2 GiB |
| recorder and runtime metadata | 268,435,456 | 114 | fixed 256 MiB HBM contingency; no credit for records normally residing on CPU |
| HBM and fragmentation safety | 4,294,967,296 | 1,821 | greater of 5% usable HBM and 4 GiB |
| **total** | **6,711,934,976** | **2,847** | independent block rounding retained |

The resulting lower bound is:

```text
24,529 - 2,847 = 21,682 blocks
21,682 - 21,000 =    682 blocks
682 * 16          = 10,912 tokens
```

The 4 GiB HBM reserve is additional to the workload's 16,000-token safety
margin. The former covers allocator, fragmentation, and unmodeled HBM risk;
the latter covers live-KV demand.

## Persistent-buffer bound

The registered run geometry is fixed at 32 sequences, 20,480 model tokens,
16-token blocks, and therefore 1,280 block-table columns. Five w512-specific
buffers are charged at an eight-byte-per-element upper bound even where the
live dtype is narrower:

| buffer | elements | upper-bound bytes |
| --- | ---: | ---: |
| window block table | 40,960 | 327,680 |
| window sequence lengths | 32 | 256 |
| column arange | 1,280 | 10,240 |
| source-column indices | 40,960 | 327,680 |
| sink-column mask | 1,280 | 10,240 |
| **derived total** |  | **676,096** |

The bound rounds this total to 1 MiB. Per-step intermediates are not credited
here; they are covered by the separate 2 GiB graph/workspace reserve. The
registered alternate scratchpad FULLCG path remains disabled.

## Evidence grade and fail-closed boundary

This is `measured_base_plus_conservative_delta_bound` with relation
`conservative_candidate_upper_bound`: it upper-bounds incremental memory and
thereby produces a lower bound on remaining KV blocks. It is deliberately not
`measured_exact` evidence for the w512 candidate:

- `candidate_realization_match=false`;
- `exact_post_capture_measurement=false`;
- the older `static_proxy_projection` remains an
  `optimistic_proxy_ceiling`; and
- exact post-capture capacity remains mandatory before action admission.

The bound is invalid if source geometry drifts, a reserve is exceeded, the
single target-owned KV pool or alias invariant fails, an unregistered graph or
scratchpad path is enabled, or post-capture capacity falls below 21,000
blocks. Any such event stops the screen rather than borrowing from the 682
block floor.

## Artifacts and validation

```text
schemas/p4_b0_resource_bound.schema.json
data/p4/p4_b0_conservative_resource_bound.json
data/p4/p4_b0_conservative_resource_bound_validation.json
scripts/validate_p4_b0_resource_bound.py
tests/test_p4_b0_resource_bound.py
```

Validate with:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research/97_composition_runtime/scripts/validate_p4_b0_resource_bound.py \
  --bound \
  research/97_composition_runtime/data/p4/p4_b0_conservative_resource_bound.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  research/97_composition_runtime/tests/test_p4_b0_resource_bound.py -q
```

The focused suite passes 18 tests. It covers source-hash and geometry drift,
every reserve, independent block rounding, result arithmetic, fail-closed
invalidators, exact-measurement overclaim, and authority inflation. The full
Phase 97 suite passes 211 tests plus 21 subtests. Ruff 0.14.0 check and
format-check pass the two new Python files. No GPU command was run.

## Next step

Create a separate run-ready/authorization package for the matched boot-static
OFF/K4/w512 acceptance screen. That package must pin the exact boot command,
GPU assignment, source hashes, resource stop conditions, and the narrow w512
capture enablement. Only an explicit decision in that package may authorize
the GPU screen. Runtime w512 switching remains outside this step, and even a
successful screen cannot admit the action without exact post-capture resource
evidence.
