# Phase 97 P2 result — CPU boot-resource preflight

Status: **complete at the CPU/evidence layer. The measured minimal B0 is
admitted for the non-scored engineering envelope. B1 is rejected fail-closed
until co-resident weight, graph/workspace, and RL refresh memory are measured
and included in an exact post-capture capacity record. No GPU run or engine
change was made.**

## Inputs

- Fixed proxy environment: Qwen3-8B, TP1/PP1, one H100 80GB, target BF16 KV,
  16-token blocks, and exactly one target-owned shared KV pool.
- Workload: non-scored RL engineering envelope with 320,000 hard live KV
  tokens and a 16,000-token capacity margin.
- B0 evidence: the historical target-matching/shared-KV boot, measured after
  its minimal K4 graph capture.
- B1 evidence: the historical W4A8/shared-KV static boot. This path did not
  co-reside with the target-matching draft path and is an optimistic capacity
  ceiling, not exact B1 evidence.

The workload is an engineering assumption, not a measured production trace.
It cannot authorize scored work.

## Result

At 16 tokens per block, the resource envelope requires:

```text
ceil((320,000 hard tokens + 16,000 safety tokens) / 16) = 21,000 blocks
```

| candidate | evidence | available blocks | headroom before unknowns | decision |
| --- | --- | ---: | ---: | --- |
| minimal B0 | exact post-capture measurement | 24,529 | +3,529 | admit |
| desired B1 | W4A8-only optimistic ceiling | 21,928 | +928 | reject |

B1 fails for five independent reasons:

1. its capacity evidence is not the requested co-resident realization;
2. co-resident weight HBM is unknown;
3. the combined graph/workspace HBM is unknown;
4. RL weight-refresh peak HBM is unknown; and
5. RL refresh pinned-host memory is unknown.

The 928-block proxy margin is therefore descriptive only. It must not be used
to admit B1, and the unknown quantities remain JSON `null`, never proxy zeros.

## Contract implemented

- `environment.schema.json` fixes target quantization, KV specification,
  topology, HBM budgets, software realization, and the shared-KV-only rule.
- `workload.schema.json` requires concurrency, length, live-KV, prefix-cache,
  RL synchronization, objective, preemption, and routing assumptions.
- `boot_candidate.schema.json` separates exact deployable evidence from a
  static projection. The existing strict boot manifest remains integer-only.
- `plan_boot_class.py` validates all inputs, replays the frozen live-KV trace,
  checks block/spec identity, and rejects unknown or unreserved B1 resources.

An exact future B1 record must bind the capacity measurement to both resident
weight paths and the complete graph set. For RL, its refresh HBM must be
reserved by that measurement and its pinned-host use must fit the declared
budget. An ordinary-inference deployment does not require recurring refresh
workspace after initialization, but it still requires exact co-resident
weight, graph, and post-capture capacity evidence.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s research/97_composition_runtime/tests -p 'test_*.py' -v
```

Result: **32/32 pass**: 13 resource-preflight tests plus the existing 19
shared-KV invariant tests. The negative cases cover unknown B1 quantities,
unreserved graph memory, pinned-host overflow, capacity shortfall, environment
and KV-block mismatch, invalid workload traces, and assumed/scored mixing.

The checked-in planner output is `data/preflight/plan_rl_capacity_v1.json`.

## Decision and boundary

- Carry minimal B0 into P3's CPU registry/K/OFF integration.
- Do not admit B1 or begin B1 runtime profiling.
- Any additional B0 window/skip/K graph changes the resident pool and requires
  a new exact capacity record before admission.
- After B1 co-residency and refresh exist in P6, repeat post-capture capacity
  measurement for the exact resident objects and rerun this planner.
- Replace the assumed workload with a frozen measured trace before scored or
  service-value work.
