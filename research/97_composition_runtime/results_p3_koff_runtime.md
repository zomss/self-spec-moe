# Phase 97 P3 result — minimal B0 K/OFF passive runtime

Status: **PASS for the CPU/synthetic contract. The exact measured B0 is bound
to one `OFF` action and one target-matching `K=4` action, and the passive trace
replay passes all accounting, interval, transition, resource, version, and
shared-KV checks. This is not live-engine or performance evidence.**

Date: 2026-08-09.

Follow-up: the minimal-B0 contract has since been wired into the live engine
and CPU-validated. See `results_p3_live_engine_wiring.md`. The evidence in
this document remains the original synthetic replay and is not retroactively
treated as live GPU evidence.

## Question

Can the smallest resource-admitted Phase 97 boot expose safe K/OFF runtime
recourse, consume passive target-step records, and fail closed without adding
window, layer-skip, quantized-weight, private-KV, or new GPU work?

## Frozen P3 boundary

P2 provides exact evidence only for the measured minimal B0 graph pool. P3
therefore registers exactly:

| action | target graph | draft graph | shared KV |
| --- | --- | --- | --- |
| `off` | `target-k1` | none | target-owned pool remains allocated |
| `target-matching-k4` | `target-k5` | `draft-target-matching-k1` | same binding, pool, and true slot mapping |

`K=2`, other K values, window actions, skip actions, B1, and quantized draft
weights are not inferred from this evidence. Adding any graph changes the
candidate realization and requires a new exact post-capture resource record.

The strict HBM binding closes as:

```text
76,968,728,985 usable bytes
- 24,529 shared-KV blocks * 2,359,296 bytes/block
= 19,097,557,401 measured non-KV bytes
```

That residual equals the boot manifest's declared fixed terms exactly. The
synthetic workload uses 21,000 of the 24,529 admitted shared-KV blocks.

## Artifacts

| artifact | purpose |
| --- | --- |
| `data/p3/boot_b0_minimal_k4.json` | strict B0 boot and exact HBM/graph binding |
| `data/p3/actions_b0_minimal_k4.json` | two-action K4/OFF registry |
| `data/p3/runtime_snapshot_b0_synthetic.json` | one shared-target-KV binding snapshot |
| `schemas/passive_koff_trace.schema.json` | closed passive trace interface |
| `data/p3/passive_koff_trace_synthetic.json` | six-segment, non-scored transition trace |
| `scripts/replay_koff_trace.py` | resource/package validator and policy replay |
| `data/p3/replay_koff_trace_result.json` | exact checked-in replay output |
| `tests/test_koff_runtime.py` | positive and fail-closed P3 tests |

## Replay result

The trace deliberately covers normal selection and both required fallback
paths:

| start step | H | event | selected | policy result |
| ---: | ---: | --- | --- | --- |
| 0 | 32 | initial exploit | K4 | K4 interval dominates `OFF` |
| 32 | 16 | K4 target graph unavailable | `OFF` | `requested_ineligible` |
| 48 | 144 | forced dwell | `OFF` | policy override remains safe |
| 192 | 16 | stale K4 exploit request | `OFF` | `stale_evidence` |
| 208 | 16 | stale K4 probe | K4 | probe may leave the exploit tie-set |
| 224 | 32 | post-probe exploit | K4 | endpoint evidence is fresh |

The selector uses the Phase 96 interval-dominance rule:

```text
S = [tau_lo / q_hi, tau_hi / q_lo]
T_epsilon = {a: S_hi(a) >= (1 - epsilon_sel) * max_b S_lo(b)}
OFF = [1, 1]
```

The replay reports:

| action | segments | H | D | A | E | U | tau_eff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `off` | 3 | 176 | 0 | 0 | 176 | 176 | 1.0 |
| `target-matching-k4` | 3 | 80 | 80 | 160 | 240 | 0 | 3.0 |

All segments close `E + C = A + H`; every engine-step active-request sum
equals its segment's `H`; accepted tokens never exceed `K * D`; all 16 engine
steps use the selected action's graph descriptors; and no preemption or
recomputation is present. Probe duty is exactly `16 / 256 = 6.25%`, equal to
the declared synthetic ceiling.

Freshness is provenance-checked. Target step 0 is the explicitly trusted
synthetic seed. A later observation time becomes legal only at the endpoint
of a segment that actually selected the speculative action with `D > 0`.
Consequently, the stale exploit fails closed, the probe is allowed, and only
the probe endpoint at step 224 can refresh the final exploit.

## Fail-closed coverage

The 18 P3 tests reject or verify:

1. graph-pool, HBM-term, and `max_k` divergence from exact P2 evidence;
2. incorrect target/draft graph roles;
3. shared-KV binding, pool, or true-slot-mapping changes;
4. target-matching draft/target version divergence;
5. counter-closure, accepted-token, and OFF-accounting violations;
6. engine-step/H disagreement and action/graph disagreement;
7. a switch inside a target step or loss of the `OFF` graph;
8. graph-unavailable and stale-evidence fallback;
9. stale-probe allowance and post-probe freshness;
10. fabricated observation times and inverted intervals;
11. probe-duty overflow; and
12. interval-dominance equality at the epsilon boundary.

Together with P1/P2, all 50 Phase 97 CPU tests pass.

## Commands and result

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research/97_composition_runtime/scripts/replay_koff_trace.py \
  --environment research/97_composition_runtime/data/preflight/environment_qwen3_8b_h100_tp1.json \
  --workload research/97_composition_runtime/data/preflight/workload_rl_capacity_v1.json \
  --candidate research/97_composition_runtime/data/preflight/candidate_b0_measured.json \
  --boot research/97_composition_runtime/data/p3/boot_b0_minimal_k4.json \
  --actions research/97_composition_runtime/data/p3/actions_b0_minimal_k4.json \
  --runtime research/97_composition_runtime/data/p3/runtime_snapshot_b0_synthetic.json \
  --trace research/97_composition_runtime/data/p3/passive_koff_trace_synthetic.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s research/97_composition_runtime/tests \
  -p 'test_*.py' -v
```

Result:

```text
replay status: pass
total target steps: 256
total engine steps: 16
probe target-step fraction: 0.0625

Ran 50 tests
OK
```

No GPU process, model boot, profiler, or scored measurement was run.

## Limitations

- Trace counters, decode times, and intervals are synthetic and
  `scored=false`; they establish plumbing, not speedup.
- The replayer is phase-local and is not yet fed by live vLLM scheduler and
  proposer records.
- Interval construction and calibration remain Phase 96 responsibilities.
- Transition-cost fields remain synthetic zero values; no latency claim is
  made.
- P3 proves neither token correctness nor live tensor aliasing during a
  switch. P1's CPU object checks remain the available alias evidence.

## Decision

P3 closes at its registered CPU/synthetic boundary. P4 should begin by wiring
this exact action id, target-step boundary, graph descriptor, and passive
counter interface to the existing K/OFF execution path before admitting a
window action. A window action may enter only with action-specific acceptance
evidence and either the unchanged exact graph/resource realization or a new
post-capture resource candidate.
