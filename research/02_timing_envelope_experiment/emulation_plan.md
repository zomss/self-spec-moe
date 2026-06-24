# Emulation Plan: Forcing Multi-Node-Like Communication

Date: 2026-06-23

## Motivation

We do not currently have a true multi-node EP setup with DeepEP/DBO available.
To continue Phase 02, emulate the missing condition:

```text
low-batch decode where exposed EP all-to-all latency is large
```

The goal is not to claim real multi-node performance. The goal is to quantify
how much exposed communication must exist for Self-MoE-spec to matter.

## Emulation Levels

| Level | Method | Status |
| --- | --- | --- |
| L0 | Analytical delay injection from measured local timings | Implemented via `emulate_comm_delay.py` |
| L1 | Backend substitution: slower allgather/naive backend | Partially done with DP2 allgather sanity |
| L2 | vLLM all-to-all delay injection patch | Future; closer to systems behavior |
| L3 | Trace replay from real multi-node DeepEP logs | Future; strongest emulation if traces are available |

## Delay-Injection Model

The first emulation uses:

```text
T_base = T_compute_local + T_comm_delay
T_draft_local = T_compute_local
T_verify = T_verify_compute + T_comm_delay
```

DBO-like overlap is modeled as:

```text
T_base_dbo = T_compute_local + (1 - h) * T_comm_delay
T_verify_dbo = T_verify_compute + (1 - h) * T_comm_delay
```

where `h` is the hidden communication fraction, default `0.5`.

## Sweep

Use:

```text
comm_delay_ms = 0.5, 1, 2, 4, 8
k = 1, 2, 4, 8
B = 1, 2, 4, 8
beta = 0.6, 0.7, 0.8, 0.9, 1.0
```

The most important output is the threshold communication delay where:

```text
beta_min@1.3x <= 0.9
```

## Commands

Generate emulated timings:

```bash
.venv/bin/python research/02_timing_envelope_experiment/emulate_comm_delay.py \
  --compute-local-ms 4 \
  --verify-compute-ratio 1.0 \
  --comm-delays-ms 0.5,1,2,4,8 \
  --batch-sizes 1,2,4,8 \
  --draft-lengths 1,2,4,8 \
  --output-csv research/02_timing_envelope_experiment/data/timings_comm_delay.csv
```

Run the envelope:

```bash
.venv/bin/python research/01_communication_advantage/speedup_envelope.py \
  research/02_timing_envelope_experiment/data/timings_comm_delay.csv \
  --target-speedup 1.3 --format csv \
  > research/02_timing_envelope_experiment/data/envelope_comm_delay_1p3.csv
```

Run acceptance sweep:

```bash
.venv/bin/python research/01_communication_advantage/speedup_envelope.py \
  research/02_timing_envelope_experiment/data/timings_comm_delay.csv \
  --acceptance-rates 0.6,0.7,0.8,0.9,1.0 --format csv \
  > research/02_timing_envelope_experiment/data/acceptance_sweep_comm_delay.csv
```

## Interpretation Rule

If emulation shows that the method needs an unrealistic artificial delay to
reach useful speedup, the idea depends too heavily on extreme multi-node
communication. If useful speedup appears at plausible delay values, the next
step is measuring local-routing acceptance.
