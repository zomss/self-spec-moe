# Results: Artificial Communication-Delay Emulation

Date: 2026-06-23

## Status

This is an analytical emulation for multi-node-like exposed communication. It
does not require actual multi-node hardware.

## Model

The emulation assumes:

```text
T_compute_local = 4 ms
T_base = T_compute_local + T_comm_delay
T_draft_local = T_compute_local
T_verify = T_compute_local + T_comm_delay
```

DBO-like overlap is modeled with hide fraction `h = 0.5`:

```text
T_base_dbo = T_compute_local + 0.5 * T_comm_delay
T_verify_dbo = T_compute_local + 0.5 * T_comm_delay
```

This means the emulation asks:

> How much extra exposed communication delay is required for Self-MoE-spec to be
> useful?

## Generated Artifacts

| Artifact | Purpose |
| --- | --- |
| `data/timings_comm_delay_compute4.csv` | Emulated timing input |
| `data/envelope_comm_delay_compute4_break_even.csv` | Break-even envelope |
| `data/envelope_comm_delay_compute4_target_1p3.csv` | 1.3x target envelope |
| `data/acceptance_sweep_comm_delay_compute4.csv` | Acceptance-rate sweep |

## Key Result for `B=1, k=4`

| Delay | Stack | `S_max` | Speedup at beta=0.9 | `beta_min@1.3x` |
| ---: | --- | ---: | ---: | ---: |
| 0.5 ms | no DBO | 1.098 | 0.899 | NA |
| 0.5 ms | DBO hides 50% | 1.049 | 0.860 | NA |
| 1 ms | no DBO | 1.191 | 0.975 | NA |
| 1 ms | DBO hides 50% | 1.098 | 0.899 | NA |
| 2 ms | no DBO | 1.364 | 1.117 | 0.976 |
| 2 ms | DBO hides 50% | 1.191 | 0.975 | NA |
| 4 ms | no DBO | 1.667 | 1.365 | 0.875 |
| 4 ms | DBO hides 50% | 1.364 | 1.117 | 0.976 |
| 8 ms | no DBO | 2.143 | 1.755 | 0.747 |
| 8 ms | DBO hides 50% | 1.667 | 1.365 | 0.875 |

## Interpretation

For `T_compute_local = 4 ms` and `k = 4`:

- If exposed communication is only 0.5-1 ms, the method is not useful.
- At 2 ms exposed delay, a 1.3x gain is theoretically possible without DBO, but
  requires nearly perfect acceptance (`beta ~= 0.98`).
- At 4 ms exposed delay, the method becomes plausible: without DBO, beta 0.9
  gives about 1.37x, and 1.3x needs beta about 0.88.
- If DBO hides 50% of communication, the effective threshold shifts upward:
  the DBO case with 4 ms delay looks like the no-DBO case with 2 ms delay.
- With 8 ms exposed delay, the method has strong room even with DBO-style
  overlap.

## Conclusion

Yes, we can emulate multi-node behavior locally by injecting communication
delay analytically. The useful regime begins when exposed communication is
roughly comparable to local compute:

```text
T_comm_delay / T_compute_local >= 0.5
```

For the default `k = 4`, a strong result with beta around 0.9 needs exposed
communication close to local compute time, especially if DBO already hides part
of the communication.

This reinforces the two required demonstrations:

1. real or emulated exposed communication is high enough,
2. local-only draft acceptance is at least around 0.9 for the target regime.
