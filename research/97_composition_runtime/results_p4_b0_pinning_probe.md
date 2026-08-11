# P4 B0 CPU-pinning probe — result

Date: 2026-08-11

Verdict: **CPU dispatch contention confirmed as the cause of the V13
certification failure.** All four cells that exceeded 5% within-cell spread
improved, by factors of 9× to 251×, and rounds below the 95% floor fell from
six to one. This probe is non-scored and grants no re-screen, scoring, or
admission authority.

## Result

| cell | V13 spread | probe spread | factor |
| --- | --- | --- | --- |
| `target-matching-k4/R6/s0` | 24.64% | **0.161%** | 153× |
| `target-matching-k4/R8/s1` | 18.07% | **0.205%** | 88× |
| `target-matching-w512-masked-k4/R1/s0` | 13.00% | **1.371%** | 9× |
| `target-matching-w512-masked-k4/R5cot/s1` | 6.87% | **0.027%** | 251× |

- rounds below the 95% floor: **6 → 1**
- all four cells now sit far inside the 2% cross-boot certification bar
- the 20 cells that already passed hold a median spread of 0.264%
- 96 captures across block 1's `k4` and `w512` boots, both on the reserved lane

## What was changed

Nothing about the GPUs, the lane assignment, or the measurement contract. Only
CPU reservation:

- lanes narrowed from `0-95`/`96-191` (all 192 cores, zero headroom) to
  `0-15`/`96-111`;
- the Lean server, ours, pinned to `32-79`, including the `lake` parents so
  newly forked repls inherit; and
- 112 cores left free.

The Lean server was deliberately left **running** so the mitigation faced the
real disturbance rather than an idle box.

## What this does not show

- **One cell is over 5% that was not before**: `w512/R5/s0` at 9.8%. The noise
  is reduced, not eliminated. This is consistent with the load still unpinned
  on this box — `clangd` at ~50% and `zed-remote-server`, which belong to
  another user and are not ours to pin. A re-screen can still fail
  certification if an episode lands in the wrong cell.
- The probe covered block 1's two **draft** boots only. `off` boots and blocks
  2 and 3 were not exercised, though `off` never showed the effect.
- Four cells at four rounds each is a narrow base. The result is strong
  evidence for the mechanism on the cells that failed, not proof the box is
  now quiet.

## Operator-induced contamination, and how it resolved

During `k4/R8/s1` rounds r1–r4 an unpinned single-threaded JSON parse run by
the operator (affinity `0-191`, overlapping the probe's reserved `0-15`)
consumed one core for ~400 s — the same class of disturbance the probe was
testing for. This was recorded in the run's `operator_note.json` **before** the
result was known, with the reading rule fixed in advance: clean counts as
stronger evidence, noisy counts as unattributable and requires a rerun.

The cell returned 0.205%. It stayed clean while an extra CPU hog sat on its
reserved cores, so it counts, and no rerun is needed. The operational rule
stands regardless: no CPU-heavy work on this box while a measurement is live.

## Evidence the new hygiene records produced

Both additions from the same change paid for themselves on their first run.

`observations/<boot_id>.json` confirms what actually executed, rather than what
was declared: `chain_runtime_mode=PIECEWISE` on both boots (no silent fallback
to `NONE`), observed device UUID equal to the declared lane GPU, and 16 cores
`0-15`. The V13 screen could report none of this — its captures declared GPU 4
while running on GPUs 0 and 1.

The per-descriptor PIECEWISE log replaced a one-shot line and immediately
showed the chain's padding across the batch sweep:

```text
batch_size=1  input_batch_size=5     (5x)
batch_size=10 input_batch_size=10    (none)
batch_size=11..16 input_batch_size=20
```

This is the `(1 + K) = 5` capture-size rounding. R1 runs at batch 1 and is
46.5% of the screen's wall clock, so it pads 5×. At batch 1–8 on a dense 8B
model the draft forward is weight-bandwidth-bound, so the extra rows are close
to free — but this is now measured rather than assumed, and it is visible for
every regime instead of only the first dispatch.

## Boundary

Non-scored. Grants no authority. A re-screen remains a separate decision
requiring its own source-bound authorization, and the residual `w512/R5/s0`
excursion means certification is likely but not assured.

## Artifacts

- `data/p4/p4_b0_pinning_probe_authorization_v2.json`
- `data/p4/run_b0_pinning_probe_v2/probe_result.json`
- `data/p4/run_b0_pinning_probe_v2/operator_note.json`
- `data/p4/run_b0_pinning_probe_v2/observations/*.json`
- `data/p4/run_b0_pinning_probe_v1/failure.json` (consumed 48-cell refusal)
- `scripts/run_p4_b0_pinning_probe.py`
- `tests/test_p4_b0_pinning_probe.py`
