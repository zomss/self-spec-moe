# P4 B0 GPU contention probe authorization V1

Date: 2026-08-10

Decision: **APPROVE one non-scored GPU-0/GPU-1 contention probe only.** This
does not authorize the 432-capture value screen, scoring, P4a engineering, or
action admission.

The probe compares the same registered `target-matching-k4`, R8, content-seed
0 workload under four engine boots:

1. GPU 0 alone;
2. GPU 1 alone;
3. GPU 0 while GPU 1 runs concurrently; and
4. GPU 1 while GPU 0 runs concurrently.

Each boot records the four registered R8 rounds with the same-event `S_dec`
currency. GPU 0 and GPU 1 use disjoint rendezvous ranges, CPU affinities, and
compile-cache roots. The serial boots run first; the concurrent pair runs only
after both serial boots close.

The gate passes only when:

- every episode retains at least two rounds under the frozen 95% rule;
- concurrent slowdown is at most 1% on each GPU;
- the two per-GPU contention ratios disagree by at most 1%; and
- serial and concurrent cross-GPU rates each disagree by at most 2%.

Any execution or threshold failure preserves the attempt without retry,
resume, score input, or dual-GPU authority. A pass permits preparation of a
separate source-bound block-parallel authorization; it does not itself launch
the value screen.

Artifacts:

- `data/p4/p4_b0_gpu_contention_probe_authorization_v1.json`
- `schemas/p4_b0_gpu_contention_probe_authorization.schema.json`
- `scripts/run_p4_b0_gpu_contention_probe.py`
- `scripts/validate_p4_b0_gpu_contention_probe_authorization.py`
- `tests/test_p4_b0_gpu_contention_probe.py`

## Observed execution

The exact V1 command was consumed once. Both serial baselines completed, but
the parent stopped before launching either concurrent child because port
`47296` was already an established endpoint owned by another process. The
registered `47000`--`47399` ranges overlap the host ephemeral range
`32768`--`60999`; this is a launcher preflight failure, not a GPU or self-spec
runtime failure.

V1 remains preserved and unscored. Its serial results are not reused. A fresh
package must rerun all four boots with non-ephemeral port ranges; the checked
candidate interval `20000`--`20399` was free at diagnosis time.

See `data/p4/run_b0_gpu_contention_probe_v1/diagnosis.json`.
