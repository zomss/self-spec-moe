# P4 B0 GPU contention probe authorization V2

Date: 2026-08-10

Decision: **APPROVE one fresh, non-scored GPU-0/GPU-1 contention probe.**

V1 completed both serial baselines but stopped before the concurrent pair
because its `47xxx` rendezvous ranges overlapped the host ephemeral TCP range.
V1 is consumed and preserved; none of its measurements are reused.

V2 reruns the complete four-boot matrix with fresh output and compile caches.
It binds these non-ephemeral, disjoint ranges:

- serial GPU 0: `20000`--`20099`;
- serial GPU 1: `20100`--`20199`;
- concurrent GPU 0: `20200`--`20299`; and
- concurrent GPU 1: `20300`--`20399`.

The workload, CPU affinities, four R8/K4 rounds, 95% episode filter, 1%
per-GPU contention bound, 1% contention-ratio agreement bound, and 2%
cross-GPU agreement bound are unchanged. A pass permits preparation of a
separate block-parallel value-screen authorization; it does not itself grant
execution or scoring authority for the 432-capture screen.

Artifacts:

- `data/p4/p4_b0_gpu_contention_probe_authorization_v2.json`
- `schemas/p4_b0_gpu_contention_probe_authorization_v2.schema.json`
- `scripts/run_p4_b0_gpu_contention_probe_v2.py`
- `scripts/validate_p4_b0_gpu_contention_probe_authorization_v2.py`
- `tests/test_p4_b0_gpu_contention_probe_v2.py`
