# Phase 60 results — 1-rail (1 NIC/node) three-way: no-spec vs EAGLE3 vs World A

Setup: see [README.md](README.md). Qwen3-30B-A3B, 2-node DP16/EP16, short
context (2k), chat + on-distribution prompts (same file as the Phase-57 8-rail
chat references), b8/32/64, greedy, two-length decode slope. Only fabric change
vs the 8-rail references: `NCCL_IB_HCA='^mlx5_8'` -> `'=mlx5_0'` (1 NIC per
node, per-GPU inter-node bandwidth /8). All four arms completed FIRST TRY (no
DP16-EAGLE wedge this time); every row is tight (no suspect flags; worst
decode-std 13% at World A b32, rest <=2%).

## Rail-engagement check (PASS)

`run_smoke_nccl.sh` (nospec b8, NCCL_DEBUG=INFO): every rank on BOTH nodes
logs exactly one HCA:

```
h107: NCCL INFO NET/IB : Using [0]mlx5_0:1/RoCE [RO]; OOB enmlx0:192.168.0.17
h106: NCCL INFO NET/IB : Using [0]mlx5_0:1/RoCE [RO]; OOB enmlx0:192.168.0.16
```

## 1. The premise fails: 1 NIC/node is NOT high-f at this scale

Implied f from the no-spec step-time ratio (same token count both sides, so
t1/t8 = inverse tok/s ratio; comm grew, compute didn't:
f_1rail ~= 1 - t8/t1):

| batch | t_8rail (s) | t_1rail (s) | t1/t8 | f_1rail |
|---:|---:|---:|---:|---:|
| 8  | 1.963 | 2.636 | 1.34 | **0.26** |
| 32 | 2.591 | 2.828 | 1.09 | **0.08** |
| 64 | 3.140 | 3.621 | 1.15 | **0.13** |

Cutting inter-node NIC bandwidth 8x grew the no-spec decode step only
9-34%. **Real f stayed 0.08-0.26 — nowhere near the ~0.7 expected.** This is
Phase 51/52's latency-bound-collectives finding, now confirmed from the
bandwidth side: at 30B decode payloads even ONE 400G NIC is far from
bandwidth-bound; the collective cost is latency/hop overhead, which rail count
does not change. The largest relative hit is at b8 (1.34x), the SMALLEST
payload — consistent with per-NIC QP contention adding latency (8 ranks now
share one HCA), not with a bandwidth wall (which would grow with batch).

**Consequence: the high-f regime cannot be realized by rail restriction at
this model scale.** Phase 50's f~0.7 emulation corresponds to a fabric whose
per-collective LATENCY is several times this one (injected delay), not to a
bandwidth-starved version of it. A real high-f deployment needs latency
inflation (oversubscribed/multi-hop fabrics, congestion) or much larger
per-step payloads (bigger hidden, much larger per-rank batch), not fewer NICs.

## 2. Three-way table (1 rail): tok/s (accept) [speedup vs 1-rail no-spec]

| config | b8 | b32 | b64 |
|---|---:|---:|---:|
| no-spec     | 388.5 [1.00x] | 1448.6 [1.00x] | 2262.4 [1.00x] |
| EAGLE3 K=1  | 409.7 (1.70) [1.05x] | 1584.6 (1.72) [1.09x] | 2204.8 (1.72) [0.97x] |
| EAGLE3 K=2  | 669.2 (2.20) [**1.72x**] | 1759.8 (2.19) [1.21x] | 2183.5 (2.21) [0.97x] |
| World A K=2 | 333.6 (2.92) [0.86x] | 793.7 (2.92) [0.55x] | 1020.1 (2.92) [0.45x] |

8-rail chat references (Phase 57, same prompts/harness): no-spec
521.6/1581.1/2611.1; EAGLE K1 568.7/1840.1/2475.7; EAGLE K2
732.0/2047.6/2234.9. World A 8-rail (Phase 52, NON-chat prompts): ~1233 at
b64, accept 2.90.

Step-time growth 1-rail/8-rail per arm (chat-parity arms):

| batch | no-spec | EAGLE K1 | EAGLE K2 |
|---:|---:|---:|---:|
| 8  | 1.34 | 1.39 | 1.09 |
| 32 | 1.09 | 1.16 | 1.16 |
| 64 | 1.15 | 1.12 | **1.02** |

- **Acceptance is fabric-invariant**, again: World A 2.922-2.923 (= Phase 52's
  2.90; chat prompts don't move it), EAGLE 1.70-1.72 (K1) / 2.19-2.21 (K2) =
  the 8-rail chat values to the third digit.
- **The predicted verify-comm inflation penalty for EAGLE never
  materializes**: EAGLE's verify carries 2-3x the tokens of a no-spec step, so
  on a bandwidth-bound fabric its step should grow MORE than no-spec's under
  an 8x bandwidth cut. Measured: same or LESS (K2 b64: 1.02x vs no-spec
  1.15x). Latency-bound collectives price draft VOLUME at ~0 — the opposite of
  the high-f law — and EAGLE K2 b8 actually IMPROVES its speedup at 1 rail
  (1.72x vs 1.40x at 8 rails): no-spec pays the per-step latency hit every
  token, the spec cycle only once per accept_len tokens.

## 3. b64 ranking (1 rail) — the headline

| rank | config | tok/s | vs no-spec |
|---:|---|---:|---:|
| 1 | **no-spec**  | 2262.4 | 1.00x |
| 2 | EAGLE3 K=1   | 2204.8 | 0.97x |
| 3 | EAGLE3 K=2   | 2183.5 | 0.97x |
| 4 | World A K=2  | 1020.1 | 0.45x |

**The Phase-50 emulated crossover (World A > EAGLE, possibly > no-spec, at
high f) is NOT confirmed on real hardware — because rail restriction does not
produce high f.** At the real f this fabric delivers (~0.13 at b64), the
8-rail ordering persists: no-spec wins at serving batch, EAGLE is break-even,
World A stays draft-compute-bound at 0.45x (8-rail Phase 52: 0.38-0.41x; the
comm-free draft saves comm that is not the cost). The prediction itself
(accept-per-verified-token = comm efficiency) remains UNTESTED at real high f:
what this experiment establishes is that the common 1-NIC-per-node production
topology is NOT that regime for a 30B/2k-context EP16 workload — a useful
negative result. The regime where the crossover could bind needs f >~ 0.5 from
latency or payload, not from NIC count.

Secondary: at small batch the slower fabric HELPS spec decode relatively
(EAGLE K2 b8 1.72x — its best 2-node speedup measured on this testbed): spec's
fewer, chunkier steps amortize per-step fabric latency.

## Data / logs

- `data/w72n_q30b_1rail_nospec_nospec_cg_nospec.json`,
  `data/w72n_q30b_1rail_eagle_spec_cg_K{1,2}.json`,
  `data/w72n_q30b_1rail_worldA_spec_cg_K2.json`,
  `data/w72n_q30b_1rail_smoke_nospec_cg_nospec.json` (smoke, b8 1-iter).
- `data/rail_engagement_evidence.txt` (per-rank `NET/IB : Using` lines, both
  nodes).
- Runner logs under `logs/` (gitignored):
  `{nospec,eagleK1,eagleK2,worldA}_try1{,_h106}.log`,
  `smoke_nccl_h10{6,7}.log` (full NCCL_DEBUG=INFO smoke).
