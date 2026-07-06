# Phase 66 results — shared-KV self-draft (eliminate draft KV duplication)

Code: `VLLM_SELF_SPEC_SHARED_KV=1` (default off) binds the drafter's 48
attention layers to the TARGET layers' KV tensors via the existing
cross-layer-sharing plumbing (runner `shared_kv_cache_layers`), with the
draft's KV writes kept ENABLED and PAD-masked down to not-yet-verified
slots (appended sampled-token slot + chain-drafted slots). See commit
"[W7][66] shared kv: draft binds to target KV tensors".

## Validation ladder

| rung | check | ref (duplicated KV) | shared-KV | verdict |
|---|---|---|---|---|
| 1a | canary 2k, W=0, K=2 accept | 3.000 (P62, bit-exact) | **3.000** (378.4 tok/s) | PASS |
| 1b | canary 2k, W=64, K=2 accept | 2.540 (P62) | **2.689** (200.1 tok/s) | PASS (better) |
| 2 | pool tokens/rank (canary cfg) | 217,696 | 435,408 (**2.00x**) | PASS |
| 2b | pool tokens/rank (16k 1-node) | 221,664 (P62 B2) | 443,344 (**2.00x**) | PASS |
| 3 | 16k W512 K=4 accept (1-node DP8) | 4.584 (P62 B2) | **4.676** (206.2 tok/s) | PASS |

## Pool restoration (16k protocol configs)

Measured 2-node pools, tokens/rank at gpu_mem 0.90 (dup refs from
P64/P65):

| engine | dup (192 KiB/tok) | shared (96 KiB/tok) | b12 | b13 | b14 | b24 | b32 |
|---|---|---|---|---|---|---|---|
| no-spec (measured) | -- | 577,472 | 34% | 37% | 40% | 68% | 90% |
| EP-routed / node-local self-spec | 236,512 | **490,176** (2.07x) | 40% | 43% | 47% | 80% | **106%** |
| fp8-replica self-spec | 115,968 | **243,632** (2.10x) | 80% | 87% | 94% | 161% | -- |

(2-node measured pools, tokens/rank at gpu_mem 0.90; demand 16,313
tok/req.) The sharing restores b24 residency for the weight-shared arms
(80%) and b12 residency for the fp8 replica (80%, was 171% in P65 —
the replica costs 246.5k tokens = 22.6 GiB at the halved 96 KiB/token).
b32 is 106% of the shared pool: still over, and measurement confirms the
Phase-64 preemption livelock survives at just 6% over (a_ep K4 b32,
3000 s: 1592 scheduler "Waiting:" lines, re-prefill bursts ~1160 tok/s
alternating with ~2 tok/s decode, zero completed passes). On the
replica the residency knee is measured directly: b13 (87%) 335.1+-46.2
with 88 preemption waves, b14 (94%) 217.4+-18.9 with 128 — the knee
bites between 80% and 87% of pool. Largest clean batches: b24
(weight-shared arms), b12 (replica).

## 2-node 16k measurement (Phase 64/65 protocol + shared KV, W512 s16)

no-spec denominators, measured THIS phase (same day/stack, own launches):
b12 407.9+-13.1 (P64's 395.7 holds), b24 529.4+-1.4 (never measured
before — the throughput peak), b32 496.5+-27.4 (P64's 379.0 does NOT
hold: +31% stack drift since P64 — all speedups below use TODAY's
denominators).

| arm | K | b12 tok/s (accept) | x | b13 (waves) | b14 (waves) | b24 tok/s (accept) | x | b32 |
|---|---|---|---|---|---|---|---|---|
| A ep bf16 | K=2 | 186.4+-0.5 (2.933) | 0.46x | -- | -- | 301.1+-2.3 (2.903) | 0.57x | -- |
| A ep bf16 | K=4 | 207.1+-4.0 (4.746) | 0.51x | -- | -- | 334.1+-1.2 (4.638) | 0.63x | LIVELOCK |
| B fp8 replica | K=2 | 251.0+-5.0 (2.894) | 0.62x | -- | -- | over-pool | -- | -- |
| B fp8 replica | K=4 | **350.2+-5.8 (4.660)** | **0.86x** | 335.1+-46.2 (4.651) | 217.4+-18.9 (4.626) | over-pool | -- | -- |
| C node-local | K=2 | -- | -- | -- | -- | 337.6+-4.1 (2.430) | 0.64x | -- |
| C node-local | K=4 | -- | -- | -- | -- | 239.2+-1.7 (3.274) | 0.45x | not run |

Accept guards all PASS: A K4 4.746/4.638 vs 4.676 ladder ref (drift
<=0.04); B 4.660/4.626 (above the 4.48-4.55 expectation — the shared
cache gives the fp8 replica target-exact window reads, recovering the
P62 arm-C accept cost); window engaged everywhere (win_seq ~530 of
~16.4k). Arm C answers its open question: the window x node-local
accept composition is 2.430 @K2 / 3.274 @K4 = a 0.84/0.71 factor on the
EP-routed accept — worse than the hoped 0.85-0.9 and compounding with
chain depth, so node-local only pays at K=2 (337.6 vs A's 301.1: +12%,
still 0.64x).

## Fixed/marginal split (cycle = F + K*D from the K2/K4 pairs)

| arm | batch | cycle K2 ms | cycle K4 ms | F ms | D ms/step |
|---|---|---|---|---|---|
| A ep bf16 | b12 | 188.8 | 275.0 | 102.6 | 43.1 |
| A ep bf16 | b24 | 231.4 | 333.2 | 129.6 | 50.9 |
| B fp8 replica | b12 | 138.4 | 159.7 | **117.1** | **10.6** |
| C node-local | b24 | 172.8 | 328.5 | (17.1) | (77.8) |

(C's solve is unphysical — F below the verify forward alone; the linear
model breaks because the accept collapse changes the K4 cycle's batch
composition. A/B solves are consistent with P64/P65.)

The comm-free draft's marginal step is now essentially at its GPU floor
(D 10.6 ms at b12 vs 21.6 at b6 in P65 — the chain CPU dispatch now
overlaps the larger GPU step), while the EP-routed draft stays
comm-bound (D 43-51 ms; ~26 ms NCCL per draft forward, P64 trace). The
cycle is now FIXED-cost dominated: for B at b12, cycle 159.7 = verify
~29.4 (no-spec step) + chain 4x10.6 = 42.5 + **propose-fixed ~87.7 ms**
(step-0 draft forward + gathers/compaction + sampling glue + DP coord +
bookkeep — 2x the P65-b6 fixed of 44 ms; which term scales with batch
needs a trace).

## Headline verdict

**No arm crosses 1.0x at b24-b32. Best: 0.86x (arm B, fp8 comm-free
replica + shared KV, K=4, b12 — 350.2 vs 407.9); at b24 the best is
0.63-0.64x (A K4 / C K2).** The shared-KV fix did exactly what it
promised — pool 2.07-2.10x, serving-batch residency restored, accept
preserved or improved — and the E2E gap still stands, for two reasons:

1. **The denominator moved.** Today's no-spec is 529.4 at b24 (never
   measured in P64) and 496.5 at b32 (+31% vs the P64 ref). The A-arm
   projection "~1.0-1.2x at b32" was against 379; against today's
   engine the same tok/s would be 0.63-0.67x.
2. **The propose path's fixed cost (~88 ms/cycle at b12, ~84 at b24)
   dwarfs the now-nearly-free marginal draft steps.** Speculation buys
   4.66 tokens/cycle but the cycle costs 5.4x a no-spec step (159.7 vs
   29.4 at b12).

### FULL-CG projection (from measured F/D)

Chain-FULL-CG (P65 scoping: the piecewise chain's ~16 ms/step CPU ->
GPU-floor D ~= 7 ms; NCCL floor stays for EP-routed: D_cg ~= 26+8 = 34):

| arm | point | cycle_cg ms | tok/s | x today | x needed for 1.0 |
|---|---|---|---|---|---|
| B fp8 replica | b12 K4 | 117.1 + 4x7 = 145.1 | 385 | **0.95x** | F <= 109 ms |
| A ep bf16 | b24 K4 | 129.6 + 4x34 = 265.6 | 419 | 0.79x | D <= 20 (< NCCL floor: unreachable) |

So chain-FULL-CG alone takes the best arm to ~0.95x — parity, not a
win. The remaining lever is F: at D=7, crossing 1.0x at b12 needs
F <= 109 (8 ms away); 1.3x needs F <= 77 — i.e. the next phase must
attack the ~88 ms propose-fixed block (step-0 + glue + coordination),
not the chain. The EP-routed draft cannot reach 1.0x at b24 at any K
with F ~= 130: even a zero-CPU chain leaves it comm-bound.

## Data

Committed: `data/w72n_q30b_p66_{aep,brep,cnl}*_spec_cg*.json` (arm
points), `data/w72n_q30b_p66_nospec_b24_*.json` +
`data/w72n_q30b_p66_nospec_recheck_*.json` (denominators),
`scripts/analyze.py` (table + F/D solves). On disk only: engine logs
under `logs/` (`a_ep_K4_b32_try1.log` = the 106% livelock evidence;
`b_rep_K4_b13_try1.log` / `b_rep_K4_b14_try1.log` = the 87%/94%
preemption waves; `a_ep_K4_try1.log` = a fabric-manager Error-802
launch failure on h107 — nvidia-fabricmanager was restarted 12:48,
recovered by the runner's retry).
