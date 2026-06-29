# Results: comm-free pruning of the exact-verify tree -- NO-GO (31a)

Date: 2026-06-29. Qwen3-30B-A3B, 0.5E local-routing draft (bf16; FP4 would lower beta
~0.03, separately characterized). Wide tree [2,2,2,2] (30 nodes). Stage A reuses the
Phase 27 tree-attention verify; Stage B uses the measured Phase-24 forced-PCIe cost models
(`S_draft(T)=15.3+0.0152T`, `S_verify(T)=22.7+0.0495T`, comm-free draft / full-EP verify).

## Stage A -- the pruner works (near-oracle), but barely beats the chain

`L_full(wide [2,2,2,2]) = 3.83` accept (of max 4 -- branching h(2)~0.98 beats chain
attrition). Pruned accept length vs verify node budget N:

| N | P_conf (free) | P_exact (oracle) | recall_conf | direct chain at N | direct d2b2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1.73 | 1.85 | 0.45 | chain_k2 1.77 | -- |
| 3 | 2.50 | 2.62 | 0.65 | chain_k3 2.44 | -- |
| 4 | 3.13 | 3.33 | 0.82 | chain_k4 3.08 | -- |
| 6 | 3.40 | 3.58 | 0.89 | -- | 1.96 |
| 8 | 3.58 | 3.71 | 0.94 | -- | -- |

Findings: **(1) free confidence-pruning is near-oracle** (P_conf ~= 95% of P_exact). **(2)
pruned-wide barely beats the chain** at matched verify nodes (+0.05-0.06 accept, ~+2.5%;
oracle ceiling +7%). **(3) pruned-deep >> shallow tree** (N=6: 3.40 vs full_d2b2 1.96) --
depth carries accept length, so a pruned deep tree dominates a shallow one. But the chain
(pure depth) was already the efficient shape.

## Stage B -- economics kill it (the draft cost, not the verify)

Speedup vs autoregressive (measured cost models):

| design | B=8 | B=128 | B=512 |
| --- | ---: | ---: | ---: |
| best chain | 1.18 | **1.18** | **1.27** |
| full_d2b2 (Phase 27) | **1.22** | 0.89 | 0.62 |
| wide-no-prune (verify 30) | 1.14 | 0.46 | 0.24 |
| **31a pruned-wide (P_conf)** | 1.19 | 0.87 | 0.66 |
| 31a pruned-wide (ORACLE) | 1.22 | 0.91 | 0.69 |

- **High batch (B>=128): 31a is far below the chain** (0.66-0.87 vs 1.18-1.27). Fails GO
  gate B-31a (>=10% over chain) decisively.
- **Low batch (B=8): 31a (1.19) ~= full_d2b2 (1.22)** -- no improvement; even the ORACLE
  pruner only ties Phase 27's best.

**Root cause (the key insight):** pruning saves VERIFY comm but you still pay the full
WIDE-TREE DRAFT (a full-model comm-free forward over wide frontiers -- e.g. 178 ms at
B=512 vs the chain's 92 ms). Pruning the verify cannot recover a draft you already paid.
The chain wins at high batch (cheap draft + near-same accept at matched verify); full_d2b2
wins at low batch. Even a perfect pruner does not change this.

## Implication for 31b (EAGLE composition) -- also likely NO-GO

31a's free signal (`P_conf`, the draft's own confidence) is **already ~95% of the oracle**.
For 31b, EAGLE's free head probs (`P1`) are the analog of `P_conf` -- if they are similarly
near-oracle, a SEPARATE local-routing MoE pruner pass (`P2`, ~0.4x a verify) has <=5%
headroom to improve the ranking, which cannot justify a near-full extra forward. This is
exactly the README's risk #1 (`P2 ~= P1`). So 31b is gated NO-GO by the same evidence,
without needing an EAGLE head.

## Verdict

**Direction closed (clean negative).** Comm-free pruning of the exact-verify tree does not
beat the chain (high batch) or Phase 27's small tree (low batch): the free signal already
prunes near-optimally (no room for an expensive pruner), and pruning saves only the verify
while the wide-tree draft cost dominates. **Recommend plain EAGLE-on-MoE with a CHAIN (high
batch) or small tree (low batch)** -- the verify-comm-scaling insight (Phase 27: small/chain
trees on comm-bound MoE) is the durable contribution; tree pruning adds nothing on top.

Self-MoE-spec does NOT earn extra keep at the verify-pruning layer once EAGLE exists.

## Quantization note (FP4 is the thesis; why it doesn't rescue this on H100)

The pruner/draft cost above is comm-free local routing at **bf16**. The project's bit-width
is **FP4** (memory thesis), so the right question is whether FP4 cheapens the draft. On
**H100 (Hopper) FP4 is a memory lever, not a speed lever**: no FP4 tensor cores, so vLLM
dequantizes FP4->bf16 and the matmul runs in bf16. Therefore:
- High batch (compute-bound, where 31a fails 0.66x vs chain 1.27x): FP4 gives ~no compute
  speedup -> the wide-tree DRAFT cost that kills 31a is unchanged -> **NO-GO holds with FP4
  on H100**. The bf16-speed economics above already represent FP4-on-Hopper.
- Low batch (read-bound): FP4 cuts weight-read ~4x -> ~10-25% draft speedup, but that
  regime only tied full_d2b2 -> verdict unchanged.

(FP8 *would* speed the draft on H100 -- Hopper has FP8 tensor cores -- but at 2x the FP4
memory, it is off the memory thesis, so it is not the project's design point.)

**Hardware caveat:** on **Blackwell** FP4 has native tensor cores -> an FP4 comm-free draft
IS faster on compute (high batch too), so the wide-tree draft cost drops and the prune
economics could improve/flip. So "FP4 makes the pruner cheap enough" is **false on H100,
plausibly true on Blackwell** -- this is tested directly below.

## Two-phase cascade economics (quantized phase-1) -- NO-GO even on Blackwell

The user's framing: a CASCADE -- EAGLE draft -> cheap **quantized comm-free phase-1**
(prunes the wide tree, lossless) -> full-precision full-EP **phase-2** verify (small tree).
Valid and lossless (phase-1 only prunes; phase-2 does the exact rejection sampling), and it
elegantly unifies World A (FP4 local routing = phase-1) + World B (EAGLE + full verify).
But does the quantized phase-1 make it efficient? Swept the phase-1 cheapness
`q` (phase-1 = `q*S_skip(B*N_wide)`, q=1 bf16; FP8-H100 ~0.6; FP4-Blackwell ~0.4),
EAGLE draft treated as FREE -> generous (`prune_cascade_economics.py`):

| batch | chain (ref) | phase-1 FP8 q0.6 | FP4-Blkwl q0.4 | ideal q0.2 | free q0 |
| ---: | --- | --- | --- | --- | --- |
| 8 | baseline | LOSE | LOSE | LOSE | ~tie |
| 128 | baseline | LOSE | LOSE | LOSE | ~tie |
| 512 | baseline | LOSE | LOSE | LOSE | ~tie |

**Cross-over: the cascade beats the chain only if `q < 0.01-0.07`, i.e. phase-1 < ~1-2 ms
(vs a full 1-token verify of 23-48 ms) -- essentially FREE.** No quantization reaches that:
FP8-H100, FP4-Blackwell, and even a free EAGLE draft all LOSE.

**Root cause (cleaner than the 31a draft-cost story):** pruning a wide tree requires
*forwarding the whole wide tree* in phase-1 (cost scales with `N_wide`), and the accept-
length gain of a pruned-wide tree over a plain chain is only ~+2.5% (Stage A; +7% even at
oracle) because `h(b)` saturates (the chain already captures most of the accept length). So
phase-1 is pure added cost chasing a tiny gain -- you are strictly better off **drafting a
chain directly** and skipping the wide-tree-then-prune entirely. This holds even when
phase-1 is comm-free AND quantized AND the draft is free.

**When the cascade WOULD pay:** only on an acceptance profile with a *large* tree-vs-chain
gap (top-1 often wrong but top-b right -- high-entropy/ambiguous decoding). On Qwen3's
confident profile (`h(1)=0.875` already high), the gap is too small. So the cascade is
**profile-dependent and NO-GO here**; quantizing phase-1 helps its cost but cannot
manufacture headroom that the acceptance profile doesn't provide.
