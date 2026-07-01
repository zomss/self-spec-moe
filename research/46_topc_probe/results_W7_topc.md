# W7 top-C draft-prune elasticity probe — results

**Question.** The comm-free full-replica self-spec draft computes the model's full
top_k experts per token. If instead the DRAFT computes only the **top-C**
largest-gate-weight experts (C < top_k, renormalized over C — a *smart* prune),
how fast does accept_len fall vs how much draft compute is saved? Lossless
regardless of C (the full-top_k VERIFY corrects). This gates whether the **W2b
bounded expert-cache** build is worth it.

**Setup.** HEAD `3e98800d3` (worktree `w7-topc-probe`). Piecewise-on trio
(`DRAFT_FULL_REPLICA=1 DRAFT_LOCAL_ROUTE=1 DRAFT_FULL_CG=1 COMPILE_CONSISTENT=1
DRAFT_CHAIN_PIECEWISE=1`), forced-PCIe (`NCCL_P2P_DISABLE/NVLS/IB`), K=2, greedy,
batch 64, DP8, FP8 draft, `VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0`.
Probe: `VLLM_SELF_SPEC_DRAFT_TOPC=C`. accept_len from spec metrics
(`1 + accepted/num_drafts`); draft_ms from the env profiler `draft_forward`
region (a single in-chain decode-step draft forward). g(C)=draft_ms(C)/draft_ms(top_k).
Projection: `speedup(C,f) = accept_len(C) / (K·g(C)·(1−f) + 1)`, K=2.
Path confirmed modular (log: "DRAFT layer L: pruning top_k=K -> computing C ...",
VERIFY keeps full top_k); no monolithic-warning. Sanity: at C=top_k the prune is a
no-op — Qwen3-30B C=8 accept 2.914 reproduces the recon full-replica ~2.9; g(top_k)=1.

---

## 1. accept_len(C) and g(C)

### Qwen1.5-MoE-A2.7B (top_k=4, 60 experts, 4 shared experts + dense)

| C | accept_len | draft_ms | g(C) | verify_ms |
|---|-----------|----------|------|-----------|
| 4 (full) | 2.871 | 10.08 | 1.000 | 35.7 |
| 3 | 2.024 | 9.44 | 0.937 | 30.9 |
| 2 | 1.318 | 9.05 | 0.898 | 33.2 |
| 1 | 1.032 | 8.68 | 0.861 | 38.0 |

Accept **collapses fast** (2.87 → 2.02 at 4→3, −30%) while draft compute barely
moves (g only 0.94). g **floors near 0.86** — this model's routed MoE is a small
fraction of the draft forward (4 shared experts + dense + attention dominate), so
skipping routed experts saves almost nothing. Bad trade at every C.

### Qwen3-30B-A3B (top_k=8, 128 experts, no shared expert)

| C | accept_len | draft_ms | g(C) | verify_ms |
|---|-----------|----------|------|-----------|
| 8 (full) | 2.914 | 15.82 | 1.000 | 63.6 |
| 7 | 2.893 | 16.16 | 1.022 | 59.3 |
| 6 | 2.845 | 14.60 | 0.923 | 55.2 |
| 5 | 2.776 | 14.68 | 0.928 | 59.2 |
| 4 | 2.677 | 13.31 | 0.841 | 54.9 |
| 3 | 2.526 | 12.31 | 0.778 | 54.1 |
| 2 | 2.143 | 10.49 | 0.663 | 49.6 |
| 1 | 1.060 | 10.41 | 0.658 | 64.1 |

Accept **holds well down to C=3** (2.914 → 2.526, only −13% at 3 of 8 experts),
then **collapses at C=1** (1.06 ≈ no-spec — a single expert is a useless draft).
g **floors near 0.66** (C=1,2): the attention + router + dense/norm fixed cost is
~2/3 of the draft forward, so pruning routed experts saves at most ~34%. g is
flat/noisy from C=8→5 (the expert GEMM is small next to the fixed floor; profiler
std ~2-3 ms) and only drops meaningfully at C≤4. Much gentler elasticity than
Qwen1.5-MoE (128 experts, no shared expert → routing is more redundant, so
dropping the low-weight tail barely hurts accept).

---

## 2. Projected speedup(C,f) and optimal C

### Qwen1.5-MoE-A2.7B (K=2)

| C | f=0.40 | f=0.62 | f=0.80 |
|---|--------|--------|--------|
| 4 (full) | **1.305** | **1.631** | **2.051** |
| 3 | 0.953 | 1.183 | 1.473 |
| 2 | 0.635 | 0.784 | 0.970 |
| 1 | 0.508 | 0.624 | 0.768 |

**Optimal C = 4 (= top_k, the full draft) at every f.** Pruning never wins on
Qwen1.5-MoE — accept falls faster than compute saves.

### Qwen3-30B-A3B (K=2)

| C | f=0.40 | f=0.62 | f=0.80 |
|---|--------|--------|--------|
| 8 (full) | 1.324 | 1.656 | **2.081** |
| 7 | 1.300 | 1.629 | 2.054 |
| 6 | **1.350** | **1.672** | 2.078 |
| 5 | 1.314 | 1.628 | 2.025 |
| 4 | 1.332 | 1.633 | 2.003 |
| 3 | 1.306 | 1.587 | 1.926 |
| 2 | 1.193 | 1.425 | 1.693 |
| 1 | 0.592 | 0.707 | 0.839 |

**Optimal C per f:**

- f=0.40: **C=6**, speedup 1.350 vs full(C=8) 1.324 → **+1.9%**
- f=0.62: **C=6**, speedup 1.672 vs full(C=8) 1.656 → **+1.0%**
- f=0.80: **C=8** (full), 2.081 — pruning gives no gain (as expected, the benefit
  concentrates at low/moderate f)

Pruning beats the full draft only marginally, and only at low/moderate f. The best
case (C=6 @ f=0.4, +1.9%) is within a few percent of the profiler's g-noise (the
g=1.02 at C=7 and the flat g from C=8→5 show the expert GEMM is a small, noisy
fraction of the draft cost). C=4 (half the experts) is roughly break-even with the
full draft at every f.

---

## 3. Losslessness — CONFIRMED (top-C prune is lossless)

The full-top_k VERIFY corrects the pruned draft's proposals, so the final output is
independent of C. Verified with a 3-way diff on Qwen3-30B, batch 64, 140 tok/req
(`scripts/diff3.py`):

| comparison | byte-identical | first divergence |
|---|---|---|
| **spec C=4 vs spec C=8** | **True (0/64 reqs differ)** | — |
| spec C=8 vs no-spec | False (53/64) | req 0 pos 6 |
| spec C=4 vs no-spec | False (53/64) | req 0 pos 6 (SAME) |

**spec C=4 == spec C=8, byte-for-byte** → the top-C prune (C=4, half the draft
experts dropped) is **lossless relative to the full comm-free draft**. This is the
losslessness the probe asked for: the verify makes accept-corrected output
independent of C.

The spec-vs-no-spec mismatch is a **pre-existing** property of the comm-free spec
path, NOT caused by top-C: C=8 (the unchanged full draft) diverges from no-spec at
the *exact same* position (req 0 pos 6) as C=4, because the comm-free full-replica
MoE reduce differs from the target's EP all-to-all reduce in bf16 (documented in
research/40_num_divergence), so the verify's greedy argmax occasionally flips on a
near-tie vs the no-spec forward. Since C=8 and C=4 are identical and diverge from
no-spec identically, the divergence is orthogonal to the prune.

---

## 4. Verdict

**No — the accept/compute tradeoff does not justify the W2b bounded-cache build.**

- **Qwen1.5-MoE (top_k=4):** pruning is strictly worse at every f (optimal C = full
  top_k). Accept craters (−30% at 4→3) while g floors at 0.86 (tiny routed MoE next
  to 4 shared experts + dense). Zero upside.
- **Qwen3-30B (top_k=8):** pruning gives at best **+1.9% projected speedup** (C=6 @
  f=0.4), shrinking to +1.0% at f=0.62 and **0% at f=0.8**. The gain is (a) tiny,
  (b) confined to low f, and (c) within the profiler's g measurement noise.

The reason both models disappoint is the same: g has a **high floor** (0.66 for
Qwen3-30B, 0.86 for Qwen1.5-MoE) because the draft forward is dominated by
attention + router + dense/norm, which top-C does not shrink. So even C=1 saves
only ~34% of draft compute, while accept has already collapsed. The W2b build would
add memory savings on top, but the accept/compute half of the ledger is unfavorable
— the compute saving is too small (high fixed floor) to pay for the accept it costs.
The full-top_k comm-free draft (C=top_k) remains the right operating point.
