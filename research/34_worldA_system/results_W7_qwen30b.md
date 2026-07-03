# W7-Qwen30B: headline test of the fully-optimized World A self-spec stack at large EP

**Model:** Qwen3-30B-A3B (128 experts, top-8, 48 layers, GQA / FlashAttention — NOT
MLA). **Layout:** attention-DP + EP, tp=1, **EP = DP** across the visible H100s.
**Fabric:** REAL forced-PCIe (`NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0
NCCL_IB_DISABLE=1`) — no A2A emulation. **Stack (spec):**
`VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE=1` (comm-free local-routing draft) +
`VLLM_SELF_SPEC_DRAFT_FULL_REPLICA=1` (draft holds all 128 experts) +
`VLLM_SELF_SPEC_DRAFT_FULL_CG=1` (draft FULL cudagraphs) + draft FP8
(`speculative_config.quantization="fp8"`, target stays bf16). **No-spec baseline:**
same DP+EP layout, no speculative_config. Method: two-length decode slope
(OUTLEN=160, SHORTLEN=32), CUDA graphs ON, WARMUP=2, ITERS=3, greedy, identical
prompts/outlen spec vs no-spec. Harness `scripts/w7_qwen30b_timing.py`,
driver `scripts/w7_qwen30b_serial.sh`.

## TL;DR

**The headline stack does NOT cross 1.2–1.35× on Qwen3-30B — it never crosses 1.0×
(best ≈ 0.46×).** Two independent acceptance failures, neither present in the prior
V2-Lite validation, dominate:

1. **Acceptance collapses at DP≥4 for the full-replica draft** — per-token accept
   0.46 (DP2) → 0.034 (DP4) → 0.004 (DP8); accept_len 1.92→1.07→1.01. This is
   present for BOTH the PIECEWISE (correct-attention) and FULL_CG draft, so it is
   **not** the FULL-CG attention capture — it is the full-replica draft itself at
   data-parallel size > 2. **This is the dominant finding and it makes the
   speedup question moot at EP=4/8.**
2. **The draft FULL_CG capture is itself lossy on Qwen3's FlashAttention (FA3)
   backend** — even at DP=2 it drops accept_len 2.00 (PIECEWISE) → 1.60 (FULL_CG).
   Isolated to FULL_CG (not FP8, not skip-rebuild, not sample-in-graph). The fg2
   "1-split exact attention" fix that restored MLA accept does not carry to FA3.

Because acceptance is ~1.0 at EP=8, **the EP-scaling hypothesis (bigger EP → bigger
saved all-to-all → higher speedup) cannot be tested as intended**: there is almost
no acceptance to monetize. The FULL_CG speedup is flat ~0.44× across EP=2/4/8.

FP8 full-replica **fits** at every DP (no OOM; ~13–76 GiB/GPU). One crash: at
**batch=256 with FULL_CG** the draft hits "CUDA graph capturing detected at an
inappropriate time" (engine dies; b8–128 unaffected).

---

## 1. EP-scaling (batch=64, K=2): speedup vs EP

| DP=EP | no-spec tok/s | spec(FULL_CG) tok/s | **sp(FCG)** | accept_len(FCG) | spec(PIECEWISE) tok/s | sp(PW) | accept_len(PW) |
|------:|------------:|-----------------:|--------:|-----------:|-------------------:|------:|----------:|
| 2 | 3435 | 1534 | **0.446** | 1.44 | 561 | 0.163 | 1.92 |
| 4 | 2803 | 1144 | **0.408** | 1.07 | 375 | 0.134 | 1.07 |
| 8 | 1855 |  809 | **0.436** | 1.01 | 314 | 0.169 | 1.02 |

- **Speedup does NOT rise with EP.** FULL_CG is flat ~0.41–0.45× across EP=2/4/8;
  PIECEWISE ~0.13–0.17×. No-spec throughput drops with EP (3435→2803→1855,
  consistent with the all-to-all growing), but the spec side never benefits because
  acceptance is gone.
- **accept_len collapses with EP for BOTH variants** (1.44/1.92 at EP2 → ~1.0 at
  EP8). Raw counts (b64, K2): PW DP2 = 17577 accepted / 38408 drafted (0.458);
  PW DP4 = 2490 / 68060 (0.037); FCG DP8 = 282 / 72406 (0.004). The drafts are
  produced fine; the verify rejects ~everything at DP≥4.

## 2. Full sweep at EP=8 (DP=8) — spec draft = FULL_CG (named headline stack)

no-spec tok/s @ EP8: b8=496, b32=1191, b64=1859, b128=2619, b256=3402.

| batch | K | accept_len | spec tok/s | no-spec tok/s | **speedup** |
|------:|--:|----------:|----------:|-------------:|----------:|
| 8   | 2 | 1.03 | 189 | 496  | 0.381 |
| 32  | 2 | 1.01 | 548 | 1191 | 0.460 |
| 64  | 2 | 1.01 | 795 | 1859 | 0.427 |
| 128 | 2 | 1.01 | 699 | 2619 | 0.267 |
| 256 | 2 | — (CRASH) | — | 3402 | — |
| 8   | 3 | ~1.0 | 182 | 496  | 0.368 |
| 32  | 3 | ~1.0 | 470 | 1191 | 0.395 |
| 64  | 3 | ~1.0 | 685 | 1859 | 0.368 |
| 128 | 3 | 1.01 | 595 | 2619 | 0.227 |
| 256 | 3 | — (CRASH) | — | 3402 | — |
| 8   | 4 | ~1.0 | 139 | 496  | 0.281 |
| 32  | 4 | ~1.0 | 388 | 1191 | 0.326 |
| 64  | 4 | ~1.0 | 529 | 1859 | 0.284 |
| 128 | 4 | — (CRASH) | — | 2619 | — |
| 256 | 4 | — (CRASH) | — | 3402 | — |

Best FULL_CG: **speedup 0.460 at (batch=32, K=2)** — all points < 1.0; accept_len
≈ 1.0 throughout (the DP=8 collapse). **Does not cross 1.2–1.35× (or 1.0×).**

**FULL_CG large-batch crash** ("CUDA graph capturing detected at an inappropriate
time"): at K=2/3 it dies at batch=256; at K=4 it dies earlier, at batch=128. The
engine dies and the batch loop stops; smaller batches in that K are valid.

> The PIECEWISE sweep (FULL_CG off) was started as a supplementary "correct-accept"
> comparison but skipped after EP-scaling: at EP=8 PIECEWISE accept is also ~1.0
> (same DP≥4 collapse) while the PIECEWISE draft forward is ~4× slower, so it adds
> no signal at large wall-clock cost. The FULL_CG sweep above IS the requested
> headline-stack sweep.

## 3. Accept-length diagnostics

### 3a. FULL_CG vs FP8 vs skip-rebuild (DP=2, b8, K=2) — isolates the FULL_CG bug

| config | accept_len | per-token | conclusion |
|---|---:|---:|---|
| FP8 + FULL_CG (headline) | 1.601 | 0.300 | regressed |
| bf16 + FULL_CG | 1.596 | 0.298 | ⇒ NOT FP8 (bf16 == fp8) |
| FP8 + FULL_CG + DISABLE_SKIP_REBUILD | 1.577 | 0.288 | ⇒ NOT skip-rebuild |
| FP8 + FULL_CG + no-skip + no-sample-in-cg | 1.587 | 0.294 | ⇒ NOT sample-in-graph |
| **FP8 + PIECEWISE (FULL_CG OFF)** | **2.002** | **0.501** | **recovers** |

**The FULL_CG draft attention capture is lossy on Qwen3 (FA3 / GQA).** Root cause:
FA3's per-build `scheduler_metadata` (computed from `seq_lens`/`max_seq_len`,
num-splits-dependent) is baked into the captured graph via a persistent buffer; the
fg2 fix capped `max_num_splits=1` for MLA, but on FA3 the captured schedule still
does not replay correctly for the draft's per-step *growing* sequences → the 2nd+
draft token attends incorrectly → rejected. (MLA's decode kernel does not depend on
this the same way, which is why fg2 sufficed on V2-Lite but not here.)

### 3b. DP≥4 collapse root-cause (DP=2 vs DP=4, PIECEWISE so attn is correct)

<!-- DP4_DIAG_TABLE -->

## 4. Comm-free confirmation (EP=8)

<!-- COMMFREE -->

Structural: the full-replica draft builds `use_ep=False expert_map=None` (logged at
load), i.e. it holds all 128 experts as a non-EP replica and therefore **cannot**
issue an MoE all-to-all (no EP dispatch/combine). The EP=8 VERIFY does the real
all-to-all.

## 5. FP8 fit / losslessness

- **FP8 full-replica fits at every DP** (DP2 ≈ 35–74 GiB, DP8 ≈ 13 GiB/GPU after
  KV; no OOM). The draft loads FP8 (`Draft model quantization: fp8 (target stays
  None)`).
- **Losslessness:** <!-- LOSSLESS -->

## 6. Anomalies (summary)

1. **DP≥4 acceptance collapse of the full-replica draft (dominant).** Accept fine at
   DP=2 (per-token 0.46), ~0 at DP=4/8 (0.034 / 0.004), for BOTH PIECEWISE and
   FULL_CG. The full-replica draft inherits `data_parallel_size=N` with
   `enable_expert_parallel=False` (`speculative.py:1020`); the prior V2-Lite
   validation only ever ran DP=2. See 3b for the bf16/fp8/EP-shard isolation.
2. **FULL_CG draft attention lossy on FA3** (3a): accept 2.0→1.6 at DP=2, isolated
   to FULL_CG. fg2's MLA fix does not cover FA3.
3. **batch=256 + FULL_CG crash** (CUDA-graph capture at inference time).

## Files
- Harness: `scripts/w7_qwen30b_timing.py` (adds `W7_FULL_CG`, `W7_FULL_REPLICA`,
  `W7_LOG_A2A` knobs), driver `scripts/w7_qwen30b_serial.sh`,
  lossless `scripts/w7_qwen30b_lossless.py`, analyze `scripts/w7_qwen30b_analyze.py`.
- Data: `data/w7q_epscale_*`, `data/w7q_sweep8_*`, `data/w7q_diag_*`,
  `data/w7q_acc_*`, `data/w7q_commfree_*`, `data/w7q_lossless_*`.
