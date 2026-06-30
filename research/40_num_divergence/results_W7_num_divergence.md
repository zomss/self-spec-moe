# W7 numerical divergence: pin the cause of the MoE accept collapse + test the FP32 fix

**Question.** World A's comm-free FULL-REPLICA draft accept COLLAPSES at large EP
(Qwen1.5-MoE DP=8: comm-free draft accept **2.18** vs the EP all-to-all verify
**5.00**, phase 38). The stated hypothesis: the divergence is the **MoE reduce
STRUCTURE** — the comm-free draft sums a token's top-k experts in ONE local bf16
`moe_sum`, while the EP verify sums per-shard then bf16-reduces cross-rank; same
math, different bf16 summation associativity → divergence; fixable by FP32
accumulation in both reduce paths.

**Setup.** `Qwen/Qwen1.5-MoE-A2.7B` (qwen2_moe, 60 experts top-4, 24 layers,
non-MLA), **DP=8 → EP=8**, forced-PCIe (`NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0
NCCL_IB_DISABLE=1`), greedy, **K=4**, **bf16**, `VLLM_USE_DEEP_GEMM=0
VLLM_MOE_USE_DEEP_GEMM=0`. Accept runs: draft_model self-spec, `FULL_CG=1` +
`COMPILE_CONSISTENT=1`. Dumps: EAGER main-model reduce-structure probe.
Worktree `/data/smcho/ssm-num`, branch `w7-num-divergence`.

Configs (the two routing flags toggle the reduce structure):
- **A** comm-free full replica: `DRAFT_FULL_REPLICA=1 DRAFT_LOCAL_ROUTE=1` (one
  local bf16 sum of all top-k experts).
- **C** EP all-to-all (= the verify): `DRAFT_FULL_REPLICA=0 DRAFT_LOCAL_ROUTE=0`
  (per-shard bf16 sum + cross-rank bf16 `reduce_scatterv`).
- **D** (probe control) plain MoE, EP off, no comm-free skip = canonical standard
  MoE.

New env `VLLM_SELF_SPEC_MOE_FP32_ACCUM=1` forces FP32 accumulation in BOTH MoE
reduce paths: the local `moe_sum` (`TritonExperts.moe_sum` + `fused_moe.py` +
`topk_weight_and_reduce.py`) and the cross-rank `reduce_scatterv`
(`cuda_communicator.py`, upcast partials → fp32 NCCL reduce → downcast).
Instrumentation env `VLLM_SELF_SPEC_MOE_NUM_DUMP` dumps per-token router logits,
expert ids/weights, post-combine MoE output, and final logits at one layer.

---

## TL;DR (the verdict)

**The collapse is NOT bf16-reduce associativity, and FP32 accumulation does NOT
fix it — it makes accept slightly WORSE and costs ~23% throughput.**

1. **The EP all-to-all is numerically faithful.** Config C (EP per-shard + cross-
   rank bf16 reduce) matches the plain MoE (D) to **0.3%** relative — i.e. the EP
   reduce structure is at the bf16-rounding floor, NOT a source of divergence.
2. **The comm-free full-replica path (A) is the outlier**, structurally (not by
   rounding): its MoE output is a different quantity from C/D. FP32 accumulation
   moves A's local sum by only ~0.1% and leaves the A↔C gap **unchanged**
   (0.539 → 0.539) — so the gap is NOT bf16 associativity.
3. **End-to-end, FP32-accum does not recover accept.** A accept **2.18 → 1.83**
   (slightly worse) with FP32 on; C stays ~5.0. FP32 is not the fix.
4. **FP32-accum is not free.** Config A decode throughput **128.9 → 99.7 tok/s
   (−23%)** — the extra fp32 work (and fp32 cross-rank bytes when EP is on) is a
   real cost with no accept benefit.

---

## Step 1 — confirm + quantify the divergence (instrumentation)

DP rank 0, layer 0, the same fixed 47-token prompt, configs A vs C.

| signal | A vs C (bf16) | reading |
|---|---|---|
| (a) router logits  | max\|A−C\| = **0.000** | routing input IDENTICAL |
| (b) expert ids/weights | set-equal **True**, max\|w\| = **0.000** | same experts, same weights |
| (c) post-combine MoE output | rel **0.539**, norms A=5.12 / C=7.29, cos 0.86 | output DIVERGES |
| (d) final next-token logits | max\|A−C\| = 15.6, **argmax FLIP** | the rejection |

**Routing matches exactly (a,b); the post-combine output diverges (c); the logit
argmax flips (d).** So the divergence is in the reduce/combine, not the router.

**But the divergence is NOT bf16 associativity, and C is not the diverging side.**
Triangulating against the plain-MoE control **D**:

| pair | rel diff | meaning |
|---|---|---|
| **C vs D** | **0.003** | EP all-to-all ≈ plain MoE → EP reduce is at the bf16 floor (faithful) |
| **A vs D** | **0.767** | comm-free full replica is the OUTLIER (norm 5.12 vs 7.29) |
| A vs C | 0.539 | the measured divergence |

The comm-free full-replica path (A) computes a structurally *smaller* MoE output
than both the EP path and the plain reference, **with identical routing and
weights**. This is a structural difference in the comm-free path, not a summation-
order (bf16) effect.

## Step 2 — the FP32-accumulation fix: does it converge A→C?

FP32 accumulation in BOTH reduce paths (env `VLLM_SELF_SPEC_MOE_FP32_ACCUM=1`,
verified firing in every DP worker's `TritonExperts.moe_sum`):

| pair | bf16-accum | fp32-accum | Δ |
|---|---:|---:|---:|
| A vs C (MoE output rel) | 0.539 | **0.539** | unchanged |
| A_bf16 vs A_fp32 | — | 0.001 | fp32 moves A's local sum 0.1% |
| C_bf16 vs C_fp32 | — | 0.002 | fp32 moves C's reduce 0.2% |
| (d) argmax flips | 1/1 | 1/1 | still flips |

**FP32 accumulation leaves the A↔C divergence essentially unchanged.** Both sides
move <0.2% (the bf16-associativity component IS that small), but the 54% gap is
untouched — confirming the gap is structural, not associativity.

**End-to-end accept (the ground truth), DP=8 K=4:**

| config | accept_len | per_tok | per-position [p0,p1,p2,p3] |
|---|---:|---:|---|
| **A bf16** (comm-free) | **2.18** | 0.295 | [0.415, 0.293, 0.249, 0.224] |
| **A fp32** (the "fix") | **1.83** | 0.207 | [0.330, 0.197, 0.160, 0.142] |
| **C bf16** (EP verify) | **5.00** | 1.000 | [1.0, 1.0, 1.0, 1.0] (bit-faithful) |
| **C fp32** | **5.00** | 1.000 | [1.0, 1.0, 1.0, 1.0] (still bit-faithful) |

FP32-accum does **not** pull A toward C — it makes A **worse** (2.18 → 1.83). The
comm-free draft and the EP verify remain different computations; making each
"more accurate" in fp32 does not make them agree. C stays 5.0 (draft == verify ==
EP, fp32 keeps them matched) — confirming fp32-accum is lossless when the two
paths already share the structure, and useless when they don't (A).

## Step 3 — cost of FP32 accumulation

| config | gen_s (2048 tok) | tok/s | vs bf16 |
|---|---:|---:|---:|
| A bf16 | 15.89 | 128.9 | — |
| A fp32 | 20.54 | 99.7 | **−23%** |
| C bf16 | 5.13 | 399.0 | — |
| C fp32 | 7.29 | 281.0 | **−30%** |

FP32 accumulation is **expensive** (A −23%, C −30% decode throughput): the local
`moe_sum` upcast/sum/downcast is extra work every layer, and on the EP path the
cross-rank reduce moves fp32 (2× bytes). Since World A exists for a wall-clock
win, a −23% cost with no accept benefit (A), or −30% on the already-correct path
(C), is a clear net loss.

---

## Verdict

- **Is the divergence the bf16 reduce STRUCTURE?** No. The EP per-shard+cross-rank
  bf16 reduce is faithful to plain MoE (0.3%). The divergence is the comm-free
  full-replica path computing a structurally different (smaller) MoE output with
  identical routing — not a bf16 summation-associativity effect.
- **Does FP32-accum fix it losslessly + cheaply?** No on both counts. It leaves
  the MoE-output gap unchanged (0.539→0.539), does NOT recover accept (2.18→1.83,
  slightly worse), and costs ~23% throughput. FP32 accumulation is the wrong
  lever — the cause is structural, upstream of the bf16 reduce.

## Files
- Instrumentation: `vllm/model_executor/layers/fused_moe/num_divergence_dump.py`,
  hooks in `runner/moe_runner.py` + `logits_processor.py`.
- FP32-accum fix: `experts/triton_moe.py`, `fused_moe.py`,
  `topk_weight_and_reduce.py`, `distributed/.../cuda_communicator.py`; env in
  `vllm/envs.py`.
- Harness: `scripts/dump_moe.py`, `scripts/analyze_dump.py`,
  `scripts/analyze_all.py`, `scripts/accept_run.py`, `scripts/run_accept.sh`.
- Data: `data/dump_{A,C,D,A_fp32,C_fp32}/`, `data/acc_*.json`.
