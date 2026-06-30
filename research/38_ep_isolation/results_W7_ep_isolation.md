# W7 EP-isolation: which lever recovers the comm-free draft's accept at large EP

**Question.** World A's comm-free FULL-REPLICA draft accept COLLAPSES at large EP
(Qwen3-30B DP8 accept ~1.0, present even EAGER — phase 37 §4). The compile bug is
already fixed (`VLLM_SELF_SPEC_COMPILE_CONSISTENT=1`), so the residual divergence
is the comm-free draft summing its experts LOCALLY vs the verify's EP all-to-all
(an FP-reduction divergence that worsens with EP width). Before building either of
two multi-week levers, measure whether they recover accept:
- **L1 — EP-path draft:** route the draft through the EP path (resident-expert
  shard, matching the verify's per-resident reduction order) instead of the non-EP
  full replica.
- **L2 — verify-context-KV:** draft attends to the verify's clean KV, not its own
  drifted KV.

**Setup (no source changes).** Proxy `Qwen/Qwen1.5-MoE-A2.7B` (qwen2_moe, 60
experts top-4, 24 layers, non-MLA), **DP=8 → EP=8**, forced-PCIe
(`NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1`), greedy, **K=4**,
**bf16** (NO FP8 confound), batch 16, OUTLEN 128. Always
`VLLM_SELF_SPEC_DRAFT_FULL_CG=1` + `VLLM_SELF_SPEC_COMPILE_CONSISTENT=1` (compile
bug FIXED → the ONLY remaining divergence is comm-free-vs-EP).
`VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0`. draft_model self-spec, draft =
target. Harness `scripts/ep_isolation.py`, driver `scripts/run.sh`. Accept from the
engine spec-decode metrics; per-position from
`vllm:spec_decode_num_accepted_tokens_per_pos` (pos-0 rate = accepted-at-depth-0 /
num_drafts). All five runs error-free; GPUs torn down between runs (no leaked
workers; 0 MiB resident after).

---

## TL;DR (the verdict)

**Neither L1 (EP-path / quantized EP-shard) nor L2 (verify-context-KV) recovers the
comm-free draft. BUILD NEITHER as a fix for the collapse — they don't address the
cause.**

1. **L1 dead.** The EP-path draft (**B = 1.67**) is *not* better than the
   full-replica (**A = 2.18**) — it is **worse**. Routing through the EP shard does
   NOT avoid the FP divergence; it adds a coverage loss on top of it. So a
   quantized EP-shard + hot cache (L1) inherits the SAME comm-free-vs-EP divergence
   and gains nothing.
2. **L2 dead.** The rejection is **per-step at depth 0**, not accumulated KV drift.
   Config A rejects ~49–59% of the FIRST draft token (K=1 fresh-context pos-0 accept
   = **0.519**; K=4 pos-0 = 0.415). A clean verify-context KV cannot rescue a token
   that is already wrong at depth 0 from the local-vs-EP MoE sum.
3. **C is the only thing that works, and it is the thing we are trying to avoid.**
   The EP-full draft (real all-to-all = the verify) gives **accept 5.0, per_tok 1.0,
   per-position [1,1,1,1]** — perfect. This proves the divergence is *purely* the
   comm-free skip of the all-to-all (not the model, KV, or compile). Recovering
   accept requires the draft to DO the all-to-all — which is exactly the comm cost
   World A exists to remove.

---

## Test 1 — which draft routing recovers accept at DP=8 (isolates L1)

| cfg | flags | accept_len | per_tok | per-position accept rate [p0,p1,p2,p3] |
|---|---|---:|---:|---|
| **A** full-replica comm-free | `FULL_REPLICA=1 LOCAL_ROUTE=1` | **2.18** | 0.295 | [0.415, 0.293, 0.249, 0.224] |
| **B** EP-shard comm-free | `FULL_REPLICA=0 LOCAL_ROUTE=1` | **1.67** | 0.168 | [0.309, 0.160, 0.111, 0.091] |
| **C** EP-full (upper bound) | `FULL_REPLICA=0 LOCAL_ROUTE=0` | **5.00** | 1.000 | [1.0, 1.0, 1.0, 1.0] |

- **B < A** (1.67 < 2.18). The EP-path draft does **not** beat the full replica — it
  is worse. Verdict logic ("B >> A → build the quantized EP-shard") is **NOT met**.
  B is doubly penalized: it still skips the all-to-all (same FP divergence as A) AND
  it only routes to its ~7-8 resident experts of 60 (~12% coverage), so its router
  picks the wrong experts far more often than A's full-replica router.
- **C = 5.0, per-position all 1.0.** The real all-to-all draft is bit-faithful to the
  EP verify at every depth → the divergence is *exactly* the comm-free skip, nothing
  else (model, KV, compile, FP8 all ruled out — bf16 here, compile-consistent on).

**L1 verdict: do NOT build the quantized EP-shard + hot cache as a collapse fix.**
The EP shard is the worst of the comm-free options. A hot cache adds coverage, but
coverage is not the cause — A already has FULL coverage (all 60 experts) and still
sits at 2.18 because its LOCAL sum diverges from the EP all-to-all sum. More
resident experts → still a local sum → still diverges. (B is even an *upper* bound
on what an EP-shard+cache could route correctly, since at the resident shard B and
the verify use the same per-resident reduction; the divergence is the MISSING
non-resident contributions and the cross-rank reduction order, which a cache on top
of B cannot supply without the all-to-all.)

## Test 2 — per-step MoE divergence or accumulated KV drift (isolates L2)

| cfg | K | pos-0 accept rate | accept_len | reading |
|---|---:|---:|---:|---|
| **A** | 4 | **0.415** | 2.18 | 59% of first draft tokens already rejected at depth 0 |
| **A** | 1 | **0.519** | 1.52 | even with K=1 (fresh context, single draft/step) ~48% of FIRST tokens rejected |
| **C** | 4 | 1.000 | 5.00 | EP-full: every depth-0 token accepted |
| **C** | 1 | 1.000 | 2.00 | EP-full K=1 perfect |

- The collapse is **per-step at depth 0**, NOT accumulated drift. At K=1 the draft
  does exactly one forward per accepted token (no self-KV chaining within a step),
  on the verify's just-committed context, and **still ~48% of first tokens are
  rejected** for config A. That is the per-step local-vs-EP MoE divergence flipping
  the greedy token immediately.
- The depth profile decays ([0.42→0.29→0.25→0.22] for A) — there IS some additional
  per-depth loss from the draft chaining its own KV — but it sits *on top of* an
  already-large depth-0 rejection. Fixing only the deeper KV (L2) leaves the depth-0
  wall untouched: best case it would lift A's later positions toward A's own depth-0
  rate (~0.42), nowhere near C's 1.0.

**L2 verdict: do NOT build verify-context-KV as a collapse fix.** It targets
accumulated drift; the dominant loss is per-step at depth 0, which a clean KV does
not touch.

---

## The verdict: which lever(s) to build

**Build NEITHER L1 nor L2 to fix the comm-free accept collapse.** Evidence:
- **L1 (quantized EP-shard + hot cache):** the EP path (B=1.67) is worse than the
  full replica (A=2.18), not better — the EP shard inherits the same comm-free FP
  divergence and adds coverage loss. The cause is the LOCAL sum vs the EP
  all-to-all, which neither sharding nor a hot cache removes.
- **L2 (verify-context-KV):** the rejection is per-step at depth 0 (K=1 pos-0 accept
  0.519 for A), not accumulated KV drift, so a clean verify KV cannot rescue it.
- **The only recovery is C** — the draft performing the real all-to-all, which *is*
  the EP comm World A is built to avoid. There is no comm-free routing of this draft
  that reproduces the EP verify's reduction; the divergence is the all-to-all reduce
  itself.

**Implication for the project.** A net DP=8 speedup from a comm-free self-spec draft
is blocked by a *structural* identity: the verify's MoE output is the EP all-to-all
reduction, and any draft that skips that all-to-all computes a different (locally
reduced, and for B partially covered) sum, whose greedy token diverges immediately.
The lever that would actually help is not on the draft-routing axis at all — it is
**making the comm-free local sum numerically reproduce the EP all-to-all reduction**
(e.g. a reduction-order / accumulation-precision match, or accepting in a
tolerance-relaxed verifier), or moving to a regime where the comm itself is cheap.
The two proposed levers (L1, L2) do not touch this identity.

### Anomalies / notes
- No OOM; all configs ran (full-replica draft + EP verify fit at DP=8 on 80 GB,
  ~21 GiB/GPU resident).
- **Proxy caveat (magnitude, not direction):** on this 60-expert top-4 proxy, A
  collapses to 2.18 at DP=8, milder than Qwen3-30B's ~1.0 (128 experts top-8 → more
  summed terms, larger FP divergence, lower resident coverage fraction effect). The
  *direction* (B<A<<C, depth-0 per-step rejection, C=perfect) is the load-bearing
  result and is unambiguous; the 30B would only make A and B worse relative to C,
  strengthening the same verdict.

## Files
- `scripts/ep_isolation.py` — single-config (A/B/C) DP=8 harness; captures
  accept_len, per_tok, per-position accept vector; bf16, FULL_CG +
  COMPILE_CONSISTENT, self-teardown.
- `scripts/run.sh` — serial driver (A/B/C K=4 + A/C K=1), own-worker teardown
  between runs.
- `data/ep_iso_t1_{A,B,C}_dp8_K4_cg.json`, `data/ep_iso_t2_{A,C}_dp8_K1_cg.json` —
  per-run results (incl. emitted tokens).
- `logs/` — per-run engine logs + `sweep.log`.
