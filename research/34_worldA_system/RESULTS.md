# World A — Training-Free, Comm-Free, Lossless Self-Speculative Decoding for Communication-Bound MoE-EP Serving

**Consolidated results, Phases 34–47.** This document ties the system build and the
debugging/optimization arc into one story, with the final measured numbers. Companion
per-phase docs live under `research/34_worldA_system/` and `research/{35..47}_*/`.

---

## TL;DR

World A skips the expensive expert-parallel **all-to-all** on the speculative *draft* pass
(the draft holds a full replica of the experts and routes locally, communication-free),
while the *verify* pass stays full-EP and exact — so the output is **losslessly** the
verifier's distribution. On a fully-optimized stack (Qwen3-30B-A3B, DP8/EP8, forced-PCIe),
the comm-free self-spec is:

- **Lossless** — draft issues **zero** real collectives; output byte-identical to the verifier.
- **A wall-clock win across the communication-bound regime** — measured **1.03–1.30×** at
  small/mid batch on a single node, **rising monotonically with the communication fraction `f`**,
  and **projected ~1.6–2.1× at realistic multi-node `f≈0.6–0.8`**.
- **Tracking the cost model** to within ~5–20% at low `f`, validating it in its intended regime.

The central finding of the arc: **every "fundamental limit" we hit along the way turned out to
be a fixable implementation artifact, not physics.** The honest negatives that survive are (a) the
win is a function of the deployment's communication fraction (parity when comm is free, e.g. NVLink;
a growing win as comm dominates), and (b) the full replica has a **memory** cost at very large scale
that needs a bounded cache (W2b) — a memory constraint, not a speed one.

---

## 1. Mechanism (W0–W2)

- **W0 — self-spec via the V1 `draft_model` path.** The draft is a full second instance of the
  same model with its own KV; the framework provides lossless rejection sampling, accept/commit,
  draft-KV rollback, and cudagraphs. Env: `VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE`.
- **W1 — comm-free local routing.** On the draft forward, the MoE skips the dispatch/combine
  all-to-all (`all2all.py`); each rank routes its own tokens to locally-resident experts. Verified
  by the collective counter: draft `real=0`, verify `real>0`.
- **W2a — full replica.** The draft is made non-EP (`enable_expert_parallel=False`) so every rank
  holds **all** experts and routes to the full top-k locally → full coverage → high acceptance,
  comm-free by replication. Env: `VLLM_SELF_SPEC_DRAFT_FULL_REPLICA`.

**Losslessness** is by the rejection-sampling theorem: the exact full-EP verifier corrects any
draft, so the served distribution is the verifier's regardless of how lossy the draft is. (Greedy
is bit-exact modulo the pre-existing batched-verify MoE FP non-associativity documented in W0/B1,
which affects any parallel verifier.)

---

## 2. The cost model (Phase 24 / C1)

Measured on the forced-PCIe engine:

```
T_verify(T)  = 22.7 + 0.0495·T    ms   (full-EP verify, communication-bound)
T_compute(T) = 15.3 + 0.0152·T    ms   (comm-free forward)
speedup      = E_tokens(β,k)·T_verify / (k·T_compute + T_verify)
```

Predicted **~1.2–1.35×** single-node, rising toward `accept_len` as the communication fraction
`f` grows. With the draft at its compute floor this reduces to the clean form

```
speedup = accept_len / (k·(1−f) + 1)      →   ~1.0× at f→0 (NVLink) ;  →accept_len at f→1 (multi-node)
```

`f` is a **machine constant** (fabric bandwidth ÷ compute throughput, and model arithmetic
intensity), *not* a batch knob: batch reveals `f` (it plateaus ~0.62 on forced-PCIe once GEMMs
saturate), it does not raise it. Higher `f` comes from a slower fabric (real inter-node IB) or
wider EP — the natural direction for large-model serving.

---

## 3. The measurement arc — every "fundamental limit" was a fixable artifact

The initial end-to-end number was **0.42×** (a loss), and the story of Phases 34–47 is peeling
back six successive "walls," each of which turned out to be an implementation bug or a
mis-attribution rather than a property of the design.

### 3.1 "Fundamental loss (0.42×)" → the draft ran **eager** (W7-MB/FG)
The draft never received cudagraphs (`uses_draft_model()` was skipped when initializing cudagraph
keys). It ran fully eager (~46 ms/forward vs ~12 ms captured). FULL cudagraph on the draft →
recovered most of the loss on V2-Lite.

### 3.2 Accept collapse under FULL-CG → cudagraph-capture correctness (W7-FG2/FA/GQA)
Capturing with `attn_metadata=None` produced a zero-fill attention stub (MLA); the FA3/GQA decode
kernel froze `seqused_k` at capture, wrong for the draft's growing sequence. Fixes: build real
capture metadata (MLA); run FA3 draft attention correctly. Also fixed a DP=8 engine crash.

### 3.3 "Structural comm-free-vs-EP accept collapse that worsens with EP width" → **two config bugs** (W7-DP/EP/NUM/REPLICA)
This looked exactly like a physics law: acceptance fell 4.88 (DP2) → 2.18 (DP8), and neither
FP8, bf16-reduce-order/FP32-accumulation, nor verify-context-KV moved it. It was **not** physics.
The **"full replica" was never a replica**:
- **Bug A** — the `enable_expert_parallel=False` override was silently dropped (the draft was
  built from the target config, and FusedMoE read the global config), so the draft was **EP-sharded
  to 7–8 of 60 experts at DP8** — coverage that shrank precisely with DP.
- **Bug B** — even after A, the non-EP MoE in a pure-DP group flatten-TP'd to `tp_size=dp_size`,
  sharding each expert and relying on an all-reduce that is a **no-op** with no TP group → each rank
  emitted a **1/N-sharded** MoE output.

The genuine replica (`tp=ep=1`, all experts resident and fully computed) makes the draft MoE output
**bit-identical to plain MoE (rel-err 0.0)** and acceptance **flat ~4.9 across DP2/4/8** — the
"collapse that worsens with EP width" vanished entirely. *(We rejected the `temperature` workaround
here as a dodge; that insistence is what forced the numerical root-cause hunt that found the bugs.)*

### 3.4 Draft/verify divergence at scale → `COMPILE_CONSISTENT` (W7-CC)
The compiled draft diverged from the compiled verify due to **batch-shape-variant kernels** (draft
at decode shape ~1 tok, verify at 1+K). `VLLM_SELF_SPEC_COMPILE_CONSISTENT` forces batch-invariant
kernels so draft ≡ verify, both compiled. Single-GPU accept 1.57 → 5.00.

### 3.5 "Draft-compute bound" → the chain ran **fully eager** → **PIECEWISE** (W7-RECON/GRAPH)
A term-by-term profile showed the in-chain draft forward was **65.4 ms = 4.15× `T_compute`** — but
the accept-achieving draft is not fundamentally expensive: the **captured step-0 forward is 18.5 ms
≈ `T_compute`**. The chain was running `CUDAGraphMode.NONE` (fully eager) because the FA3 fix needed
eager *attention* — but it threw away the compiled *body* too. Normal decode uses **PIECEWISE**
(attention eager, body captured). Switching the draft chain NONE → PIECEWISE
(`VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE`) dropped the in-chain forward **65.4 → 14.5 ms (4.5×)**,
byte-identical output. **This is the single change that flips the thesis from loss to win.**
(The FA3 replay-safe full-graph kernel — "the wall" — is a documented **no-go**: PIECEWISE already
lands *below* the captured step-0, so there is nothing left for it to recover.)

### 3.6 "Prune experts to cheapen the draft" → **no-go** (the g-floor) (W7-TOPC)
A smart top-C-by-gate-weight prune saves too little: even C=1 cuts the draft forward only ~34%
(Qwen3-30B) / ~14% (Qwen1.5-MoE), because at decode the draft is dominated by **attention / router /
dense / norm**, which pruning doesn't touch. Best projected net **+1.9%** (within noise), 0% at high
`f`. The full replica is the right *compute* draft; expert reduction is only justified for **memory**
at scales too large to replicate (W2b).

**Also confirmed en route:** CPU orchestration is *not* a lever — the recon's "31 ms" was a
subtraction artifact; the true self-spec-scoped CPU is ~1.7 ms, the rest being the general EngineCore
scheduler + the DP sync barrier (W7-CPU). And **K=2 is optimal** (K=4's extra forward costs more than
its marginal acceptance; W7-KRETUNE).

---

## 4. The win curve (Phase 47 — the headline)

Qwen3-30B-A3B, DP8/EP8, forced-PCIe, FP8 full-replica comm-free draft, **fully optimized**
(genuine replica + PIECEWISE + local-route + FULL-CG + compile-consistent), **K=2**, greedy.
Speedup vs no-spec, sweeping emulated exposed A2A per collective:

| A2A µs/coll | batch 32 | batch 64 | batch 128 |
|---:|---:|---:|---:|
| 0 (native)  | **1.16×** | **1.03×** | 0.72× |
| 100         | 1.20×     | 1.08×     | 0.81× |
| 250         | 1.23×     | 1.13×     | 0.93× |
| 500         | 1.25×     | 1.17×     | 0.96× |
| 1000        | **1.30×** | **1.24×** | **1.14×** |

- **Small/mid batch wins everywhere, including native** (no added comm); large batch is compute-bound
  at low comm and crosses into a win in the comm-bound regime (~616 µs measured).
- **Monotone rising with communication**, as the model predicts; `accept_len` flat **2.88–2.91**.
- **vs pre-piecewise** (Phase-42, K=4, plateau 0.82×): piecewise bought **+0.36 to +0.84 absolute**
  speedup; the b64 native point went **0.36× → 1.03×**.
- **Conservative lower bound:** the genuine replica issues **zero** real collectives, but the
  benchmark's emulated A2A sleep is not gated by the local-route flag, so the draft is *charged*
  communication it never performs. The **shielded** curve (verify-only delay) has b32/b64 winning
  everywhere and reaches **2.18 / 1.95 / 1.46×** at 1000 µs.
- **Cost-model tracking:** measured is within ~5–20% of `accept_len/(K(1−f)+1)` at low `f` (and
  slightly exceeds it at `f=0`, the FP8 draft being cheaper than the formula assumes).
- **Multi-node projection** at realistic **`f≈0.6–0.8`**: **~1.6–2.1×** (cost-model) /
  **~1.15–1.95×** (measured-shielded) — the deployment win.

---

## 5. The honest ledger — real vs artifact

| Claim during the arc | Verdict | Reality |
|---|---|---|
| 0.42× "fundamental loss" | **artifact** | draft ran eager; FULL-CG fixes it |
| Accept collapse worsens with EP width (structural) | **artifact** | two config bugs; genuine replica → flat ~4.9 |
| Comm-free-vs-EP FP divergence is the wall | **artifact** | it was the sharding bug, not bf16 reduce order |
| Draft-compute bound | **artifact** | fully-eager chain; PIECEWISE → `T_compute` |
| FA3 replay-safe full-graph needed | **no-go, unneeded** | PIECEWISE already below captured step-0 |
| Prune experts to win | **no-go** | g-floor: pruning saves ~≤34%, costs accept |
| CPU orchestration is 27% of the gap | **mis-attribution** | true self-spec CPU ~1.7 ms |
| **Win requires heavy comm** | **real** | parity at f→0; grows to ≈accept_len at f→1 |
| **Full replica costs memory at scale** | **real** | needs bounded hot cache (W2b) for huge models |

---

## 6. Implementation (the artifact)

All changes are **env-gated, default-off → byte-identical**, merged on `research/self-spec-moe`:

- `VLLM_SELF_SPEC_DRAFT_LOCAL_ROUTE` — comm-free local routing on the draft (W1).
- `VLLM_SELF_SPEC_DRAFT_FULL_REPLICA` — non-EP full-replica draft, `tp=ep=1` (W2a + Bug A/B fixes).
- `VLLM_SELF_SPEC_DRAFT_FULL_CG` — FULL cudagraph draft decode.
- `VLLM_SELF_SPEC_COMPILE_CONSISTENT` — batch-invariant kernels so compiled draft ≡ verify.
- `VLLM_SELF_SPEC_DRAFT_CHAIN_PIECEWISE` — **the win**: chain runs PIECEWISE (attention eager, body
  captured) instead of fully eager.
- `VLLM_SELF_SPEC_DRAFT_TOPC` — draft top-C expert prune primitive (probe; reused by a future W2b
  memory loader).
- `VLLM_SELF_SPEC_FAST_PARSE`, `VLLM_SELF_SPEC_PROFILE(_FINE)` — orchestration D2H hygiene + profilers.

Recommended operating config: `FULL_REPLICA + LOCAL_ROUTE + FULL_CG + COMPILE_CONSISTENT +
DRAFT_CHAIN_PIECEWISE`, `K=2`.

---

## 7. Limitations & future work

1. **Real multi-node validation.** The single node can only *emulate* the comm-bound regime (A2A
   delay). The gold-standard number is a genuine inter-node IB run at `f≈0.6–0.8` (projected 1.6–2.1×).
2. **Benchmark shielding.** Gate `_emulate_exposed_a2a_delay` by the local-route flag so the measured
   curve equals the (higher) shielded curve rather than a conservative lower bound.
3. **W2b bounded cache — for memory, not speed.** Models too large to replicate need a resident
   globally-hot top-C cache; the `DRAFT_TOPC` primitive + a custom loader realize it. The accept cost
   is the price of *fitting*, not a speed lever (Phase 46).
4. **FA3 replay-safe attention kernel — documented no-go.** Would only fold the already-cheap eager
   attention into a full graph; PIECEWISE is at/below the captured floor, so no payoff.

---

*Every merged change is lossless and env-gated; the default vLLM path is unchanged. The result stands
on the merged implementation + the measured win curve, reproducible via
`research/47_sweep2/scripts/run_sweep2.sh`.*
