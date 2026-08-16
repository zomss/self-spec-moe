# Self-MoE-Spec Research Status

Date: 2026-08-16 (current state); 2026-08-15 (executive summary); the phase table and
all sections from "Phase Summary" downward are the historical rollup last
revised 2026-06-25 and are NOT amended.

## Executive Summary (2026-08-15)

**The project has pivoted three times. The name is now a historical
artifact: the work is neither MoE-specific nor about communication.**

The current thesis is **"No Universal Draft"** (Phase 79 onward): for
**training-free, distribution-preserving self-speculative decoding**, the
optimal draft lever is **keyed to the deployment** — architecture x scale x
kernel realization x regime (batch x context x content) x hardware — and
choosing wrong costs **30-60%**. Four lever classes are in scope (weight
quantization, KV window, layer skip, KV-cache quantization); trained
drafters, relaxed acceptance, and tree width are explicitly out.

Four contributions, all CLOSED or near it (`paper/arxiv/STATUS_CONTRIBUTIONS.md`):

* **C1 — there is no single lever.** Five distinct surfaces across
  architectures. Dense reaches **1.91x** (b32/16k, W4+win512 K6 on the
  floor-free chain, delivering ~100% of its registered roofline) and
  **2.77x** at b16/32k. But **MLA loses under every lever — 0.56x at an
  acceptance of 5.9/6**, a pure cost loss at near-perfect acceptance — and
  MoE tops out at 1.15x, where the composition that wins on dense scores
  0.63x. One global lever across a mixed fleet leaves **+39.1%** on the
  table.
* **C2 — search, as a measured theorem.** Acceptance importance **cannot be
  scored, only measured**: five proxy classes were falsified, with angular
  importance actually *inverted* (rho = -0.768). What survives is a bounded
  measurement budget — cost O(#levers + #confirmations), ~21 boots, **~1.5%
  of exhaustive**, 2-4 GPU-hours to onboard a new column — reaching regret
  **0.46-1.61%** against an oracle.
* **C3 — a switching system that fails closed.** It arms where measurement
  says it pays and refuses where it cannot know, which is why MLA and MoE
  come out gated rather than forced.
* **C4 — demonstrated on both** ordinary serving (**1.80x** wall aggregate,
  1.90x at b16 on reasoning shapes) and RL rollout.

### What was retracted on the way here

The previous summary described a **communication-bound MoE** thesis with a
"validated ~1.2-1.3x single-node PCIe speedup". **That is retracted.** The
number came from a *forced-PCIe emulation*; on real two-node fabric the same
system measures **0.29-0.49x** — a large loss (Phases 52, 53, 61). The
companion belief that "the bigger win lives inter-node" was **falsified**:
cutting to one NIC per node changed the step by only 5-25% at 236B, because
the decode all-to-all is latency- and overhead-bound, not
wire-bandwidth-bound, on 400G-class NICs. **World A is closed as a method on
real hardware of this class** (Phase 61), and `research/paper_outline.md`
and `research/FINAL_REPORT.md` describe that superseded paper.

The redirect came from re-reading the literature (Phase 58): the negative
was regime-specific, and **long context** — KV-bound at any batch — was the
untested win regime. That produced window-KV drafting (Phase 62: per-token
acceptance **0.957 at W=512, a 30x KV read reduction**) and the lever-vs-regime
contradiction (Phase 75) that became C1.

**Numbers predating August 2026 should be read with care.** Phase 96 found
that under natural EOS the speculative and autoregressive arms drain
asymmetrically, contaminating any e2e ratio at batch > 1 on heavy-tailed
regimes; re-measuring under equal work moved MoE from 8/33 wins to 5/33 (one
cell 1.288 -> 0.954). Three further defects were fixed in the same phase — an
autotune boot lottery worth 40% run-to-run, a decode-vs-wall currency
mismatch, and a compile-cache key collision — and Phase 84 retracted a lever
whose gate was circular.

### Where the work is now

**Phase 98** demonstrates the two-round selector end to end and has now
measured EVERY registered claim across four scored campaigns — see "Current
state" immediately below. Its D3 result also prices the open engineering
question: per-regime switching beats every static configuration by the
registered margin in only 7% of workload mixes, so **Phase 97**'s
composition runtime — still unresolved, its B0 value screen still unscored —
is buying less than it was assumed to. The live paper is `paper/` (`c1.md`,
`c2.md`, `c3.md`, `arxiv/`); the open gaps are the hardware axis (measured
on one box) and C3's draft status.

---

### Historical framing (superseded, retained for the record)

**The project pivoted at Phase 18.** The original naive design --

```text
local top-k expert draft + exact full-MoE verification
```

-- is a **scoped negative result** (Phases 00-17): device-local routing never clears
`beta >= 0.8` on real models and is *anti-correlated* with the high-EP regime that
supplies the communication advantage. Single-node NVLink experiments were weak
because NVLink makes EP **compute-bound** (machine-balance argument below) -- a
fabric artifact, not a fundamental limit.

From Phase 18 the draft mechanism is **quantization, not lossy local routing**, and
the framing is **communication-bound MoE serving**. As of Phase 24 the direction is
**validated end-to-end on this hardware**: every lever's acceptance cost is measured,
and the benefit is **measured on real PCIe** (Phase 24 3e): forcing the EP all-to-all
over PCIe-SHM (`NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1` -> "via
SHM/direct", no NVLink/NVLS/IB, no hang), the real comm fraction is f=0.37-0.65
(growing with batch). **Corrected speedup (Phase 27): ~1.2-1.3x across batch** -- the
3e-3g composition under-counted the verify cost (the spec verify is one forward over
B*k tokens, so its comm-bound all-to-all scales with k; using the 1-token verify cost
was optimistic). With correct scaling the comm-free draft (~23ms) nearly equals the
comm it saves (~25ms), so even perfect acceptance caps the single-node win at ~1.35x.
Modest, lossless. The PCIe-exact gap is CLOSED on this box; the bigger win lives
inter-node (IB, needs a multi-node box). Tree drafting (Phase 27) helps only at LOW
batch (latency; small balanced tree, +8% over chain), not at serving batch. The
thesis, design ladder, and measured-vs-open status are in "Current Positioning" below;
the phase table records the full path. "Historical Record" documents the pre-pivot
(Phases 00-17) investigation and is superseded by the positioning section.

## Current state — Phase 98, two-round selector (2026-08-16)

> The executive summary above and the phase table below are the **historical
> rollup through Phase 27** (last revised 2026-06-25). They are not amended
> here. This section records where the work actually stands; the phases
> between are documented in their own folders.

Phase 98 demonstrates the two-round selector end to end: Round 1/2 predict
COST from single-lever profiles and eliminate soundly, Round 2's successor
measures ACCEPTANCE where it must, and D3 puts the composed selector against
the omniscient ceiling and every static alternative. **Every registered
claim of the phase is now measured** — four scored campaigns, where the
phase previously had none.

* **G98-C (Round 2 cost, D1'): SCORED.** 47/48 held-out (cell, regime) pairs
  covered (**97.9%**), median relative error 0.53%, and all six regimes
  resolvable — including R5/R5cot, which Round 1 reported NOT RESOLVABLE.
  The second clause was scored separately: the sound elimination rule
  `(K+1)/q_lo < 1.015` **fired five times and was correct every time — zero
  false eliminations, non-vacuously**, where Round 1 had reported the rule
  NOT EXERCISED for want of a scored surface.
  `98_selector_demo/results_g98_c.md`.
* **G98-D (acceptance, D2a/D2c): SCORED.** The product bound
  `f_set >= prod f_i` is violated in **6 of 132 rows, and every violation
  carries window = 128** stacked with layer skipping at short context.
  Composition is otherwise CONSTRUCTIVE — the interaction ratio's median
  exceeds 1 for every lever set and reaches **1.670 at R4**, because a
  window that already discards the context removes what layer skipping
  would have degraded. The failure direction is benign: an over-optimistic
  screen admits too much and confirmation catches it; it cannot falsely
  eliminate. D2(c) resolves the u-axis in 107/264 adjacent bucket pairs.
  `98_selector_demo/results_g98_d.md`.
* **D2(b) (knapsack identity): SCORED, fails by one control at one count.**
  The knapsack beats **11 of 12** count-matched controls, winning decisively
  at k=4 and k=8 and inverting at k=16. The finding is where it inverts: the
  knapsack's own objective, the product of per-layer retentions, ranks every
  arm **exactly right at k=4 and k=8 (Spearman +1.000 at both)** and then
  fails at the top at k=16, where its predicted-best set (product 0.585, by
  far the highest) measures tau 1.244 against the frozen set's 2.080. So
  additive per-layer measurement identifies BAD sets at every count and
  stops identifying the BEST set once the count is aggressive — **Phase 90's
  non-additivity theorem reproduced on a new lever, model and box, with the
  boundary now located between 8 and 16 layers of 36.** Also measured: the
  first block is worth almost the whole draft (skipping layer 0 alone drops
  tau 8.083 -> 1.244), and k=16 is outside the lever's usable range for
  acceptance regardless of which layers are chosen.
  `98_selector_demo/results_d2b_knapsack.md`.
* **D3 (end-to-end selector value): SCORED, one clause each way.** On h103,
  62 boots. The selector reaches **95.1% of the omniscient composite**
  (94.8% under the legacy instrument), clearing the registered 90% target,
  and runs **1.371x over static-OFF**. It fails the +2% clause against a
  single composed static, which it leads by only 1.4% — though it passes
  against every single-LEVER configuration at 1.141x, and which clause
  applies turns on a reading of the preregistration. Its one large error is
  C3-shaped: at R4 speculation loses outright (OFF 684.9 against its armed
  pick 522.3) and it armed anyway, in the regime whose predicted margin was
  already the thinnest in the map.
  `98_selector_demo/results_g98_e.md`.
* **The frontier is the number that sizes Phase 97.** Across 126 workload
  mixes the 90% target holds in 112 (89%), but the selector beats every
  static by +2% in only **9 (7%)**. On almost every mix one well-chosen
  static configuration lands within 2% of per-regime selection — the Phase-82
  dwell-time law restated in another currency. Speculation itself clearly
  pays; SWITCHING between compositions is what looks thin, and that argues
  for pricing the value case before building more of the runtime engine.

**Measurement environment.** Both campaigns run on **h104 (bare metal)**.
The previous box was a QEMU/KVM guest whose host-side "clamp" — a fixed
+4-5 ms per engine step, invisible to every in-guest instrument — gated 40+
campaign attempts without a single accepted cell. It is localised, not
closed: the engine makes **~160k clock reads per batch-1 step**, so a
+40-66 ns per-read slowdown reproduces the whole clamp, making pvclock read
cost the prime suspect. The verdict needs one probe run on that box.
`98_selector_demo/results_clamp_investigation.md`.

## Phase Summary

| Phase | Status | Main Finding |
| --- | --- | --- |
| `00_proposal` | Complete | Idea is plausible and not clearly duplicated, but the viable regime is narrow. |
| `01_communication_advantage` | Complete | Communication-disabled draft only helps when exposed all-to-all is large. |
| `02_timing_envelope_experiment` | Complete for local/emulated setting | Single-node EP is weak; forced IB is unstable; delay emulation is available. |
| `03_local_routing_acceptance` | Initial real-model tests complete | Real router coverage and next-token proxy are weak for tested models. |
| `04_expert_placement_acceptance` | Complete | Optimized placement + hot replication helps only modestly; top-1 locality is high but full top-k locality is low. |
| `05_top1_local_draft` | Complete | Cheaper top-1/2/4 local drafts still weak; best ~0.40 sampled acceptance on Qwen1.5-MoE. |
| `06_local_draft_policy` | Complete | Simple confidence/locality gating cannot isolate a high-acceptance subset. |
| `07_recent_model_acceptance` | Complete | Recent models better at low EP but collapse at high EP; acceptance and communication advantage are anti-correlated. |
| `08_negative_result` | Complete | Negative result scoped to no-shared-expert local-only drafting. |
| `09_rebalancing_ceiling` | Complete | Static replication needs ~0.5-0.8E experts per device for beta >= 0.8. |
| `10_temporal_locality_cache` | Complete | Verify-warmed/request-local cache halves the needed cache, but still scales with EP. |
| `11_intranode_ep_draft` | Complete | Hierarchical draft acceptance is feasible; timing needs high inter-node A2A fraction. |
| `12_ep_width_latency` | Complete | Single-node NVLink EP-width changes barely help; benefit is inter-node-bound. |
| `13_affinity_placement_gated_draft` | Complete | Affinity/request scheduling gives high G=2 acceptance but no draftable G=4 subset. |
| `14_intranode_semantic_pattern` | Complete | Semantic-Parallelism-like locality pattern is visible at G=2, but not an end-to-end speedup. |
| `15_replication_phi_envelope` | Complete | EPLB/cache replication makes G=4 viable (latency-bound, monotonic in phi); Qwen3 ~1.18x at R=2x iff inter-node A2A >= ~200us/coll. |
| `16_intranode_throughput` | Complete | Local draft halves MoE-FFN weight read at serving batch; real vLLM decode pins phi_moe~0.65-0.76 and T_draft/T_full~0.63; end-to-end throughput gain ~1.07x (b=0.85) to ~1.12x (b=0.9) at k=2, break-even by k=4. |
| `17_layerskip_local_draft` | Complete | Layer-skip is a bad trade: acceptance drops super-linearly vs linear cost saving, so layer-skip+local never beats pure local (~1.0x). Cheap-draft-via-skip lever fails. |
| `18_quantized_expert_throughput` | Complete | Serve-FP8 is ~2.2-3.4x but LOSSY. The lossless use is an FP8 draft + bf16 verify: FP8 is a near-perfect draft (acc 0.954 vs bf16) at 0.29x cost -> ~1.9x LOSSLESS throughput (the cheap draft layer-skip never was). Local routing is unnecessary under FP8. Cost: ~1.5x expert memory. FP8 gain peaks at moderate batch (~3.4x@B64), ~1x@B1. |
| `19_multinode_plan` | Plan | Novelty delineation: QuantSpec (ICML25) covers quantized-draft for KV/memory; SS-MoE covers expert-subset on-device. Ours = inter-node all-to-all amortization (local draft + full-EP verify), provably outside their scope. Specifies the 2-4 node go/no-go experiment. |
| `20_injection_emulation` | Complete | Real-IB loopback not viable here (fabric refuses NIC->NIC hairpin + self-loopback); emulate inter-node A2A via GPU-stream latency injection instead. Measured (Qwen3, attention-DP4+EP): ~145 exposed collectives/step, f=0.58-0.84 at 163-320us/coll. Break-even exposed latency is only ~11-18us/coll (3.3us for FP8-beta) -> lossless 1.1-3.3x across the plausible exposed-after-DBO range. Generalizes: GPT-OSS-20B (24L, MXFP4) -> N~70 (~3/layer, structural), f=0.55-0.72, speedup 1.4-2.3x -> win scales with MoE depth. Remaining gate: DBO overlap residual (needs 2-node IB), now with threshold d*~15-35us. |
| `21_streaming_locality` | Complete | DRAM-offloaded expert draft (keep GPU at ~bf16 verify shard, stream quantized experts) is PCIe-viable at low batch. Real-weight routing is skewed (Qwen3 top-25% experts = 60% of traffic) and temporally local (B=1, k=4 cycle touches only ~20/128 experts/layer; reuse 1.6-3.1x). With a 12.5-25% verify-warmed cache, per-cycle streamed bytes fall BELOW the 50 GB/s hideable budget k*T_step*BW (Qwen3 C=16,k=4: fit 0.52; C=32: 0.27), holding extra HBM to ~2-5 GB vs +14.5 GB to replicate. Mid/high batch needs larger cache or skip-cold (verify corrects, no stall). beta near 0.95 where fully hidden, graceful to ~0.78 floor otherwise. Confirmed on GPT-OSS-20B too. Overlap-can't-hide-PCIe holds only for full-set swaps, not this reuse-amortized partial stream. |
| `22_fp4_acceptance` | Complete | FP4 expert-draft acceptance holds (the bit-width the per-node-replication memory math requires). Real Qwen3-30B, weight-only W4A16 expert quant, rejection-sampling acc vs bf16 target (Phase 18 method; FP8 anchor 0.967 ~ Phase 18's 0.954). NVFP4 (E2M1+E4M3/16 scale) = 0.921 (best FP4), MXFP4 (E8M0/32) = 0.905, INT4-g128 = 0.901 -- all ~0.05 below FP8, far above the local floor (~0.78). Maps to ~2.0-2.1x lossless at the Phase 20 high-exposure point (vs FP8 2.3x). So FP4 is simultaneously small enough to replicate per node AND accurate enough to draft -- acceptance side of the intra-node-EP direction validated. NVFP4 = Blackwell-native target; MXFP4 = ~1.5pt-lower Hopper fallback. Caveat: one-step W4A16 proxy, one model. |
| `23_activation_quant_acceptance` | Complete | Activation quant (the COMMUNICATION axis -- the EP all-to-all moves activations) buys comm reduction almost for free. Real Qwen3-30B, per-token fake-quant of dispatch input + combine output in Qwen3MoeExperts, rejection-sampling acc vs bf16 (w4a16 anchor 0.913 ~ Phase 22, validates patched forward). FP8 activations FREE (W4A8 0.915 = W4A16 0.913 within +-0.01 noise -> 2x comm at ~0 cost); NVFP4 activations cost ~0.014 (W4A4 0.899 -> 4x comm); both memory-free. MXFP4 activations worse (0.866) -> use NVFP4 on the wire. Weight+activation FP4 errors barely compound. Design ladder: FP8 act (free, 2x) -> NVFP4 act (~0.014, 4x) -> local routing (eliminate, +HBM). Quantization unified across compute (weights) AND communication (activations), lossless via verify. Cost side of all levers now measured; remaining gap = comm-bound f-vs-batch BENEFIT (PCIe/inter-node). |
| `24_commbound_throughput` | **Stage A complete (GO)** | The make-or-break BENEFIT side, now measured on real transports (not injection). `NCCL_P2P_DISABLE=1` (the PCIe-exact knob) HANGS this NVSwitch box -- proven on a trivial 2-GPU all-gather (same wall as Phase 02/20 forced transports). Forced sockets work as a real comm-heavy endpoint. Result (Qwen3-30B, attention-DP8+EP): NVLink step ~15 ms flat (compute, comm ~free); off NVLink (sockets) the step is **84-89% communication** (f=0.84 at global B=8 -> 0.89 at B=128, growing with batch -- machine balance confirmed). Composed lossless speedup of a comm-free local-routing draft + exact verify (S_draft~=S_nvlink, S_verify=comm-bound, measured beta): **>=1.3x across the whole plausible PCIe range** (socket comm deflated 4x, beta=0.82) up to **2-3.5x** at the measured socket point; 1.6-1.9x at beta=0.92; win grows with batch. GO. **Stage B1 done** (`results_stageB.md`): lockstep local-draft/full-verify cycle implemented; real multi-token acceptance at 0.5E gives implied beta~0.80 = Phase 18's one-step 0.82 (validates the one-step-beta -> T(k,beta) composition); losslessness holds by the rejection-sampling theorem -- greedy bit-exact matched plain bf16 in 11/12 cases, the lone miss proven (control) to be batched-vs-sequential GPU float non-associativity in the MoE (a single batched forward over a known plain-greedy sequence itself flips ~1/64 near-tie positions; affects ANY parallel verifier), not the accept logic. **B1 k-sweep** (`stageB_ksweep.py`): measured REAL multi-token accepted-length at k=2..12 -- validates the geometric k-curve (real/geometric = 1.00 to k=4, 0.97 at k=6-8, 0.90 at k=12; per-position acceptance ~flat at 0.85, implied beta drifts only 0.85->0.83). Real-data k-curve (8-GPU PCIe B=512) peaks at k=4 -> 1.45-1.66x (beta~0.85 local draft) -- so the k-curve is real-data-grounded in the optimal region, not just modeled. **Stage B2a done** (`scope_B2.md`, results in `results_commbound_throughput.md` 3b): added `VLLM_SELF_SPEC_SKIP_A2A` (shape-preserving local stand-in for the MoE EP collective; tiles the local chunk -> valid expert ids, no cross-rank comm; timing only). Measured the real comm-free draft step on the socket engine: **18.6-19.7 ms = 1.25-1.30x the NVLink compute floor** (skip removes 80-86% of the step = the comm), passing the GO gate (<=1.3x) and validating Stage A's NVLink draft proxy on the real engine. Recomposed lossless speedup with the measured draft: 1.9-2.3x (beta 0.82) / 2.5-3.1x (beta 0.92), growing with batch. **Stage B2b-pragmatic done** (`results_commbound_throughput.md` 3c): measured the in-loop overhead the composition ignores (rank-local rejection sampling + verify-warmed cache update). Rejection sampling negligible (0.25-0.5 ms); naive 48-layer cache loop ~4 ms (vectorizes to <1 ms) -> total overhead ~1.5-1.8% of the cycle. Overhead-accounted integrated lossless speedup: **1.9-2.2x (beta 0.82) / 2.5-3.1x (beta 0.92)** at the socket operating point, growing with batch -- overhead does not materially erode the win. With B1 (algorithm correct + lossless) + B2a (real comm-free step), this is the integrated estimate without the full distributed driver. **B2b-full deferred** (true local-mode MoE + step-level distributed driver -> real end-to-end tokens/s + system losslessness; ~1.5-2 wk, high risk, still socket fabric) -- best done on a real PCIe/multi-node rental where the PCIe-exact + DBO-baseline gaps also live. **Real PCIe-link calibration (3d, bench_pcie_link.py)**: measured PCIe 55 (1-hop) / 27.5 (staged) vs NVLink 389 GB/s = 7-14x slower. The socket f=0.84 was TCP-latency-inflated; real PCIe is bandwidth-bound -> f ~0.05-0.1 low batch (weak), ~0.3-0.5 serving batch -> realistic lossless **~1.2-1.4x (beta 0.92) at serving batch** -- socket 2.5-3.1x is inflated. **ACTUAL real-PCIe (3e)**: forced PCIe-SHM via `NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1` (channels "via SHM/direct" GPU<->host<->GPU over PCIe; the missing knob was NVLS-disable -- plain P2P_DISABLE left NVLink multicast active, and the hang was NCCL->NET->IB). Measured PCIe-SHM EP decode 23/27/35/48 ms @ B=8/32/128/512 -> real f=0.37/0.43/0.56/0.65 -> lossless **~1.2-1.8x (beta 0.92), 1.78x @ serving batch** (1.1-1.5x beta 0.82). PCIe-exact gap CLOSED on this box. **End-to-end (3f)**: with BOTH step times measured on the forced-PCIe engine (draft skip-A2A 17.9-19.9 ms, comm-free confirmed; verify 23-48 ms) + measured accepted-length (B1 k-sweep) + measured ~1.5% overhead -> real lossless **~1.39x at serving batch B=512** (beta~0.85 local draft; ~1.58x if FP4-full beta0.92 w/ geometric multi-token). Every input now measured on the target engine; only a single-process integrated stopwatch (B2b-full) remains. **Batch scaling (3g)**: measured to global B=2048 -- f rises 0.54->0.62 (B512->1024) then PLATEAUS at 0.62 (B2048), speedup plateaus at **~1.47x (beta0.85) / ~1.7x (beta0.92)** -- more batch does NOT keep helping (at high batch both comm and compute scale with batch -> f converges to the structural comm:compute ratio ~0.62, not ->1). So the single-server-PCIe win is capped ~1.5-1.7x regardless of batch. Still open (need real multi-node): inter-node IB (bigger win, f=0.6-0.8 -> 1.6-2.3x), DBO baseline (DeepEP). See `24_commbound_throughput/results_commbound_throughput.md`. |
| `25_local_routing_strategy` | Complete | Two ways to raise the comm-free local-routing draft's beta (its weak point, Phase 24). (1) **Shared-expert anchor** -- the one local-routing lever never tested (Phase 08 scope was no-shared). On Qwen1.5-MoE (shared expert = ~45% of MoE-output norm), an always-local shared expert lifts local-routing acceptance by ~0.09-0.13, up to **2.5x at tiny cache** (0.067E: 0.38 vs 0.15 without shared); to hit beta~0.74 needs ~0.5E routed cache with shared vs ~0.75E without. Lift is EP-invariant (shared always local -> escapes R~cN) and scales with the shared mass fraction (Qwen1.5-MoE's 45% is high; DeepSeek ~10-15% -> smaller but real). (2) **FP4 cache scaling** (Qwen3): local routing interpolates coverage-limited -> quant floor; NVFP4 -> **0.92 at full coverage** (bf16 -> 1.0), so beta ~ min(coverage(C), quant) and FP4 makes a large local cache affordable. Strongest comm-free draft = shared anchor + FP4 routed cache. Router patch validated by C=E -> 1.0/0.92 sanity (caught+fixed a norm_topk_prob=False weight mis-normalization). **Follow-up: cross-architecture confirmation on DeepSeek-V2-Lite** (native impl; 64 routed top_k 6 + 2 shared, ~48% shared mass): shared anchor lifts local routing by **+0.25 to +0.58** -- the difference between useless (0.006-0.05 at small cache) and usable (0.26-0.63); even larger than Qwen. Mechanism generalizes across architectures. (An alpha-scaling proxy to emulate a *smaller* shared fraction failed -- intermediate scaling is OOD; endpoints reproduce Exp 1.) **Third architecture: Moonlight-16B-A3B** (DeepSeek-V3 sigmoid+noaux_tc gating, ~65% shared): lift +0.13 to +0.54 -- anchor robust across 3 architectures / 2 gating types. Cross-model the lift tracks shared fraction AND routed specialization (DeepSeek family's specialized routed experts collapse without the anchor -> bigger lift than Qwen). **Genuinely untested + hardware-gated**: all 3 runnable shared-expert MoEs are high-fraction (45-65%); the small-fraction regime (V3 1-shared ~11%, GLM-4.5) is frontier-scale only -> "small shared -> smaller lift" stays a (mechanism-backed) prediction needing a multi-GPU rental. **Three-way (shared + local + FP4)**, the realized comm-free draft, measured on V2-Lite + Moonlight: levers compose as beta ~ min(coverage-with-shared(C), quant-floor) -- FP4 costs <=0.03 up to 0.5E, caps at ~0.91-0.93 at full coverage; no destructive compounding. Bonus: the shared anchor buffers quant error too (with-shared FP4-full 0.93 vs without-shared 0.54). So comm-free FP4 draft on a shared-expert model = ~0.82 @ 0.5E, ~0.92 @ full -- comm-free AND cheap-memory AND anchor-lifted together. See `25_local_routing_strategy/results_local_routing.md`. |
| `26_accept_length_frontier` | Complete | Baseline accept-length-per-GB frontier for the comm-free FP4 local-routing draft (Qwen3). Accept-length-per-GB is ~CONSTANT (~0.32 acc/GB) up to 0.5E coverage, then a sharp KNEE (0.115 @0.75E, 0.04 @full) -- router mass concentrates in the top ~half of experts. Low-bit dominates (FP4-full beta0.92 beats FP8-half at equal GB). Operating points: knee FP4 0.5E = 8.2 GB/dev accept 2.64; max FP4-full = 16.3 GB accept 3.26. Beat-bar: >0.32 acc/GB below 8 GB, or accept >3.26 at <=16 GB. See `26_accept_length_frontier/results_frontier.md`. |
| `27_tree_drafting` | Complete | Tree drafting to raise accept length at minimal verify cost. Measured h(b)=P(verify-next in draft top-b): 0.875/0.979/1.000 for b=1/2/3 -- branching is very effective (top-3 ~always contains verify's token). BUT verify is one forward over B*nodes tokens -> verify cost scales with tree size (and with chain k). **This corrects 3f**, which used the 1-token verify cost for a k-token verify: with correct scaling, even perfect acceptance caps the high-batch speedup at ~1.35x (k=1) -- the comm-free draft (~23ms) nearly equals the comm it saves (~25ms). Optimal STRUCTURE is batch-dependent: low batch (B=8, latency-bound verify -> nodes ~free) wants a small balanced tree (full d=2,b=2, 6 nodes -> 1.21x, +8% over chain); mid/high batch (B>=128) wants a 1-token chain (trees give no gain, node count inflates the comm-bound verify). **Corrected single-node-PCIe speedup ~1.2-1.3x across batch** (down from 3f's optimistic 1.39x); tree drafting is a LOW-BATCH (latency) lever, and the best tree is small+shaped, never big. **Tree-attention validation (DONE, `tree_verify.py`)**: real accept lengths match h(b)+geom within 2-4% (full d2b2 1.95 vs pred 1.94; +23% over chain k2 1.58). **Losslessness**: the low global greedy_match (0.70 trees vs 0.94 chain) is CASCADE not a bug -- per-cycle fresh-reference control = 0.990 (one B1-numerics flip/~98 tokens). **KV correctness** (user-flagged): draft/verify KV diverge after the 1st MoE layer -> draft KV must not be reused at verify; harness recomputes full-weight verify (use_cache=False), and a real impl commits verify's KV (draft KV scratch). **Verify-context-KV draft upside MEASURED** (`verify_kv_draft.py`): giving the draft the verify's full-weight context KV (local routing only for the predicting token, still comm-free) lifts beta by only **~+0.017** (0.867->0.883; +0.12 accept@k4) -- and that's the best case (first token). So context fidelity is NOT the draft's bottleneck (0.5E local context already approximates full); the conservative beta ~0.85-0.875 is nearly tight, and the variant isn't worth its KV-plumbing for ~1pt. Single-node accept-length levers now mapped; open objective = memory (verify-warmed dynamic cache). See `27_tree_drafting/results_tree.md`. **Decision: verify-context-KV draft is now the DEFAULT** (small standalone gain, but it makes the dynamic-cache win possible -- Phase 28). |
| `28_dynamic_cache` | Complete (depth-0) | Memory axis: verify-warmed dynamic cache vs static top-C, with the verify-KV draft (`dynamic_cache.py`). **Big win.** depth-0 beta=overlap vs cache C: static needs C=64 (8.2 GB) for 0.90; **verify-warmed EMA hits 0.885 at C=16 (2 GB, 4x less)**; **last-committed-token cache hits 0.992 at C=8 (1 GB, 8x less AND higher beta)**. Mechanism: with verify-KV the context is full-weight, so only the predicting token's MoE matters; it is shaped by the LAST COMMITTED token's experts, which verify already knows -> caching those (top_k=8 -> C=8) gives beta~1. The "oracle" is REALIZABLE (last-token cache, causal). Beats the Phase 26 static frontier decisively (beat-bar beta~0.84 at <8 GB; got 0.99 at 1 GB). Scope: depth-0 = k=1 (the HIGH-BATCH optimum from Phase 27), so it applies to serving directly; k>1/trees (low-batch) need predicted experts for deeper draft tokens -> decay (follow-up). Speedup ceiling unchanged (~1.35x at k=1); the win is MEMORY (8x) + reliability (0.84->0.99). **Depth>0 decay (`depth_decay.py`)**: the last-token C=8 trick is depth-0 ONLY -- it COLLAPSES after the first draft token (0.99->0.36->0.21; deeper draft tokens predict from unverified draft tokens not in the 8-expert cache) -> accept ~1.6. For k>1/trees you need a broad cache (EMA/static C=32, 4 GB) -> beta ~0.6-0.8 across depth, accept ~2.9 (~2x memory win vs static C=64). **Even the ORACLE decays with depth** (0.99->0.84->0.63): the in-cycle draft tokens are local (verify hasn't run on them -> no clean KV), so accumulating local-draft context degrades deep predictions regardless of cache -- a FUNDAMENTAL cap on comm-free-draft depth. So: memory win is **8x at k=1 (high batch), ~2x for trees (low batch)**; deep trees capped by context degradation. **BATCHED-MEMORY CORRECTION (`union_coverage.py`)**: experts are SHARED weights per DEVICE, not per request -> with B concurrent requests the resident set must cover their UNION. union(B) per layer (E=128): 8 (B=1) -> 41 (B=8, 32%) -> 77 (B=32, 61%) -> 103 (B=128, 80%). So the 1 GB/beta0.99 per-request win is LOW-BATCH only; at high serving batch each device needs ~full FP4 replication (16 GB, beta0.92) OR globally-hot top-C + skip-cold (8 GB covers 90% -> beta0.85, batch-independent, verify corrects). **Tension: comm-bound speedup wants HIGH batch, memory win wants LOW batch -- they conflict.** The dynamic per-request cache is a low-batch/latency optimization; high-batch serving pays ~full FP4 replication. **Skip-cold beta MEASURED (`skip_cold.py`)**: real beta with a fixed globally-hot top-C cache (cold-routed tokens draft their top resident experts + renorm) EXCEEDS the coverage proxy -- C=32 (4 GB) 0.708, **C=64 (8 GB) 0.936 (bf16) / ~0.91 FP4, batch-independent** (renorm recovers the dropped low-weight experts' mass). So the realistic high-batch design = **globally-hot 0.5E + skip-cold (8 GB, beta ~0.91), 2x less than full FP4 replication (16 GB, 0.92) at ~the same beta**; low batch -> dynamic cache 1-4 GB / beta 0.99. See `28_dynamic_cache/results_dynamic_cache.md`. |
| `29_cross_model` | Complete (skew) | Cross-model validation of the Phase 26-28 enablers (all were Qwen3-only). Routed-expert skew on DeepSeek-V2-Lite (64 routed+2 shared) and GPT-OSS-20B (32, no shared, MXFP4-dequant) via forward hooks (DeepSeek gate is functional -> pre-hook block + recompute; greedy softmax top_k). **union(B) GENERALIZES** (per-device resident grows with batch to ~full routed set by B~32 on all; FASTER for small-E: DeepSeek/GPT-OSS ~88% @B=32 vs Qwen3 61%). **Routing skew holds** (global-hot 50%-E cache covers Qwen3 0.92 / GPT-OSS 0.78 / DeepSeek-routed 0.70 -- all skewed, Qwen3 most). **Shared-expert anchor is the differentiator**: DeepSeek's routed experts cache worst, but its 2 shared experts (free, batch-independent, ~48% of MoE mass, Phase 25) cover ~half the output -> a small routed cache suffices -> shared-expert models have a more favorable memory story. So the enablers are NOT Qwen3 artifacts. Scope: skew validated cross-model; beta refinements (renorm, dynamic cache, depth decay) stay Qwen3-measured, expected to generalize. See `29_cross_model/results_cross_model.md`. |
| `30_single_node_writeup` | Complete | Coherent synthesis of Phases 24-31 (`single_node_pcie_writeup.md`; folded in the Phase 31 EAGLE/cascade negative + a comparison to conventional spec decoding): the single-server-PCIe self-spec MoE mechanism end to end -- forced-PCIe recipe + measured f; corrected speedup ~1.2-1.35x (verify-cost-bound); acceptance levers (FP4/shared anchor/tree drafting, lossless + KV-correct); memory (1-4 GB low-batch dynamic cache beta0.99 / 8 GB high-batch globally-hot+skip-cold beta~0.91, batch-independent); cross-model validation; honest regime splits + fundamental ceilings (speedup ~1.35x cap, comm-free draft depth cap, verify-cost scaling); the bigger win is inter-node IB (hardware-gated). **Now also folds Phase 31** (EAGLE/MTP subsumes the comm-free-draft thesis + memory mechanism; only the draft-agnostic verify-comm-scaling survives) and a **conventional-spec comparison**: EAGLE dominates self-spec on draft cost (~8-10x cheaper head), beta (trained), and memory (no expert cache) -> higher speedup even in comm-bound MoE (~1.69x vs ~1.25x @ B512, est.); self-spec's only edge is training-free / out-of-the-box; the verify-comm ceiling is draft-agnostic and caps both. No new experiments. See `30_single_node_writeup/single_node_pcie_writeup.md`. |
| `31_eagle_tree_prune` | 31a NO-GO | Does self-MoE-spec survive EAGLE? Test: comm-free prune of the exact-verify tree (draft wide tree -> prune comm-free -> verify small). **Stage A (`prune_frontier.py`, Qwen3, wide [2,2,2,2]=30 nodes, L_full 3.83)**: free confidence-pruning is NEAR-ORACLE (P_conf ~95% of P_exact), pruned-deep >> shallow tree (N=6: 3.40 vs full_d2b2 1.96), but pruned-wide barely beats the CHAIN at matched verify nodes (+2.5%; oracle ceiling +7%). **Stage B economics (measured Phase-24 costs)**: 31a fails -- B>=128 it is 0.66-0.87x vs chain 1.18-1.27x (the WIDE-TREE DRAFT cost, ~178ms@B512, dominates; pruning saves only verify); B=8 it ties Phase 27's full_d2b2 (1.19 vs 1.22), even ORACLE pruner only ties. **Root cause:** the free signal already prunes near-optimally (no room for a costly pruner) AND pruning can't recover the wide-tree draft already paid. **31b implication:** P_conf~oracle means EAGLE's free head P1 is likely near-oracle too -> a separate local-MoE pruner P2 has <=5% headroom -> 31b NO-GO by the same evidence (README risk #1). **Verdict: direction closed (clean negative)** -- recommend plain EAGLE-on-MoE with a chain (high batch) / small tree (low batch); the verify-comm-scaling insight (Phase 27) is the durable contribution, pruning adds nothing. **Two-phase cascade (`prune_cascade_economics.py`)**: the user's "quantized comm-free phase-1 prune + full phase-2 verify" cascade (valid, lossless, unifies World A FP4-local-routing-as-phase-1 + World B EAGLE+full-verify) -- swept phase-1 cheapness q, EAGLE draft FREE (generous): beats the chain only if q<0.01-0.07 (phase-1 < ~1-2 ms, essentially free); FP8-H100, FP4-Blackwell, even free-draft all LOSE. Root cause: pruning needs forwarding the whole wide tree in phase-1, but the pruned-vs-chain accept gain is only ~+2.5% (h(b) saturates) -> phase-1 is pure added cost chasing a tiny gain; better to draft a chain directly. Would only pay on a high-tree-vs-chain-gap profile (ambiguous decoding); NO-GO on Qwen3's confident profile. Quantizing phase-1 helps cost but can't manufacture headroom. See `31_eagle_tree_prune/results_prune.md`. |
| `32_overlap_spec` | Model (positive) | Can World A compose with conventional spec via OVERLAP ("spec module every step")? Analytic model (`overlap_model.py`): hide the comm-free draft compute behind the verify's all-to-all; cycle=max(C_v+C_d, M_v). **The draft cost-vs-quality tradeoff FLIPS at R=comm/compute=2 (f~0.67).** Single-node PCIe (R<2): draft cost dominates -> cheap EAGLE wins (World A full-forward draft adds compute, can't hide). **Inter-node IB (R>2): the verify's comm hides the comm-free draft for FREE -> only beta matters -> World A's full-model draft (beta 0.92) BEATS EAGLE's small head (0.80), TRAINING-FREE** (1.92x vs 1.80x). So "spec every step" turns World A's expensive draft from a liability into an asset -- but only inter-node (the project's real-win regime). First composition where World A works WITH and BEATS conventional spec. Caveats: perfect-overlap assumption (real DBO 70-90%), data dependency needs speculative-ahead drafting, beta>EAGLE assumed, and it's hardware-gated (needs multi-node IB to measure). See `32_overlap_spec/results_overlap.md`. |
| `33_worldB_comm_tree` | Complete (measured) | **World B: comm-aware tree-spec for MoE-EP.** EAGLE3/MTP draft WIDE trees (~25-64 nodes); on MoE-EP the verify is one all-to-all over all tree nodes -> it routes ~tree-size x tokens to accept ~4 -> the comm blows up. Measured forced-PCIe verify step vs verified tokens (B_seq=64 x tree N): N=4 chain 40.9ms / N=6 pruned 47.0ms / **N=30 wide 110.8ms** (2.4x step, **4.7x bandwidth** vs the small tree). End-to-end (verify measured x Phase-31 accept x EAGLE-cheap draft): **EAGLE-default wide 1.26x, comm-aware pruned 2.61x, comm-aware chain 2.93x -> ~2.3x better**. So on comm-bound MoE-EP you should draft small/pruned trees, not EAGLE's default wide ones (the chain is the comm-floor; pruning brings a wide tree to it, 89% accept at 5x fewer nodes). This is the concrete systems win the verify-comm-scaling analysis (Phase 27/31) implies for conventional tree-spec. **Paper = World A (training-free local-routing self-spec, ~1.2-1.35x, no head) + World B (comm-aware tree-spec with EAGLE, ~2.3x over naive wide tree).** Caveat: accept = local-routing stand-in for EAGLE (ratio robust, verify comm draft-independent); inter-node widens the gap. See `33_worldB_comm_tree/results_worldB.md`. |

## Current Positioning (Phases 18-24)

**Thesis -- lossless low-precision speculative decoding for communication-bound MoE
serving.** Without NVLink/NVSwitch or across nodes, MoE inference is communication-
bound: the expert all-to-all (which moves *activations*) dominates the step. Make the
speculative **draft** a fully low-precision MoE forward -- quantized weights AND
activations, optionally local-routed -- cheap in compute, weight bandwidth, and
communication, while a bf16 full-EP **verify** keeps output exact (rejection
sampling). Because every draft step is verified, the draft may use precisions too
lossy to serve directly.

**Why communication-bound (machine-balance argument).** Expert-FFN arithmetic
intensity is ~1150 FLOP/byte (per token-expert route: ~9.4 MFLOP vs ~8 KB dispatch+
combine). Interconnect machine balance = compute_rate / link_BW (H100 ~500 TFLOP/s):

| fabric | balance (FLOP/byte) | vs FFN 1150 | regime |
| --- | ---: | --- | --- |
| NVLink/NVSwitch (~900 GB/s) | ~560 | below -> **compute-bound** | A2A not the bottleneck |
| PCIe (~50 GB/s) | ~10000 | >> -> **comm-bound ~9x** | A2A dominates |
| inter-node IB | worse | >> -> comm-bound | A2A dominates |

This **explains the single-node negatives (Phases 12/16) as an NVLink artifact**:
skipping the A2A only helps where the A2A is the bottleneck (PCIe / inter-node), not
on NVLink.

**The design ladder (measured acceptance vs bf16, Qwen3-30B; attack comm memory-free
first, then eliminate).**

| lever | attacks | comm | extra HBM | beta |
| --- | --- | ---: | ---: | ---: |
| FP8 activations | communication | x0.5 | 0 | **0.915 (free)** |
| NVFP4 activations | communication | x0.25 | 0 | 0.899 |
| NVFP4 weights | compute + enables replica | -- | replica | 0.921 |
| local routing (FP4 full replica) | communication | ->0 | ~14.5 GB/dev | ~0.92 |

FP8 anchor 0.967 (Phase 22). Errors barely compound (full W4A4 = 0.899). Use **NVFP4,
not MXFP4**, on the wire (0.899 vs 0.866). Quantization is a unified instrument:
weight quant for compute/coverage, activation quant for cheap comm shrink, local
routing for full comm elimination -- all kept lossless by the verify.

**Applicability.** Earns its keep when **bf16 does not fit per device (forced into
comm-bound EP) but FP4 does (draft can be a comm-free / comm-light local replica)** --
large MoE on memory-constrained, NVLink-less or multi-node systems. The win is a
**throughput** claim (comm-bound at serving batch); at B=1 latency on a single box the
intra-server A2A is too small (~break-even).

**Distinct from prior art (Phase 19).** QuantSpec (quantized KV self-draft) and SS-MoE
(on-device expert subset) do not cover lossless low-precision (weight + activation)
drafting that guts the EP all-to-all on comm-bound MoE serving.

**Measured (both sides now grounded).**
- *Cost side:* acceptance of every lever (Phases 18/22/23) -- weight FP4 (NVFP4)
  0.92, FP8 activations free (0.915), FP4 activations 0.899; inter-node exposed-A2A
  envelope via injection (Phase 20); PCIe streaming/locality (Phase 21).
- *Benefit side (Phase 24, the gap that is now closed):* on a real comm-bound testbed
  (sockets, since `NCCL_P2P_DISABLE` hangs this NVSwitch box) EP decode off NVLink is
  **84-89% communication** (machine balance confirmed); the comm-free draft step is
  **1.25-1.30x the compute floor** (skip removes ~85% of the step, B2a); the lockstep
  cycle is **lossless** (rejection-sampling theorem; greedy-numerics caveat attributed,
  B1) with real acceptance ~ one-step beta; and the overhead-accounted
  integrated lossless speedup is 1.9-2.2x (beta 0.82) / 2.5-3.1x (beta 0.92) at the
  **socket** point (B2b-pragmatic; overhead ~1.5%) -- but socket is TCP-inflated.
  **ACTUAL real-PCIe (Phase 24 3e):** the EP all-to-all CAN be forced over real PCIe
  here -- `NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1` -> channels "via
  SHM/direct" (GPU<->host<->GPU over PCIe; no NVLink/NVLS/IB, no hang). The hang was
  NCCL falling to NET/IB, and plain P2P_DISABLE left NVLS (NVLink multicast) active.
  Measured PCIe-SHM EP decode: 23/27/35/48 ms at global B=8/32/128/512 (vs NVLink
  15-17) -> real f=0.37/0.43/0.56/0.65 -> lossless **~1.2-1.8x (beta 0.92), ~1.78x at
  serving batch**; ~1.1-1.5x (beta 0.82). Brackets the 3d estimate; confirms socket
  2.5-3.1x was inflated. **The PCIe-exact gap is now closed on this box.**

**Now closed on this box:** the single-server-PCIe-exact point -- measured via forced
PCIe-SHM (Phase 24 3e), f=0.37-0.65, ~1.78x at serving batch (beta 0.92).

**Still open (need a real multi-node box):** the **inter-node IB** regime where the
*bigger* win lives (Phase 20: f=0.6-0.8 -> 1.6-2.3x at low batch -- inter-node IB is a
network hop, far more comm-bound than single-server PCIe's 7-14x), the DBO-overlap
baseline (needs DeepEP), and the full distributed step-level driver (B2b-full).

**Decision.** The communication-bound low-precision-draft direction is **validated
end-to-end on this hardware** (every lever's cost measured, the benefit measured,
lossless) -- but the **real-PCIe calibration tempers the magnitude**: the socket
proxy's 2.5-3.1x was latency-inflated; the bandwidth-calibrated real-PCIe win is
~1.2-1.4x at serving batch (β0.92), ~1x at low batch -- modest, a throughput effect.
Whether it clears a top-tier bar now hinges on regimes this box cannot produce: a real
PCIe/multi-node box with the LARGE inter-node A2A (Phase 20: f=0.6-0.8 -> the bigger
1.6-2.3x) and a DBO baseline. Remaining work is a hardware rental, not single-node.

---

## Historical Record (Phases 00-17, pre-pivot)

The sections below document the original local-top-k negative-result investigation
that motivated the Phase 18 pivot. Retained for provenance; superseded by "Current
Positioning" above.

## Timing Conclusion

Self-MoE-spec can only be useful in:

```text
low-batch multi-node EP decode
+ high exposed all-to-all after DeepEP/DBO
+ high local-only draft acceptance
```

Single-node EP is not enough. GPU6/7 local tests confirm this.

DeepEP and DBO should be treated as orthogonal components:

```text
local collective-free draft
+ DeepEP exact verification
+ DBO/phase-overlap at verification boundaries
```

In the current environment:

- DeepEP kernels are not installed.
- DBO cannot be measured on the target path.
- Forced same-host IB/GDRDMA fails below vLLM with `IBV_WC_RETRY_EXC_ERR`.
- Forced socket networking works but is not a practical multi-node IB proxy.

## Runtime Emulation Hook

Added research-only hooks:

```bash
VLLM_SELF_SPEC_EMULATE_A2A_DELAY_MS=<delay_ms>
VLLM_SELF_SPEC_LOG_A2A_COUNTS=1
VLLM_SELF_SPEC_A2A_COUNT_ACTIVE_FILE=<path>
```

These are for controlled local communication emulation. Defaults preserve normal
vLLM behavior.

Important caveat:

```text
VLLM_SELF_SPEC_EMULATE_A2A_DELAY_MS is per internal hook call,
not per decode step.
```

The current DP2+EP Qwen1.5-MoE path measured about:

```text
1536 active hook calls / worker / measured iteration
```

So delay values must be tiny and calibrated.

## Real Routing Coverage

### PowerMoE-3B

| EP size | Best simple `gamma` | Best simple top-k overlap | Full top-k local |
| ---: | ---: | ---: | ---: |
| 2 | ~0.59 | ~0.60 | ~0.5% |

### Qwen1.5-MoE-A2.7B

| EP size | Best simple `gamma` | Best simple top-k overlap | Full top-k local |
| ---: | ---: | ---: | ---: |
| 2 | ~0.62 | ~0.66 | ~13% |
| 4 | ~0.42 | ~0.46 | ~1% |
| 8 | ~0.32 | ~0.35 | <1% |

Hot expert replication helps but is still likely insufficient.

## Direct Acceptance Proxy

For Qwen1.5-MoE EP2, compared:

```text
full-router next-token distribution
vs.
local-masked-router next-token distribution
```

Result:

```text
best distribution overlap ~= 0.37
KL divergence: large
top-1 match: low / unstable
```

This is strong negative evidence for naive local-top-k drafting on this model.

Important caveat: this is still a **proxy**, not actual speculative acceptance.
It measures next-token distribution similarity after local router masking, but
it does not generate local draft sequences and verify them with the full model.
Actual acceptance rate `beta` remains unmeasured.

## Decision (pre-pivot, superseded by Phase 18+)

Do **not** implement naive local-top-k drafting in vLLM yet.

The likely bottleneck is acceptance, not timing. Even if communication savings
exist in a multi-node setting, the local-only draft appears too divergent on the
tested real models. However, this conclusion is based on simple placements and
acceptance proxies, not optimized placement plus actual verification.

## Recommended Pivot

Important correction: the main Self-MoE-spec design should keep the model's
default number of routed experts and reroute them into local experts. For
Qwen1.5-MoE, this means:

```text
draft_expert_top_k = default expert_top_k = 4
```

Top-1/top-2 local drafts are separate cheaper-draft variants, not the main
method.

Phase 05 tested top-1/top-2 as a cheaper variant, and also measured the default
local top-4 case. The default local top-4 case still had low one-token sampled
acceptance (`~0.37-0.38` in the best tested placements). Phase 06 tested simple
selective policies and found they cannot isolate a high-acceptance subset.
Therefore, move to either stronger placement/policy for same-top-k local routing
or to a stronger compute-saving/learned component:

1. **Same-top-k local rerouting with stronger placement/policy**
2. **Layer-skip + local-routing hybrid**
3. **Trained/lightweight draft head**
4. **Negative-result framing:** quantify when expert locality is insufficient
   for lossless local MoE speculation.

## Phase 07 Result: Recent Models

Tested recent MoE checkpoints to check whether the Phase 03-06 negative result is
a model-choice artifact. Best sampled one-token acceptance (main method,
`draft_top_k = default`):

| Model | EP2 | EP4 | EP8 |
| --- | ---: | ---: | ---: |
| Qwen3-30B-A3B (128 exp) | 0.50 | 0.31 | 0.12 |
| GPT-OSS-20B (32 exp) | 0.70 | 0.58 | 0.52 |

Recent models beat Qwen1.5-MoE (~0.40 at EP2), and GPT-OSS-20B at EP2 (0.70) is
the project's best result. But the central finding is structural:

```text
acceptance is highest at low EP (smallest communication advantage)
and collapses at high EP (the only regime with a real communication advantage).
```

More EP shards -> fewer routed experts per shard -> lower local mass `gamma` ->
lower acceptance. None of the four tested checkpoints (all without a shared
expert) clears `beta >= 0.8`, and all are weakest in the high-EP multi-node
target regime. See `07_recent_model_acceptance/results_recent_models_acceptance.md`.

## Phase 08: Negative-Result Write-up (Complete)

The scoped negative result is written up in
`08_negative_result/negative_result_report.md`. Core claims: (1) local-only
drafting never reaches `beta >= 0.8` on the four tested no-shared-expert models;
(2) acceptance and exposed communication are anti-correlated along the EP axis;
(3) scope caveat: a shared-expert anchor is untested.

## Phase 09: Rebalancing Ceiling (Complete)

Tested whether locality-aware rebalancing/replication can lift high-EP
acceptance, by sweeping a per-device replicated draft-cache budget `M` (mass-
optimal per-layer set, EP-invariant) and measuring acceptance vs `M`. Result:

```text
acceptance rises monotonically with M (your hypothesis is directionally right),
but reaching beta >= 0.8 needs M* ~= 0.5-0.8 * E (half to four-fifths of all
experts) replicated on EVERY device.
```

The required replication factor over the plain-EP budget is
`R = M*/(E/N) ~= 0.5N..0.8N`, which grows **linearly with EP**: R ~= 4x..6x at
EP8, ~17x..25x at EP32. So at the cross-node scale where the communication
advantage is large, useful acceptance requires near-full per-device replication,
negating the memory rationale for EP. See
`09_rebalancing_ceiling/results_rebalancing_ceiling.md`.

| Budget (plain EP) | Qwen3-30B | GPT-OSS-20B |
| --- | ---: | ---: |
| M=E/8 (EP8) | 0.23 | 0.36 |
| M=E/4 (EP4) | 0.47 | 0.54 |
| M=E/2 (EP2) | 0.78 | 0.63 |
| beta>=0.8 needs | M~0.54E | M~0.79E |

## Phase 10: Verify-Warmed Draft Cache (Complete)

Tested whether exploiting the verify step's routing (temporal locality) lets a
small dynamic cache replace the large static one. A per-layer LRU expert cache
warmed by verified routing beats a static cache, but modestly:

- Dynamic vs per-request static coverage gap is only ~0.04-0.10.
- Most of the gain over Phase 09 is **per-request specialization**, not temporal
  recency (recency adds the smaller part).
- `beta >= 0.8` now needs `C ~= 0.31 E` (Qwen3) / `0.47 E` (GPT-OSS) -- about
  **half** the Phase 09 static `M*`, a ~1.7-2x improvement.
- But the cache is still replicated, so `R = C*N/E ~= 0.3N..0.5N` still grows
  linearly with EP (~2.5-4x at EP8, ~10-15x at EP32). The high-EP problem is
  softened ~2x, not removed.

Acceptance validation: Qwen3 C=32 -> 0.74, GPT-OSS C=16 -> 0.83. See
`10_temporal_locality_cache/results_temporal_locality.md`.

## Phase 11: Hierarchical-EP Draft Feasibility (Complete)

Idea (user): don't disable EP for the draft -- flatten it to **intra-node EP**
(NVLink all-to-all only), verify with full EP (inter-node IB). The draft then sees
`E/nodes` experts and avoids only the expensive inter-node hop.

- **Acceptance: feasible.** Realistic contiguous node shard is weak (Qwen3 0.33 at
  2 nodes, 0.12 at 4), but a small per-node high-coverage cache (~5-8 experts/GPU,
  cheap because the node aggregates G GPUs) lifts it to ~0.74-0.78 at 2-4 nodes.
- **Timing: borderline, hardware-bound.** Measured NVLink A2A ~30 us/collective;
  a decode step has `2 x num_layers` collectives (Qwen3: 96 -> ~2.9 ms intra-node
  A2A). The draft still pays full compute + intra-node A2A, so it wins only when
  the inter-node fraction `f_inter >= ~0.5` (needs ~200 us/coll IB latency + lean
  compute). At `f_inter ~ 0.3` it is marginal (0.88 at b=0.74). Forcing a slow
  NCCL transport crashes here, so inter-node is modeled; a real go/no-go needs a
  2-4 node IB testbed. See `11_intranode_ep_draft/results_feasibility.md`.

This is the most promising framing: it moves acceptance from fatal (~0.1) to
feasible (~0.74) at low cost. The open gate is purely whether inter-node
all-to-all is a large enough fraction of the decode step.

## Phase 12: Exact EP-Width Latency (Complete)

Real vLLM decode latency (Qwen3-30B-A3B dummy, single-node NVLink, GPUs 2-5)
replacing the Phase 11 analytical timing. In this vLLM `ep_size = dp_size*tp_size`,
so EP2xDP2 = two independent `TP2+EP` engines, EP4xDP1 = one `TP4+EP` engine.

| | B=2 | B=8 | B=16 | B=32 |
| --- | ---: | ---: | ---: | ---: |
| EP4xDP1 = L4(B) ms/tok | 5.35 | 7.16 | 8.53 | 9.87 |
| EP2xDP2 = L2(B/2) ms/tok | 5.01 | 7.11 | 9.05 | 11.46 |

EP2xDP2 is only marginally faster at low batch (<=7% at B<=4), ties at B=8, loses
at B>=16. At equal batch EP4 is always faster (more GPUs/compute). So on NVLink,
reducing EP barely helps (and hurts at scale): the intra-node all-to-all is too
cheap for EP-width to matter. The hierarchical draft therefore pays off **only**
on the inter-node all-to-all, which this single node cannot produce. See
`12_ep_width_latency/results_ep_width_latency.md`.

This is the measured confirmation of the Phase 11 gate: acceptance is favorable,
but the timing benefit is inter-node-bound and unverifiable without a 2-4 node IB
testbed.

## Phase 13/14: Semantic-Parallelism-like Affinity Scheduling

Tested a Speculative-MoE/Semantic-Parallelism-style idea: cluster co-activated
experts and assign each request to the group with best coverage, then measure
local activation rate (LAR), draftable high-coverage positions, and one-step
acceptance.

At `G=2`, the pattern matches the paper qualitatively:

| Model | Contiguous LAR | Affinity LAR | Remote-volume reduction | All-step acc | Draftable acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-30B-A3B | 0.511 | 0.827 | 64.6% | 0.858 | 0.869 |
| GPT-OSS-20B | 0.512 | 0.755 | 49.7% | 0.840 | 0.888 |

At `G=4`, affinity still improves LAR, but absolute LAR remains below 0.5 and
there is no `coverage >= 0.9` draftable subset. Therefore the paper-like
intra-node locality pattern is easy to show as a routing/volume result, but a
large single-node end-to-end vLLM speedup is unlikely because Phase 12 measured
NVLink EP communication as too cheap relative to decode compute. See
`14_intranode_semantic_pattern/results_intranode_semantic_pattern.md`.

## Final State (pre-pivot negative result, superseded by Phase 18+)

Pure single-GPU local-only MoE drafting is not viable in the high-EP target
regime. The chain of evidence:

- Acceptance is too low and **anti-correlated** with the EP regime that supplies
  the communication advantage (Phase 07).
- A static rebalancing fix needs an infeasible `R ~ N/2` replication (Phase 09).
- A verify-warmed / per-request cache halves that to `R ~ 0.3N..0.5N` (Phase 10) --
  plausible at moderate EP with affordable warm-cache memory, but still
  near-full replication at cross-node EP.
- Semantic-Parallelism-like affinity/request scheduling shows the desired
  intra-node locality pattern at G=2, but does not provide a high-coverage G=4
  draftable subset and does not overturn the single-node timing limit (Phase 14).

The negative-result write-up is `08_negative_result/negative_result_report.md`.
The single remaining untested lever is an **EP-invariant shared expert** (fixed
always-local mass independent of replication, the only mechanism that escapes the
`R ~ cN` scaling). If revisited, run the Phase 07 proxy + Phase 09/10 sweeps on a
shared-expert MoE (DeepSeek / Llama-4 / GLM). Otherwise the negative result stands,
now with the rebalancing and temporal-cache mitigations quantified and bounded.
