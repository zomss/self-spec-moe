# Phase 75 — reproduce EfficientRollout, and locate our result on its roofline

Source phases: 74 (native fp8 draft = parity; window = the only win; the byte-budget
model), 72/73 (kernel-count + fixed-overhead floor), 66 (SHARED_KV), 64 (window accept),
58 (RL-regime motivation).

Paper: **EfficientRollout: System-Aware Self-Speculative Decoding for RL Rollouts**,
arXiv 2606.18967 (PDF: https://arxiv.org/pdf/2606.18967).

## Objective (the WHY)

Phase 74 concluded weight-quantization is **never** a self-spec win (fp8 = 0.97-0.99x of
bf16 across three regimes), and that **window** (draft-only KV cut) is the only lever
that moves the draft/verify ratio. EfficientRollout concludes the **opposite**: a **W4
weight-only RTN drafter** is "the most effective and practical self-drafting strategy,"
and it explicitly **evaluates and rejects sparse-attention drafting** (§B.2: τ≈3-4 at
γ=5 with a 256-token budget, vs 5.18 for W4).

Both can be true, and the byte budget says why. Their §A Table 3 (Qwen2.5-7B, A100,
TP=1, **simulation-based**) vs our measured Phase-74 model:

| | FFN/expert weights | QKVO | **Attn (KV read)** | LM head |
|---|---:|---:|---:|---:|
| **Theirs** Qwen2.5-7B dense (8k, b1) | 78.0% | 11.2% | **3.2%** | 7.5% |
| **Ours** Qwen3-30B-A3B MoE EP4 (16k, b8) | 27.4% | 8.6% | **61.1%** | 3.0% |

Inverted. Three causes: (1) MoE + EP shards experts so each rank reads *fewer* weight
bytes (5.77 GB) than a 4x-smaller dense model (14 GB); (2) KV scales as `b*ctx` — their
(8k,b1)=0.46 GB vs our (16k,b8)=12.88 GB; (3) Qwen3-30B-A3B carries 96 KiB/token of KV
(48 layers) vs Qwen2.5-7B's 56 KiB.

**So this phase is not "check if they are right." It is: reproduce their number in their
regime, then measure the transfer, and publish one roofline that predicts both.** If we
reproduce them and they reproduce our inversion, the two results merge into a single
law: *a draft-only lever wins iff it cuts the term that binds, and which term binds is
set by (dense|MoE) x (context x batch).*

This phase also absorbs the accept-vs-temperature de-risk that Phase 74 deferred:
**their τ is measured at rollout temperature 1.0**, ours was all greedy.

## What we are reproducing (exact claims, with paper numbers)

| # | claim | paper value | source |
|---|---|---|---|
| C1 | dense projections dominate rollout-tail decode | ~90% of latency, >10x attention; Tab.3: (8k,b1) FFN 78.0 / QKVO 11.2 / Attn 3.2 / LMhead 7.5 % | §4.1, Tab.3 (**simulation**) |
| C2 | W4 drafter cost ratio | idealized `Tq/Tp` = **0.360** (W8: 0.573) | Tab.4 (**idealized roofline, zero-overhead**) |
| C3 | W4 block efficiency at T=1.0 | τ = **3.59 / 5.18 / 6.70** at γ=3/5/7 (W8: 3.94/5.87/7.79); Qwen2.5-7B-Instruct, b1, 2k | Tab.4 (measured) |
| C4 | sparse-attn drafting is weak *there* | τ ≈ 3-4 at γ=5, 256-token budget | §B.2 |
| C5 | SD must be toggled off at large batch | early large-batch SD slower than AR | §4.2, Fig.4 |
| C6 | end-to-end RL gain | rollout latency **-19.6%**, step latency **-12.7%**, quant overhead 1.3-2.6 s/step | §5.2, Tab.2 |

**Do not conflate C1/C2 with measurements.** Tab.3 is captioned "Simulation-based" and
Tab.4's ratio is "predicted by our roofline model under memory-bound, zero-overhead
assumptions." Reproducing the *simulation* is arithmetic. The contribution is measuring
them — and Phase 72/73 predicts the measurement will be worse than the simulation,
because fixed overhead and kernel-count floors are exactly what a roofline omits.

## Assumptions & known deviations (record these before any number is trusted)

| axis | paper | us | consequence |
|---|---|---|---|
| GPU | A100-80GB SXM, FP16, TP=1 | H100 (SM90) unless an A100 is free | different machine balance; C1's *shares* should survive, `Tq/Tp` will not |
| int4 kernel | **Marlin** W4A16 | vLLM auto-selects **Machete** (ahead of Marlin in `kernels/linear/__init__.py:353`). Both **verified working** on h106 by `preflight.sh`. | not a forced deviation: run **both** as a controlled A/B. Force the paper's kernel with `VLLM_DISABLED_KERNELS=MacheteLinearKernel` (or `--linear-backend`). Always LOG the engaged kernel. |
| PTX | — | the `cudaErrorUnsupportedPtxVersion` seen in Phase 74 is confined to the **fp8 MoE** Marlin repack (`marlin_utils_fp8.py:270`); **dense int4 Marlin runs fine** | still gate with S0 on any new box — a different driver/torch pair may break more |
| RTN | "simplest **asymmetric** RTN" | `make_w4a16_int4_ckpt.py` uses `symmetric=True` | asym needs zero-points; test both, report τ for each |
| quantized layers | FFN + QKVO (LM-head NOT quantized) | script targets all `Linear`, `ignore=["lm_head"]` | equivalent for a dense model |
| drafter refresh | RTN re-quantized every training step | static prebuilt ckpt | equivalent for a static benchmark; only matters in S6 |
| draft KV | drafter computes its own KV | our `SHARED_KV=1` aliases the target's KV tensors | **changes τ** (draft attends exact target KV). Treat as a controlled knob, not a default |
| sampling | rollout **temperature 1.0**, top-p 1.0, group 8, max 8k | harness was greedy | `W7_TEMP`/`W7_TOPP` added to `w7_2node.py` (default 0.0 = prior phases byte-identical) |

Measurement hygiene inherited from Phase 74 (non-negotiable):
- `VLLM_SELF_SPEC_COMPILE_CONSISTENT` **OFF** (batch-invariant kernels tax spec ~2x).
- `VLLM_SELF_SPEC_PROFILE` **OFF** for any throughput number (per-region syncs, ~25%).
- Every arm must log its engaged kernel and be **rejected** if it doesn't match the
  intended one. Grep case-sensitively for the real token (`MACHETE`/`MARLIN`).
- Report per-iteration values. If a cell is bimodal or trips the harness `suspect` flag
  (rel-std > 15%), quote the median and explain the stall — never the bare mean.

## Stages (cheap -> expensive; each gates the next)

### S0 — Preflight (minutes, 1 GPU). **Hard gate.**
`scripts/preflight.sh`. Verifies: uv venv + vllm import; driver/toolkit; downloads
Qwen2.5-7B-Instruct; builds a tiny W4A16 layer and runs it, catching
`cudaErrorUnsupportedPtxVersion`; prints the engaged MP-linear kernel.
**GATE:** if no int4 W4A16 kernel runs on this box, STOP — the repro is impossible here.
Report the driver/torch pair and move to a box whose driver matches the torch CUDA build.

### S1 — C1: decode-time breakdown (1 GPU, ~1h)
Two independent estimates for Qwen2.5-7B, FP16, TP=1, full attention, at their exact
cells `(2k,b8) (2k,b4) (4k,b2) (8k,b1)`:
- **(a) analytic** byte model (reuse Phase-74's; it reproduced the README's 96 KiB/token
  independently) -> should match Tab.3 nearly exactly, since Tab.3 *is* a simulation.
- **(b) measured** kineto trace of one decode step, bucketed FFN / QKVO / Attn / LM-head
  / **other (launch + nonlinear + overhead)**.

**PASS (C1):** analytic reproduces Tab.3 within ±3 pts. **Finding either way:** the gap
between (a) and (b) is the paper's blind spot; Phase 72 measured ~2400 tiny kernels at
b1, so predict a large `other` bucket that the roofline assigns to nothing.

### S2 — C2: the *real* `Tq/Tp` (1 GPU, ~2h)
Build W4A16 + W8A16 RTN ckpts of Qwen2.5-7B-Instruct (CPU, data-free; `scripts/make_ckpts.sh`).
Measure draft-forward `Tq` and target-forward `Tp` at (2k, b1) with the same clock.
**PASS (C2):** we reproduce their *idealized* 0.360/0.573 from the roofline formula.
**Expected finding:** measured `Tq/Tp` > idealized. Quantify the overhead gap — that is
the honest cost of a W4 drafter and the direct analogue of Phase 74's fp8 result.

### S3 — C3 + the deferred temperature question (1 GPU, ~3h). **The linchpin.**
Self-spec: target = bf16 Qwen2.5-7B-Instruct, draft = W4A16 (then W8A16) ckpt, γ∈{3,5,7}.
Prompts: SimpleRL-8k-hard-style math, chat template. **temperature 1.0, top_p 1.0**
(`W7_TEMP=1.0`), and a greedy control (`W7_TEMP=0.0`) to isolate the temperature effect
Phase 74 deferred. Cross with `SHARED_KV ∈ {0,1}` and `symmetric ∈ {sym,asym}`.
- **Correctness gate (mandatory):** at `W7_TEMP=0`, spec output must be **token-identical**
  to no-spec. Losslessness comes from rejection sampling, but this also catches a
  `SHARED_KV` draft corrupting the target's KV pool.
- **PASS (C3):** τ within ±0.3 of 3.59/5.18/6.70 (W4) and 3.94/5.87/7.79 (W8).

### S4 — C5: the SD/AR crossover (1 GPU, ~2h)
Sweep `B ∈ {1,2,4,8,16,32,64}` x `S ∈ {2k,8k}` at γ*, measuring SD vs AR.
**PASS (C5):** SD wins at small B and loses at large B, with a crossover.
Cross-check: Phase 74 independently measured 0.86x at b64 on the MoE. Same law.

### S5 — the transfer (the payoff; 4 GPUs, ~4h)
Fill the 2x2 that explains both papers with one roofline:

| | weight-quant (W4) | window (sparse-attn) |
|---|---|---|
| **dense** Qwen2.5-7B, 2k, b1 | predict **WIN** (their C3) | predict **weak** (their C4: τ 3-4) |
| **MoE** Qwen3-30B-A3B, 16k, b8 | predict **parity** (P74: fp8 0.98x; W4 ceiling +10%) | predict **WIN** (P74: 1.16-1.34x) |

Both off-diagonal cells are predictions **we have not run**. Confirming them converts
"our result contradicts theirs" into "one law, two operating points."
Stretch: DeepSeek-V2-Lite (MLA, tiny KV) should behave like the *dense* row — under MLA
the KV term collapses and weight-quant should start paying on an MoE too. That is the
sharpest falsifiable prediction this phase can make.

### S6 — C6: end-to-end RL (8 GPUs, days). **Gate on S1-S4.**
veRL + GRPO, SimpleRL-8k-hard, Qwen2.5-7B, train batch 128, group 8, T=1.0, max response
8k, ~50-100 steps, with the SD toggle (S4 crossover) and adaptive γ. Without the toggle
you will not see the gain.
**PASS (C6):** rollout latency -19.6%, step latency -12.7%, quant overhead 1.3-2.6 s/step.
Note their gain is modest — the idealized 1.85x becomes ~1.24x on rollout once overheads
and the toggle are real. Budget accordingly; do not start S6 to "see if SD is good."

## Decision criteria

- **Reproduced:** S1-S3 land within tolerance -> EfficientRollout stands, and Phase 74's
  "quant is never a win" must be **scoped** to "on a sparse-MoE draft at long context,"
  not stated universally. Update `74/results_draft_cost.md` accordingly.
- **Reproduced + transfer confirmed (S5):** publish the unified roofline. This is the
  strongest outcome and the intended paper contribution.
- **Not reproduced (τ or Tq/Tp far off):** first suspect *our* deviations, in this order:
  Machete-vs-Marlin, sym-vs-asym RTN, SHARED_KV, temperature, prompt distribution. Only
  after all five are controlled is a discrepancy with the paper reportable.
- **S0 fails:** the box cannot run int4; this is an environment result, not a scientific
  one. Do not report any number from it.

## Expected next artifact

`results_repro.md`: the C1-C6 table (paper value | our value | verdict | deviation),
the measured-vs-simulated breakdown gap, and the S5 2x2. Then either the unified-roofline
paper section, or a scoping correction to Phase 74.

## Runbook (fresh server)

```bash
git fetch && git checkout research/self-spec-moe && git pull
cd research/75_efficient_rollout_repro
bash scripts/preflight.sh                 # HARD GATE. reads: int4 kernel + PTX + models
bash scripts/make_ckpts.sh                # CPU, data-free RTN -> ~/ckpts/Qwen2.5-7B-{W4A16,W8A16}*
# then S1..S4, one stage at a time, reading results before launching the next:
bash scripts/run_s1_breakdown.sh
bash scripts/run_s2_tqtp.sh
bash scripts/run_s3_tau.sh                # the linchpin; includes the token-exactness gate
bash scripts/run_s4_crossover.sh
```

Scripts follow Phase-74 conventions: run **by path** (never inline the body — `pkill -f`
matches the invoking shell's own argv and kills it), each arm logs its engaged kernel,
`data/` + `logs/` are phase-local and gitignored.
