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
regime (narrowed scope below), and optionally measure the transfer.** If we
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
| PTX | — | the `cudaErrorUnsupportedPtxVersion` seen in Phase 74 is confined to the **fp8 MoE** Marlin repack (`marlin_utils_fp8.py:270`); **dense int4 Marlin runs fine** | still gate with E0 on any new box — a different driver/torch pair may break more |
| RTN | "simplest **asymmetric** RTN" | `make_w4a16_int4_ckpt.py` uses `symmetric=True` | asym needs zero-points; test both, report τ for each |
| quantized layers | FFN + QKVO (LM-head NOT quantized) | script targets all `Linear`, `ignore=["lm_head"]` | equivalent for a dense model |
| drafter refresh | RTN re-quantized every training step | static prebuilt ckpt | equivalent for a static benchmark; only matters for the (out-of-scope) RL run |
| draft KV | drafter computes its own KV | our `SHARED_KV=1` aliases the target's KV tensors | **changes τ** (draft attends exact target KV). Treat as a controlled knob, not a default |
| sampling | rollout **temperature 1.0**, top-p 1.0, group 8, max 8k | harness was greedy | `W7_TEMP`/`W7_TOPP` added to `w7_2node.py` (default 0.0 = prior phases byte-identical) |

Measurement hygiene inherited from Phase 74 (non-negotiable):
- `VLLM_SELF_SPEC_COMPILE_CONSISTENT` **OFF** (batch-invariant kernels tax spec ~2x).
- `VLLM_SELF_SPEC_PROFILE` **OFF** for any throughput number (per-region syncs, ~25%).
- Every arm must log its engaged kernel and be **rejected** if it doesn't match the
  intended one. Grep case-sensitively for the real token (`MACHETE`/`MARLIN`).
- Report per-iteration values. If a cell is bimodal or trips the harness `suspect` flag
  (rel-std > 15%), quote the median and explain the stall — never the bare mean.

## SCOPE (narrowed) — is the weight-quant LEVER available in their setting?

We are **not** reproducing the framework (no veRL, no RL training, no SD toggle, no
adaptive γ). We are testing one thing: **does weight-only quantization actually make a
self-draft cheap enough to win, in the dense small-batch regime where they claim it
does?** Everything else in the paper only decides *when* to switch that lever on.

Their own speedup model factors the claim into exactly two measurable quantities:

```
speedup = τ / (γ · (Tq/Tp) + 1)
```

- `Tq/Tp` — the **systems** half: does W4 really make the draft forward cheap?
- `τ` — the **statistical** half: does W4 stay accurate enough to keep acceptance?

So there are **two key experiments** (E1, E2), one confirmation (E3), and one gate (E0).
The old S1/S4/S5/S6 are demoted to *diagnostics and optional contrasts* (D1-D3) — run only on
mismatch, or for the cross-regime story.

### Calibration: the byte model reproduces the paper, and predicts our number

`T(arm) = F + bytes(arm)/BW`, quantizing FFN+QKVO only (lm_head stays fp16, per §4.1):

| F (ms) | A100 W4 | **H100 W4** | A100 W8 | H100 W8 |
|---:|---:|---:|---:|---:|
| 0.0 (idealized) | 0.314 | 0.314 | 0.542 | 0.542 |
| **0.5** | **0.359** | 0.386 | 0.572 | 0.590 |
| 1.0 | 0.398 | 0.444 | 0.599 | 0.629 |
| 2.0 | 0.464 | 0.533 | 0.643 | 0.689 |

The paper's `Tq/Tp` = **0.360** (W4) / 0.573 (W8) and its 1.85x are reproduced by this
model on **A100 at F ≈ 0.5 ms**. Note the asymmetry: **H100 is the HARDER test** —
higher bandwidth shrinks the byte term, so the (GPU-independent) fixed overhead is a
larger share and the ratio worsens. Prefer an A100. On H100 predict `Tq/Tp` ≈ 0.39-0.49
and E2E ≈ 1.5-1.8x, not 1.85x. **Under-reproducing on H100 is expected and is not a
refutation** — report F alongside the ratio.

### E0 — Preflight (2 min). **Hard gate.** `scripts/preflight.sh`
An int4 W4A16 kernel must execute on-device. Verified working on h106 (Machete **and**
dense Marlin). If it fails, the box cannot run this phase; that is an environment
result, not a scientific one.

### E1 — KEY #1: measured `Tq/Tp` (1 GPU, ~1-2 h)
Standalone decode-step latency, Qwen2.5-7B-Instruct, TP=1, **b1, ctx 2k** (their point),
for **bf16 / W8A16 / W4A16**, x kernel ∈ {Machete (auto), Marlin (`VLLM_DISABLED_KERNELS=
MacheteLinearKernel`, the paper's)}. This isolates the lever from the spec harness.
- Also solve for **F and effective bandwidth** from the bf16/W8/W4 triple: three
  measured times, known byte counts, two unknowns (`F`, `BW_eff`) — overdetermined, so
  the residual tells you whether the W4 kernel is *converting bytes into time* at all.
  That is precisely the Phase-74 failure mode (fp8 halved bytes and bought nothing).
- **PASS:** `Tq/Tp(W4)` ≤ 0.50 on H100 (≤ 0.40 on A100) -> the lever exists.
- **FAIL:** ≥ 0.70 -> weight quant is not available in practice on this stack, even in
  their regime. That would be the headline, and it would need `D1` to explain it.

### E2 — KEY #2: acceptance `τ` (1 GPU, ~2-3 h)
Self-spec: target = bf16 Qwen2.5-7B-Instruct, draft = W4A16 ckpt (then W8A16),
γ ∈ {3,5,7}, b1, 2k, **temperature 1.0, top_p 1.0** (`W7_TEMP=1.0`) on math prompts,
plus a greedy control (`W7_TEMP=0.0`). This is also the accept-vs-temperature de-risk
Phase 74 deferred — every prior accept number we have is greedy.
- **Correctness gate (mandatory):** at `W7_TEMP=0`, spec output must be **token-identical**
  to no-spec. Catches a `SHARED_KV` draft corrupting the target's KV pool.
- Controlled knobs: `SHARED_KV ∈ {0,1}`; RTN `sym` vs `asym` (paper uses **asymmetric**).
- **PASS:** τ within ±0.3 of **3.59 / 5.18 / 6.70** (W4) and 3.94/5.87/7.79 (W8).

### E3 — Confirmation: end-to-end spec vs AR (1 GPU, ~1-2 h)
Measure real tok/s, spec vs no-spec, at b1 / 2k, γ = γ\*. Check it matches
`τ / (γ·(Tq/Tp) + 1)` using the **measured** E1 ratio.
- **The trap:** E1 measures a standalone, CUDA-graphed forward. The real self-spec draft
  chain runs PIECEWISE/eager (Phase 35/72), so E1's `Tq/Tp` is a **lower bound** —
  optimistic. If **E1 passes but E3 fails**, the lever exists in the kernel but the spec
  harness eats it. That is exactly the Phase-74 fp8 story (ideal +6.5% -> measured -2%),
  and it is the single most likely outcome to misread.
- **PASS:** E2E > 1.0x at b1, and within ~15% of the formula's prediction.

### Verdict logic — E1×E2 decides the lever; E3 is a SEPARATE axis

The scientific question ("is weight-quant a real self-spec lever in the dense
small-batch regime?") is answered by **E1 × E2 alone** — a *cheap* draft (E1) that stays
*accurate* (E2). E3 does **not** test the lever; it tests how much of it OUR PIECEWISE
draft chain delivers — a variable we already know is lossy (Phase 74: ideal +6.5% ->
measured -2%). So read E3 as **harness quality**, never as the method verdict. Do not
gate the lever conclusion on E3.

**Lever verdict (E1 × E2) — the result that matters:**
| E1 (draft cheap?) | E2 (draft accurate?) | conclusion |
|---|---|---|
| pass | pass | **The lever IS available in their regime.** Phase 74's "never a win" is thereby SCOPED (true for a sparse-MoE draft at long ctx, not universal). The byte-budget model already predicted this; E1×E2 *confirms* it. This is the likely outcome. |
| pass | fail | W4 is cheap but too inaccurate at T=1.0 -> their τ doesn't transfer. Exhaust deviations in order: sym-vs-asym RTN, then temperature/prompt distribution. |
| fail | — | W4 does not convert bytes->time even where weights dominate -> run D1; prefer an A100 (H100 biases against, quantified above). |

**Harness-delivery axis (E3) — read ONLY after the lever verdict, never instead of it:**
| E3 e2e vs formula | meaning |
|---|---|
| ≥ 0.85 × `τ/(γ·Tq/Tp+1)` | our harness delivers the lever cleanly |
| < that | the PIECEWISE draft chain eats it — a **systems bug on OUR side**, not a refutation of the paper. The *expected* H100 outcome (Phase-74 shape). Fix the harness, don't touch the verdict. |

---

## Optional / diagnostic (do NOT run by default)

### D1 — decode-time breakdown. **Run only if E1 misses its prediction.**
The debug tool for a failed E1, not a headline. Kineto-trace one decode step of
Qwen2.5-7B at (2k,b1) and bucket into FFN / QKVO / Attn / LM-head / **other (launch +
nonlinear)**. Compare to their Tab.3 (`FFN 78.0 / QKVO 11.2 / Attn 3.2 / LMhead 7.5` at
8k,b1). Their table is captioned **"Simulation-based"**, so the analytic side is
arithmetic and will match; the *measured* `other` bucket is what a roofline omits and is
where a failed E1 will be hiding (Phase 72: ~2400 tiny kernels at b1).

### D2 — SD/AR crossover vs batch. **Only if you want their toggle rationale.**
Sweep `B ∈ {1,2,4,8,16,32,64}` at γ\*. Expect SD to win small-B and lose large-B.
Phase 74 already measured 0.86x at b64 on the MoE, independently — same law, so this
mostly re-confirms known physics. Out of scope for "is the lever available."

### D3 — the transfer (4 GPUs, ~4h). **Optional, but this is the paper-grade result.**
Fill the 2x2 that explains both papers with one roofline:

| | weight-quant (W4) | window (sparse-attn) |
|---|---|---|
| **dense** Qwen2.5-7B, 2k, b1 | predict **WIN** (their C3) | predict **weak** (their C4: τ 3-4) |
| **MoE** Qwen3-30B-A3B, 16k, b8 | predict **parity** (P74: fp8 0.98x; W4 ceiling +10%) | predict **WIN** (P74: 1.16-1.34x) |

Both off-diagonal cells are predictions **we have not run**. Confirming them converts
"our result contradicts theirs" into "one law, two operating points."
### E4 (promoted from "stretch") — the MLA crossover: KV-size IS the knob

This is the sharpest falsifiable prediction in the phase, and the one result neither
paper contains. The unifying law says *a draft-only lever wins iff it cuts the binding
term, and which term binds is set by the KV-vs-weight byte ratio.* If that's right, then
**shrinking KV while holding "MoE + sparse-active-weight" fixed should flip weight-quant
from parity to a win** — with no change to the drafter, only to the attention that sets
the KV term. **MLA is exactly that knob.**

Controlled A/B, two MoEs at the SAME (ctx, batch), same W4 draft lever:

| MoE | attention | KV/token | KV @ (16k, b8) | predicted weight-quant result |
|---|---|---:|---:|---|
| **Qwen3-30B-A3B** (P74) | GQA, 48L | ~96 KiB | ~12.9 GB (KV-bound) | **parity** (measured: fp8 0.98x; W4 ceiling ~+10%) |
| **DeepSeek-V2-Lite** (cached) | **MLA**, 27L, kv_lora=512 | ~30 KiB (~3.3x smaller) | ~3.9 GB | **predict WIN** — KV no longer binds, weight-read rises to ~50% of the byte budget, so a 4x W4 cut moves the draft/verify ratio |

- **Model**: `deepseek-ai/DeepSeek-V2-Lite` (already cached, P74 inventory). MLA MoE,
  ~2.4B active — same activated-weight class as Qwen3-30B-A3B, so *only KV differs*.
- **Arms**: W4A16 (Machete) draft vs bf16 base vs no-spec, DP4/EP4, (16k, b8), K=K*.
  Reuse `make_w4a16_int4_ckpt.py` on DeepSeek-V2-Lite (targets Linear, ignore lm_head
  + the MLA up-projections if they must stay bf16 — check the config's quantization
  compatibility first; MLA absorbs KV proj into the attention, verify it quantizes).
- **Isolation**: run BOTH MoEs in the same session/box so the within-box
  draft/base ratio is the only quantity compared (never cross-box tok/s).
- **PASS (law confirmed)**: `W4_draft / bf16_draft` > 1 on DeepSeek-V2-Lite while it is
  ~parity on Qwen3-30B-A3B. **FAIL (law wrong or incomplete)**: still parity on the MLA
  MoE -> KV-size is not the sole crossover variable; something else (EP comm, fixed
  overhead floor from P72/73) also binds. Either outcome is publishable.

Caveat: the byte estimate assumes MLA KV = `(kv_lora_rank + qk_rope_head_dim)·L·2B` in
the absorbed form; confirm against the actual runtime KV spec before trusting the 3.3x.
And DeepSeek-V2-Lite's smaller active weight may raise the fixed-overhead share (P73),
which pushes AGAINST the win — so a null result is genuinely informative, not a bug.

### OUT OF SCOPE — C6, the end-to-end RL run (veRL + GRPO, 8 GPUs, days)
Their -19.6% rollout / -12.7% step latency needs the full framework: veRL, SimpleRL-8k,
the SD toggle, adaptive γ. **We are not reproducing this.** Note for context: the
idealized 1.85x becomes ~1.24x on rollout once overheads and the toggle are real, so the
E2E number tests the *framework*, not the lever. E1-E3 test the lever.

## Decision criteria

- **E1+E2+E3 pass:** the weight-quant lever **is** available and effective in their
  setting. Then Phase 74's "quant is never a self-spec win" must be **scoped** — it is
  true for a sparse-MoE draft at long context, not universally. Update
  `74/results_draft_cost.md` and `74/README.md`, which currently state it too broadly.
- **E1 passes, E3 fails:** the lever is real; our self-spec harness eats it (PIECEWISE
  draft chain). That is a *systems* bug on our side, not a refutation of the paper.
- **E1 fails:** weight quant does not convert bytes into time on this stack even when
  weights dominate. Run D1 before saying a word about it — and prefer an A100 first,
  since H100 biases against reproduction.
- **Before reporting ANY discrepancy with the paper**, exhaust our five deviations in
  this order: (1) H100-vs-A100 (predicted, quantified above), (2) Machete-vs-Marlin,
  (3) sym-vs-asym RTN, (4) `SHARED_KV`, (5) temperature + prompt distribution.
- **E0 fails:** the box cannot run int4; an environment result, not a scientific one.
  Report no number from it.

## Expected next artifact

`results_repro.md`: an E1/E2/E3 table (paper value | predicted-for-our-GPU | measured |
verdict | deviation), the fitted `F` and `BW_eff`, and the verdict-logic row we landed
on. Then either a scoping correction to Phase 74, or — if D3 is run — the unified
roofline figure.

## Runbook (fresh server)

Prefer an **A100** (H100 biases against reproduction — see the calibration table).

```bash
git fetch && git checkout research/self-spec-moe && git pull
cd research/75_efficient_rollout_repro

huggingface-cli download Qwen/Qwen2.5-7B-Instruct   # once, needs network

bash scripts/preflight.sh        # E0. HARD GATE: an int4 W4A16 kernel must execute
bash scripts/make_ckpts.sh       # CPU, data-free RTN, ~20 min, NO GPU. Builds
                                 #   W4A16-INT4-sym | W8A16-INT8-sym | W4A16-INT4-asym

bash scripts/run_e1_tqtp.sh      # KEY #1: measured Tq/Tp, bf16|W8|W4 x Machete|Marlin
                                 #   -> also fits F and BW_eff. STOP and read this.
E75_GAMMA=5 bash scripts/run_e2_tau.sh   # KEY #2: tau at gamma=3,5,7, T=1.0 (+controls)
                                 #   -> refuses to report tau until the lossless gate passes
E75_GAMMA=<tau-optimal> bash scripts/run_e3_e2e.sh   # confirmation: tok/s spec vs AR
```

Read each stage's output before launching the next: **E1's `Tq/Tp` sets E3's prediction,
and E2's τ-optimal γ is E3's `E75_GAMMA`.**

Knobs (all optional): `E75_GPU` (default 0), `E75_MODEL`, `CKPT_ROOT` (default `~/ckpts`),
`E75_GAMMA` (E3), `E75_ITERS` (E1).

Guardrails wired into the scripts, and why each exists:
- `which_kernel.py` — `choose_mp_linear_kernel()` returns **silently**, so a log-grep
  guardrail would be vacuous. Every E1 arm asks the chooser directly and **aborts** if it
  gets a kernel other than the one intended (Machete auto / Marlin forced).
- `check_lossless.py` — E2 refuses to report τ until greedy spec output is
  **token-identical** to greedy no-spec. This catches a weight-quantized draft corrupting
  the target's KV pool under `SHARED_KV=1`.
- E2 warns loudly if `Draft model quantization:` never appears in the load log — that
  would mean the "quantized" draft silently loaded as bf16.
- E1's analyzer fits `F` and `BW_eff` and prints **residuals**: a large positive W4
  residual means the kernel is latency/tile-bound, i.e. halving bytes bought nothing —
  the Phase-74 fp8 failure mode, distinguished from a mere overhead penalty.
- Run scripts **by path**. `pkill -f` matches the invoking shell's own argv; inlining a
  script body into `bash -c` or a heredoc makes it kill itself.

Validated before shipping: `preflight.sh` executed on h106 (Machete **and** dense Marlin
both run int4); `which_kernel.py` confirmed to flip Machete→Marlin under
`VLLM_DISABLED_KERNELS`; `analyze_e1.py` recovers a known `F`=0.500 ms and `BW`=3.35 TB/s
from synthetic timings with zero residual. The E1/E2/E3 GPU runs themselves are unrun —
they need `Qwen2.5-7B-Instruct`, which is not cached on h106.

Scripts follow Phase-74 conventions: run **by path** (never inline the body — `pkill -f`
matches the invoking shell's own argv and kills it), each arm logs its engaged kernel,
`data/` + `logs/` are phase-local and gitignored.
