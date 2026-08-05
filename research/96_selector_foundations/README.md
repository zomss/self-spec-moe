# Phase 96 — selector foundations: fix the measurement, then rebuild the
# selector as two rounds

Source: phase 95 (C3 deploy) findings, user directive 2026-08-05 ("stop C3,
make a new task for resolving these issues"). Phase 95 is STOPPED, not
closed — its measured results stand and are the input here.

## Why C3 was stopped

Phase 95 set out to deploy C2's searched map at runtime. It measured the
switching envelope on real data and got a defensible answer (dense +2.7%
cross-seed on long-context workloads, llama refuted). But the same
measurements exposed five problems that sit UNDER the selector, and no
amount of runtime engineering fixes them. Building the switcher on top would
have been building on an unvalidated base.

None of the five is a modelling opinion; each is a measured number.

---

## The five issues

### I1 — metric currency mismatch (search vs deployment)

C2's map is measured **decode-only**: the compile-cell protocol is
`T(1+N) - T(1)`, which cancels prefill by construction. Phase 95's regime
evaluation measured **end-to-end serving wall clock**. Prefill share of that
wall clock, dense AR:

| regime | prompt | gen | prefill share |
|---|---|---|---|
| R4 | 8179 | 512 | **30.0%** |
| R5 | 14037 | 512 | **46.6%** |
| R5cot | 14121 | 3072 | 15.5% |
| R6 | 62 | 256 | 13.3% |
| R1 | 63 | 1024 | 0.1% |

A decode speedup `S_dec` lands end-to-end as `1/(phi + (1-phi)/S_dec)`. At
R4's phi=0.30, the map's 1.288 becomes ~1.19; measured 1.007. The map
systematically overstates, and overstates MOST where prefill dominates —
which is the long-context region where the window lever lives.

Our own record already contains both currencies (C1 Stage B: "serving
wall-clock"; C2 compile cells: decode-only), which is how the mismatch went
unnoticed.

### I2 — the map's index is incomplete

Cells are keyed `(batch, total context)`. The window covers the context
**tail**, so what decides it is the **self-generated suffix length**, not
total context. R5 and R5cot occupy the SAME cell (b8, ~14k) and want
opposite windows, reproducibly on both boots:

| regime | gen | S w512 | S w2048 | winner | d(accept) w512->w2048 |
|---|---|---|---|---|---|
| R5 (s0) | 512 | **1.3002** | 1.2912 | w512 | +0.088 |
| R5 (s1) | 512 | **1.3253** | 1.2630 | w512 | +0.019 |
| R5cot (s0) | 3072 | 1.6998 | **1.8337** | w2048 | **+0.310** |
| R5cot (s1) | 3072 | 1.6874 | **1.7600** | w2048 | **+0.300** |

The scheduler already has the missing variable (`num_computed_tokens -
num_prompt_tokens`); the MAP has no axis for it.

Do NOT reduce this to "window ~ generated length": that rule fits
R4/R5/R5cot/R6 but fails R1 (gen 1024, w512 won) and dense R8 (gen 2048,
w512 won by 7.8%). Coverage buys tau, width costs R; at higher batch the
cost term wins. It is still `S = tau/(K*R+1)`, with generated length
entering through tau.

### I3 — the search emits point estimates with no uncertainty

Window-pick agreement between map and live was 4/6 on BOTH architectures, and
both misses were cells where the map's two windows differ by less than its
own noise: R4 0.9% apart (1.277 vs 1.288), R5 1.6% (1.338 vs 1.360). No
ranker could resolve those from that data. The defect is that the search
emits a pick instead of a tie-set, so the runtime cannot tell a real
preference from a coin flip — and switching on a phantom margin is strictly
negative once a switch costs ~20 ms (I5).

### I4 — engine-level run-to-run variance (llama), cause unknown

Replicate boots on **byte-identical prompts with greedy sampling**
(`load_regime` accepts a `seed` but never uses it; verified by hashing the
prompt sets):

| cell | S(boot 0) | S(boot 1) | swing |
|---|---|---|---|
| llama R5 w2048 | 1.3952 | 0.8153 | **-41.6%** |
| llama R5 w512 | 1.3406 | 0.9972 | **-25.6%** |
| dense, worst of 8 | — | — | -4.0% |

Discriminator already in hand: at llama R5 w512 acceptance barely moved
(2.957 -> 3.151) while throughput fell 25%, so this is **cost/kernel
variance, not acceptance** — the search is not at fault, the engine is.
Prime suspect: `enable_flashinfer_autotune: True` in the compilation config
(autotune selects kernels by boot-time timing, so different boots can land on
different kernels). Phase 88 already ships the escape hatch
(`R88_NO_AUTOTUNE`).

**This blocks everything on llama**: no throughput model can be validated
against data carrying 25-42% unexplained variance.

### I5 — the throughput model omits three measured terms

`S_K = (1 + f*K)/(K*R + 1)` is exact given tau and R, and the roofline fit is
good where margins are large (MLA b1: predicted 0.517, measured 0.529). But
phase 95 measured three terms the model does not carry:

1. **Prefill** — the model is decode-only (I1).
2. **Parked-engine cost** — the scheduler scores OFF at exactly 1.0, citing
   82-E0. That E0 measured toggle LATENCY, never the parked engine's
   THROUGHPUT. Measured: parked costs **2.5% at llama b16, ~0% at b8**. OFF's
   true score is cell-dependent and below 1.0.
3. **Transition cost** — **unpriced entirely, and dominant**. A 4.6% armed
   duty cost 4.0% throughput where steady-state arithmetic predicts ~0.4%;
   that is ~20 ms per arm/disarm cycle against an ~8 ms decode step. The 2%
   "arming rent" is a hand-set hysteresis constant, not a measured transition
   cost.

---

## What phase 95 established that STANDS (inputs, not debt)

- **The gate is worth far more than switching.** MLA unconditional spec loses
  **47% at b1** (S 0.529) with acceptance at **5.0 / 5.0 — essentially
  perfect**. The entire loss is cost (R~2.0), and no acceptance improvement
  could rescue it. C2's table predicted it to 2% (0.517 vs 0.529). Against
  that, dense's window-switching headroom is +0.9-2.7%.
- **The map transfers when the margin is large and fails when it is small.**
  MLA's 47% catastrophe transfers at 2% error; dense's 1-2% window preference
  does not survive the protocol gap. This is the honest scope statement for
  C2->C3.
- **Probes are asymmetric**: they cost 4% at a cell that should park (llama
  R8) and GAIN +2.1% at a cell that should arm (R4). Phase 91 proved they are
  load-bearing for drift detection, so this is a tuning problem, not a bug.
- E0's cross-seed method, the C2->C3 policy compiler, and the gate/probe
  harnesses all work and are reusable (phase 95 `scripts/`).

---

## Design under evaluation (user proposal, 2026-08-05)

Two rounds of offline profiling, then online tracking:

- **Round 1 (screen).** Per-cell candidate sets from minimal profiling,
  guessing unprofiled cells from profiled ones.
- **Round 2 (confirm).** The actual best lever **per regime**, chosen among
  Round 1's candidates.
- **Online.** During RL rollout, update the lever each step from the previous
  step's evidence.

**Assessment: the shape is right.** Making Round 2 per-REGIME rather than
per-CELL dissolves I2 without adding a map axis (a regime carries its own
prompt/generation split by construction), fixes I1 if Round 2 measures in the
deployment's currency, and mitigates I5's transition cost by making switching
coarse (per regime) instead of per step.

Three strengthenings are required:

- **S1 — interpolate COST, never acceptance.** C2's factorization: R is
  cell-dependent but physical (v2r's roofline, 3 params/arch); f is
  regime-only (CV 2.2-3.3%). So guess R across cells, measure f per regime.
  Interpolation may SHORTLIST under sound bounds that cannot eliminate a true
  winner (`f_comp >= prod(f_i)` admits, `<= min(f_i)*1.03` eliminates) and
  must never DECIDE. Phase 90's measurement theorem and C1's "Stage-A
  under-predicts MLA/MoE high-batch wins" are the measured warnings.
- **S2 — budget the pre-capture set as a design input.** Constraint C-C
  requires every candidate resident as a captured graph. 5 candidates x 9
  regimes = 45 configurations is not capturable. Round 2 must select from a
  small GLOBAL pool (~3-4 distinct (window, skip, K) combos); that budget
  flows BACKWARD into Round 1's shortlist size.
- **S3 — decide runtime regime identification.** Round 2's output is
  per-regime, but the engine sees only (batch, ctx) and cannot tell R5 from
  R5cot. RL rollout knows its phase; general serving does not. Resolution is
  the existing C2<->C3 unification: **the map picks the candidate set, live
  acceptance picks within it** — supported by E0 (live accept separates R5
  4.13 from R5cot 4.60).

## User-set constraints

- **C-A** Throughput modelling must be theory-precise; verify whether the
  implementation carries variance or the model is misaligned with the
  architecture's mathematics and the hardware. (I4 is the precondition; I5
  lists the three known omissions.)
- **C-B** A lever is `(composition, draft tokens)` — one action, not a
  boot-fixed config plus a per-step K. C2's search already treats it this way;
  the deployed runtime does not.
- **C-C** Switching must be efficient enough to be **hidden by the model
  forward**. Currently ~20 ms/cycle vs an ~8 ms decode step — about 2.5
  forward passes, NOT hidden.

---

## Work order (dependencies matter)

| # | task | depends | cost | why now |
|---|---|---|---|---|
| **W0** | Stack capability audit: lever binding matrix, runtime action space, gap list (`w0_infra_audit.md`) | — | 0 | **DONE 2026-08-05.** Main finding: the stack's one clean pattern is *boot-at-max, select-down* (K has it); window can join it in two grades (masked down-switch = acceptance-only, per-window graphs = cost-true); skip needs gated passthrough + per-set capture; quant and kernel realization are boot-class permanently. Runtime action today is K-only (C-B unmet); `{wnone}×{FULLCG}=∅` breaks lever-grid uniformity |
| **W1** | Root-cause llama's run-to-run variance (I4 = G5). N boots R5, autotune on/off, kernel-selection log. Bundle G6 (plumb `seed` through `load_regime` so seeds become content draws) | W0 | ~1 GPU-h | **DONE 2026-08-05** (`results_w1.md`). I4 = TWO sources. (A) flashinfer-autotune draft kernel lottery: 40.3% -> 0.7% under notune, accept locks at 3.124; notune >= best-of-tune, so every future run sets it. (B) external bimodal episode (~126/~142, −11.3%): survives notune AND core pinning, both lanes; co-tenant fabric/DRAM suspected, unproven; mitigated by episode-rejection protocol (reject rounds >5% below cell reference, ITERS>=4). **Gate MET**: spec 0.7%, AR fast-cluster 0.4–0.7%. G6 done (seed-0 byte-identical, seed-1 disjoint, verified) |
| **W2** | Separate TTFT/TPOT in the regime runner; record BOTH currencies per run (I1) | — | ~2 GPU-h | **DONE 2026-08-05** (`results_w2.md`). Currency gap has TWO mechanisms; the large one (+10-16%) is fixed non-decode time diluting strong TPOT gains, not prefill share; llama R5 has 25.6% of request time outside both phase timers. Clean llama is NOT flat (R5cot +50%, R1 w512 −28%). Deployed selector = two compensating errors: table R lottery-inflated 1.46× (true R from uncond identity: 0.40/0.43 vs 0.58/0.62) AND live f EMA +0.1 biased by optimistic pooling — w512@R5 parked 35% below its own physics. Currency decision: S_dec validates the model, S_e2e makes deployment claims, bridge terms measured. Protocol holes for W3: b1 rejection rule; long episodes need cross-boot certification |
| **W3** | **Pre-register** the threshold that justifies Round 2 + hidden switching, in decode-only currency | W2 | 0 | Avoid building for a number that may not survive the metric change |
| **W4** | Complete the throughput model (I5): prefill term, per-cell parked cost, per-transition cost. Probe discriminator already written (95/`run_e1p_probe.sh`) | W1 | ~2 GPU-h | C-A |
| **W5** | Round 1 redesign: interpolate R, emit tie-sets with uncertainty (S1, I3) | W4 | — | The search must express "cannot tell" |
| **W6** | Round 2: per-regime confirmation in the declared currency, over a global pool sized to the capture budget (S2) | W3, W5 | — | The core of the proposal |
| **W7a** | Runtime action `(composition, K)` at zero capture cost: policy schema gains a composition id; scheduler argmax over (comp, K); masked window DOWN-switch inside the boot graph (G1 + G2a) | W4 | — | Smallest change that makes C-B executable; acceptance-lever only (cost floor stays at boot window) |
| **W7b** | Cost-true hidden switching: per-window / per-skip-set captured graphs, multi-capture residency, side-stream selection (G2b + G3, C-C) | W6, W7a | — | Cannot hide a re-capture; pool sized by S2's budget = \|windows\| x \|verify widths\| (+ \|skip sets\|); G3 built only if Round 1 shortlists >1 skip set |
| **W8** | Online per-step lever update at RL rollout | W6 | — | Best-supported piece (phase 92: drift small, accept RISES over 128 steps) |

Order rationale: **W1 and W2 are cheap and unblock everything**; W3 is free
and prevents building toward an unconfirmed number. W5-W8 are the redesign
proper and should not start before W3's gate.

## Decision criteria

- **W1 gate**: llama replicate-boot swing must fall below ~5% (dense's level)
  once the cause is controlled. If it does not, llama is excluded from
  model-validation data and that exclusion is disclosed.
- **W3 gate (pre-registered before W6)**: per-regime lever selection must beat
  the best single static lever by a margin exceeding both the measurement
  noise floor and the transition cost, in decode-only currency. If it does
  not, the two-round pipeline is not built and the finding is that a
  well-chosen static lever plus the OFF gate is the deployable answer — which
  phase 95 already shows is worth 47% on MLA.
- **W7 gate**: switch latency must be under one decode step at the target
  batch, or switching stays boot-class/per-regime rather than per-step.

## Constraints

GPUs **0-1 only**; caches on `/data`; `.venv/bin/python`; commit prefix
`[W7][96]`. Pre-registration before scoring, every number a committed
artifact, refuted predictions retained (T11).

## Expected next artifact

`results_w1.md` — llama variance root cause, with the autotune on/off boot
matrix.
