# Phase 95 — C3: deploying C2's searched map at runtime

## STATUS: STOPPED 2026-08-05 (not closed) — continued in phase 96

User directive after E0 and the partial E3: stop C3 and resolve the
foundational issues E0 exposed first. **The measured results below stand and
are inputs to `research/96_selector_foundations`**, which carries the issue
list, the two-round redesign, and the work order.

Completed and trustworthy:
- **E0** (window-switching envelope, 2 boots x 2 arches x 6 regimes): dense
  +2.78%/+2.65% cross-seed on accept-binding regimes (P1 CONFIRMED), llama
  +0.08%/+0.00% (P2 REFUTED). See `results_e0.md`.
- **E3 partial** — MLA `off` + `uncond` only. The headline finding is
  complete and does not need the missing arm: unconditional spec loses **47%
  at b1** (S 0.529) with acceptance **5.0/5.0**, i.e. essentially perfect
  acceptance and a pure cost loss, predicted by C2's table to 2% (0.517).
- The R8 diagnosis (parked-engine cost + unpriced transition cost) and the
  P8 refutation (window is a cost floor, not only an accept tradeoff).

NOT done: E3 MLA `gated` (OOMed on a memory-release race; runner since fixed
to gate on free memory), E3 MoE (all arms), E1 multi-capture, E1' probe
discriminator (`scripts/run_e1p_probe.sh` is written and ready to run).

Reusable assets: `scripts/compile_from_c2.py` (C2->C3 policy compiler,
round-trip verified to 0.0124%), `score_e0.py` (cross-seed selection),
`run_e3_gate.sh`, `diag_r8_gate.sh`, `parse_kpick.py`, `run_e1p_probe.sh`.


Source phases: 82 (runtime switching, compiled policy in the scheduler),
89 (DRAM lever swap, 113 ms pinned), 91 (bandit stage-3 ladder, multi-seed
RL results), 88 (the canonical 9-regime x real-dataset eval), 93 (C1 grid),
94 (C2 composition search + weight sharing). User directive 2026-08-05:
"proceed with C3 with MLA/MoE as gate arms".

## The gap this phase closes

C3's deployed policy (`vllm/v1/core/sched/scheduler.py:1142-1260`) reads a
compiled table whose per-cell `options` are `{K, R, S_ref, f_ref}`. The
**lever identity is boot-fixed**; only K/OFF switches per step. Every C3
number to date (1.090x +- 0.044 drift, 1.300x rollout) was produced with one
hand-chosen lever and a policy that picks depth.

C2 emits a per-cell winner over singles ∪ compositions. Today's runtime can
act on the K slice of that map only. This phase extends the action space from
K to (config, K) and measures what that is worth **on the deployment path**.

## Action space and cost classes

The axes differ sharply in switch cost:

| axis | mechanism | cost class |
|---|---|---|
| K / OFF | propose fewer steps | free (deployed today) |
| KV window | `_apply_draft_kv_window` rewrites fixed-max-width buffers in place, but the window sets `n_kept` = the scratchpad gather shape (`_scratchpad_n_kept_blocks`) | **capture-shape-keyed**: one capture set per window; switch = graph select (E1 prices it) |
| layer skip | replaces decoder layers with passthroughs AND deletes their `static_forward_context` attention registrations — "load-time static: the layer set never changes after boot" (`draft_model.py:328`) | **boot-class** (not hot-switchable in the current design) |
| quant ckpt | separate resident checkpoint | **paid: 113 ms pinned DRAM swap** (89-E0) |

### Scope decision: switch the WINDOW axis, hold skip boot-fixed

Axis attribution over K-only (`scripts/headroom.py`, 2-sample basis):

| arch | +window | +skip | +both |
|---|---|---|---|
| dense | **+2.04%** | +0.76% | +2.57% |
| llama | **+2.28%** | +0.08% | +2.29% |

The window axis carries **79% of dense's and 99.6% of llama's** switching
gain, and it is the tractable one (multi-capture over the 2-3 windows a
deployment's map actually uses). Skip is the hard one — switching it would
have to re-register KV specs — and it is worth **+0.76pp on dense, +0.08pp on
llama**. So skip stays **boot-fixed**, chosen once per deployment from the map
(what C1/C2 already support). This is a priced decision, not an omission: the
forgone gain is recorded above, and P1/P2 predict the window-only system we
are actually building.

The structural finding that shapes this phase: **within every architecture,
C2's per-cell winners hold the quant lever CONSTANT** (dense `q-hum` in all 8
cells; llama `q-w4a16` in all 8) and vary only window/skip/K. So exactly ONE
draft checkpoint is ever resident — the same as today — and the expensive axis
never has to switch. Predicted consequence: zero DRAM swaps fire on
within-architecture traces (P4 below).

> **Correction (2026-08-05, after the initial commit of this README).** An
> earlier phrasing credited this to C2's weight sharing. That was wrong:
> `_weight_sharing_enabled` (`draft_model.py:146`) requires the draft's
> checkpoint AND quantization to equal the target's, so weight sharing applies
> only to `q-none` drafts. Dense and llama both carry a quantized draft, so
> weight sharing is inactive in these arms. What makes the swap unnecessary is
> the quant-constancy finding alone. Weight sharing re-enters only for a
> deployment whose map picks `q-none`.

## Arms (the accounting ladder)

Each arm isolates one layer, so the deltas attribute:

1. `AR` — no speculation
2. `static` — best single (config, K), fixed for the deployment
3. `recipe` — published fixed recipes: KnapSpec skip set; EfficientRollout
   ctrl (both already reproduced, phases 86 / 91-E6-effroll)
4. `konly` — config fixed, K/OFF per step (**today's C3**)
5. `full` — (config, K) per cell from C2's map (**the new arm**)
6. `oracle` — same picks, switch cost zeroed (upper bound)

Claim under test: `full - konly`, and what fraction of `oracle - konly` it
captures.

## Arm assignment per architecture (user decision 2026-08-05)

- **dense, llama = switching arms.** C2's map puts 5 distinct configs across
  8 cells for each; this is where the headroom is.
- **MLA, MoE = gate arms.** Both beat AR only at b32/b64 (C1 Stage B: MLA
  11/33 all at b32/b64 on `w8chan_k2`; MoE 8/33, 7 of them b32/b64). C2's
  composition grid samples that region once (b32/c2000) and finds composition
  worth only +2.1% / +1.9% there. Their C3 claim is therefore a SAFETY claim,
  not a speedup claim: the runtime must disarm where the map says spec loses
  (MLA/MoE b1 sit at S 0.65-0.69 — unconditional spec burns ~35%). Measured
  gate value on the compile-cell basis: **MLA +22.1%, MoE +21.4%** over the
  best unconditional static config — larger than the entire switching headroom
  on dense/llama, and it is what today's K/OFF policy already delivers.

  Disclosed scope limit: C2's MLA/MoE grid has 7 cells with only one at b32
  and none at b64 — one cell deep exactly where they win. This phase does NOT
  claim "composition does not help MLA/MoE"; it claims the grid samples that
  region once. Extending the grid to b32/c8000 + b64 is the recorded optional
  follow-up.

## Pre-registrations (recorded BEFORE any run)

Magnitudes come from `scripts/headroom.py` over the committed phase-94
compile-cell measurements. The predictions are about **live serving on real
datasets** — a different protocol — so they are falsifiable, not restatements.

- **P1 (dense window switching).** Trace-weighted `full` over `konly` lands in
  **[+1%, +4%]** on traces that cross the cells where the map's window winner
  changes. Compile-cell basis: **+2.04%** (window-only, 2-sample; +2.6% if
  skip could also switch). Concentrated: b32/c8000 +6.4%, b32/c2000 +5.4%,
  b8/c2000 +4.3%; three cells at +0.0-0.5%.
- **P2 (llama window switching).** Same quantity in **[+1%, +4%]**. Basis:
  **+2.28%** (window-only; skip switching adds +0.01pp — nothing). Llama is
  the CLEANEST test of the config axis: its K-only arm adds +0.0% over static
  (K2 wins every cell), so all of llama's switching value is the window axis
  by construction.

  Both bases are 2-sample truth (mean S over the R1+R2 content draws). The
  single-sample R1 figures were higher (dense +4.2%, llama +2.6%) -- the
  winner's-curse gap C2 measured, applied here to our own prediction before
  it could flatter the result.
- **P3 (MLA/MoE null).** `full` ≈ `konly` (|delta| < 1%): the OFF gate is the
  whole story. Basis: +0.0% on both.
- **P4 (free-axis switching).** Zero DRAM checkpoint swaps fire on
  within-architecture traces; all switches are metadata-only.
- **P5 (no thrash).** On a trace that stays in one cell, `full` >= `konly`
  - 1%.
- **P6 (map transfer — the real C2->C3 claim).** Per-cell predicted gain from
  C2's map correlates positively with measured live per-cell gain
  (Spearman rho > 0, reported with n and p).
- **P7 (gate safety, MLA/MoE).** On b1/b8 real-data regimes the gate keeps
  aggregate S >= 0.98; at b32/b64 it arms and beats AR.
- **P8 (null control) — REFUTED BY MEASUREMENT, 2026-08-05.** Registered as:
  R6's envelope is < half the discriminating regimes'. The premise was that a
  window cannot matter where it does not bind. **Measured false**: at R6 (end
  ctx 318) dense runs **+5.0%** faster on w512 with acceptance IDENTICAL
  (4.733 vs 4.734), llama **+12.0%**. Mechanism:
  `_scratchpad_n_kept_blocks` gathers `n_sink + ceil((window+K)/block_size)`
  blocks *every draft step regardless of the live context length*, so the
  window sets a fixed per-draft-step COST floor and w2048 pays for KV it does
  not need. R6 discriminates through cost even though it cannot through
  accept. Consequences: no window-inert regime exists, so **no valid null
  control is available**; bias control falls entirely to cross-seed selection;
  regimes are regrouped by mechanism (accept-binding vs cost-only). The
  finding strengthens the switching case rather than weakening it — at short
  context a narrow window is strictly better (same accept, less cost), so the
  window should track context length for cost reasons alone.

  Original registration retained above per T11.
  A window can only matter where it BINDS: measured end contexts (prompt +
  generation, seed 0) are R4 8691, R5 14549, R5cot 17193, R8 2166 -- all
  crossing both windows -- while **R6 is 318 and crosses neither**, so the
  map's window choice cannot affect it. R6 therefore estimates the noise
  floor of the envelope metric. If R6's envelope matches the discriminating
  regimes', E0 is measuring noise and the gate is NOT read as passed,
  whatever the aggregate says. (R1 at 1087 crosses 512 only -- reported
  separately, not counted in either group.)

Registered risks (stated now, not after the fact):

- **R0 — the envelope is a max over noisy arms.** E0's envelope takes a max
  over three windows per regime, which is the winner's-curse structure C2
  measured on its own search: under noise, E[max] exceeds the true max, so the
  raw envelope OVERSTATES the headroom. Two corrections are built in, and the
  raw max is never the headline. (i) The R6 null control estimates the bias
  directly (window cannot act there, so its envelope is pure max-of-noise);
  the reported figure is discriminating MINUS null. (ii) With two seeds,
  cross-seed selection picks each regime's window on one seed and scores it on
  the other, removing the curse rather than estimating it -- the same fix C2's
  2-sample truth scoring applied to its rankers. **A single-seed raw envelope
  is not sufficient to pass the gate.**

  Note the gate metric is AR-INDEPENDENT: envelope/konly is a ratio of two
  spec arms sharing one AR denominator, so AR boot variance cancels exactly
  and moves only the absolute S values.
- **R1 — resolution.** The effect is +2-5%; per-phase noise on these boots ran
  +-8-10% (phase 91). Only a **paired, multi-seed, same-boot AR anchor per
  seed** design can resolve it (that design resolved +4.7 points across 3
  seeds in 91). >=3 seeds required; unpaired results will not be reported as
  evidence.
- **R2 — 8B lever uniformity.** All nine phase-88 8B regime winners sat on one
  kernel with K as the selector. If real traces do not cross the window/skip
  boundaries the compile-cell map crosses, `full` has nothing to do. "K-only
  suffices at 8B; config switching pays where kernel winners split" is a
  legitimate registered outcome, not a failure.
- **R3 — b1 content luck, and llama's concentration.** C2 found 2 of 5 b1
  wins dissolve under re-drawn documents. The runtime carries C2's tie-break
  rule (prefer content-robust configs among near-ties at single-stream cells)
  rather than chasing a one-document margin. Sharpest instance: llama's
  aggregate is dominated by ONE cell (b1/c8000, +14.0%; every other llama cell
  is <= +2.6%), and that cell is a b1 cell whose composition margin C2's
  multi-draw validation already reduced to a tie. **Llama results are
  therefore reported twice: all cells, and b1-excluded.** If the b1-excluded
  llama gain is ~0, P2 is not supported regardless of the aggregate.
- **R4 — MoE quant diversity vs the gate.** MoE's per-cell full picks span
  three quant levers (`q-none`, `q-w4a16`, `q-w8chan`) -- but every cell that
  would require a quant switch sits BELOW AR and is gated OFF, so P4 (no DRAM
  swaps) still holds for the deployed arm. If a future grid extension puts a
  winning MoE cell on a different quant than its neighbours, P4 is void for
  MoE and the 113 ms swap re-enters.

## E0 OUTCOME (2026-08-05) — see `results_e0.md`

| arch | cross-seed gain (accept-binding) | gate |
|---|---|---|
| dense | **+2.78% / +2.65%** both directions | PASS (long-context workloads); +0.9% mixed |
| llama | +0.08% / +0.00% | FAIL |

- **P1 CONFIRMED** (registered [+1%,+4%], basis +2.04%).
- **P2 REFUTED** (registered [+1%,+4%], basis +2.28%): llama delivers ~0%.
  Not because no window preference exists, but because the policy's own
  boot-to-boot decision noise (-8% to -42% on R5, AR anchor stable to 0.2%)
  is an order of magnitude larger than the effect it should exploit.
- **P8 REFUTED** (see below): the window is a cost floor, not only an accept
  tradeoff.
- **Gate decision: E1 is NOT built yet.** The same measurement surfaced a
  gate-safety defect (llama R8 armed ~94% of steps at S=0.935 when it should
  have disarmed) and llama's decision instability, both tracing to one cause:
  the live-f estimate is unreliable near the `f = R` break-even (arming is
  `f > R` exactly). Fixing the estimator is worth more than the switching gain
  and must precede a switcher that would inherit the instability. New step
  **E1' (estimator repair) precedes E1 (multi-capture)**; see results_e0.md.

## Plan

**Realizability constraint found in E0 (2026-08-05).** The switching set is
`{512, 2048}`, not `{512, 2048, none}`: booting the deployed stack with a
full-context draft raises `VLLM_SELF_SPEC_DRAFT_FULLCG requires
VLLM_SELF_SPEC_DRAFT_KV_WINDOW > 0 (the scratchpad materialises the
sinks+window key set)`. Phase 94 met the same constraint and handled it by
running `w-none` PLAIN, disclosing the realization split (measured worth
-0.5% mean). Running it plain HERE would compare across two stacks inside the
very metric under test, so the `none` arm is dropped instead. This is the same
physics phase 91's E6-effroll reported from the other direction: a
full-context draft is unaffordable on H100 -- here the deployed stack does not
even admit one. The map's picks for the discriminating (long-context) regimes
are 512/2048 anyway, so the dropped arm costs little.

- **E0 — the switching ENVELOPE on real data (no engine change; gates
  everything else).** Measure before building. Today's runner already boots
  one window per engine (`run_e6d.sh:44` uses a per-window policy table), so
  the `oracle` arm is reachable now: boot each window in the map's set, run
  the same regime cells, and take the per-cell max. That envelope is the
  UPPER BOUND on what any switcher could deliver, measured on the deployment
  path, and it yields P6 (map transfer) directly.

  **Gate: if the envelope over `konly` is < 1% on real traces, the switching
  thesis is refuted on the deployment path and E1 is not built** — C3's
  deliverable reduces to the K/OFF policy on a map-chosen static config, with
  the forgone gain recorded. This ordering exists so a negative result costs
  one measurement instead of an engine build.

- **E1 — window multi-capture (only if E0's envelope justifies it).** The
  draft's `CudagraphDispatcher` keys on batch descriptor only; window sets
  `n_kept`, so each window needs its own capture set and the key must gain a
  window dimension. Budget from committed logs (E6d, the deployed config):
  capture is 0.28-1.25 GiB and 2-14 s per set, so 2-3 windows cost ~+1-2.5
  GiB and ~+10-30 s boot — affordable against 80 GiB, especially with the
  15.3 GiB weight sharing freed. Measure incremental capture time/memory,
  steady-state switch latency, and an **accept A/B around every toggle**
  (P35/82 rule: accept, never step-time alone).
- **E1 — policy schema v2 + compiler.** `options: [{config, K, R, f_ref}]`;
  scheduler argmax over (config, K); two switch-cost classes; amortization
  gate `dwell* = cost / delta_rate` (82-E3). Compiled FROM C2's search output
  (`scripts/compile_from_c2.py`) — this script is the C2->C3 handoff artifact.
- **E2 — switching eval (dense, llama).** Arms 1-6 on the phase-88 regime
  suite, paired multi-seed per R1. Runs only if E0's gate passes.
- **E3 — gate eval (MLA, MoE).** Arms 1, 2, 4 on the same suite; P7.
- **E4 — RL half.** Drift trace + GRPO thinking rollout with the (config, K)
  policy; compare against the recorded 1.090x / 1.300x incumbents.
- **E5 — map-transfer analysis.** P6 + where the map mispredicts live. The
  91-E6-val survivor-bias finding (drain tails are low-accept; cell prices
  call them spec-heaven) says live-f must override map-f — the C2<->C3
  unification, measured.

## Decision criteria

- E0: a lever whose toggle costs > ~1 s or perturbs accept by > 0.05 is NOT
  hot-switchable and leaves the v2 action space (recorded, not worked around).
- E0 gate: envelope over `konly` >= 1% on real traces, else E1 is not built
  and the phase closes on the registered R2 outcome.
- E2 gate: `full` >= `konly` on the trace aggregate at >= 2 of 3 seeds, and
  P5 holds. Failing that, the deliverable is R2's registered outcome with the
  measured reason.
- E3 gate: P7.

## Constraints

GPUs **0-1 only** (user instruction 2026-08-05; 2-7 unavailable). Caches on
`/data`; `.venv/bin/python`; commit prefix `[W7][95]`.

## Expected next artifact

`data/c3_headroom.json` (prediction basis), then `results_e0.md` with the
refreshed toggle-cost table.
