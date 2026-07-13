# Phase 82 — runtime lever switching: the strategy map as a scheduler policy

Source phases: 78 (selector: speedup = τ_β(γ)/(γR+1) over measured maps),
80 (composition + map v4), 81 (floor-free chain — the prerequisite: with
delivery ≈ 100%, the map's static rankings are real e2e rankings, so a
switching policy inherits trustworthy per-regime payoffs). Roadmap item (c),
user-set 2026-07-14: "make real system switching the lever at each regime."

## Objective

Turn the static strategy map into a LIVE serving policy: the engine observes
its regime (batch occupancy, context mix) and switches the draft lever
configuration at runtime — including to OFF. **Gate: on a workload whose
regime shifts (batch ramp and/or context growth), the switching policy beats
EVERY static configuration on aggregate tok/s (target ≥1.10× over the best
static), and is never worse than the per-regime static winner by more than
5% inside any regime.** Fail-honest: if switch costs or regime dwell times
make static-per-deployment the right answer, that is the phase's finding.

## Why this is the right next phase

- The maps say different regimes want different configs (76-E1 crossovers:
  W4→window along ctx×batch on dense; OFF→window on MoE; b8 γ*=4 vs b32
  γ*=6 in 81-E3). A deployment whose load shifts crosses those boundaries.
- 81 removed the confound: pre-fix, a "switching win" could be an artifact
  of the floor taxing configs unevenly. Post-fix, per-regime payoffs are at
  roofline, so policy deltas measure the POLICY.
- vLLM already ships one degenerate switch (`disable_by_batch_size` — spec
  OFF above a batch threshold): the existing primitive is our baseline and
  the evidence that the serving world wants this knob.

## The lever-toggle cost question (E0 — everything downstream keys on this)

Hypothesized toggle classes, to be MEASURED (per-toggle latency + resident
memory + accept perturbation):

| lever | expected mechanism | expected class |
|---|---|---|
| OFF ↔ on | skip/enable propose (disable_by_batch_size generalized) | free |
| K (γ) down/up ≤ K_max | propose fewer steps; verify width fixed at K_max — does padding tax the low-K regime? | cheap, maybe padded-verify tax |
| window size (incl. → ∞) | proposer metadata only (`_apply_draft_kv_window` params) — but scratchpad graphs are captured at a FIXED cap → recapture or multi-capture | cheap eager / capture-set cost under FULLCG |
| draft quant (W4 ↔ bf16-shared) | W4 draft is a separate resident checkpoint (~4.7 GB); both-resident = memory rent, swap = load latency | expensive unless pre-resident |
| kv-quant / skip / local-route | out of scope for switching v1 (map says niche/OFF regions) | — |

## Plan

- **E0 — toggle cost anatomy**: instrument one engine; toggle each lever
  class in isolation under steady load; measure switch latency (TTFT blip /
  TPOT spike), memory delta, accept before/after. Deliverable: the toggle
  cost table + which levers are hot-switchable. Also: verify K-down at
  runtime doesn't fight `num_speculative_tokens`-sized buffers.
- **E1 — regime detector + policy**: signal = (running batch size, mean
  effective ctx) smoothed with hysteresis (dwell-time floor from E3's
  amortization law); policy = 78-selector lookup over the hot-switchable
  config set (v1: {OFF, window-K4, window-K6, W4+window-K4/K6} on dense).
  Implementation surface: W7 serve path or engine-side step hook; config
  changes must land between cycles (never mid-chain).
- **E2 — the shifting-workload demo (the gate)**: scripted load — batch
  ramp b4→b32→b8, ctx mix 4k→32k, ~10 min trace — replayed against every
  static config and the policy. Report aggregate + per-regime tok/s, switch
  count, switch-cost total.
- **E3 — the amortization law**: min dwell time per regime for a switch to
  pay: dwell* = toggle_cost / Δrate(regime). Closes the loop to roadmap (b):
  the offline maps price Δrate; E0 prices toggle_cost; together they give a
  provably-safe hysteresis constant instead of a tuned one.

## Assumptions / constraints

- Box: GPUs dynamically shared — probe free memory at launch (81-E0 lesson);
  caches/temp on /data. GPUs 0-3 as of 2026-07-13.
- Dense (Qwen2.5-7B + W4 draft) is the v1 target — richest measured map,
  floor-free chain proven. MoE switching only if v1 lands early (its fixed
  chain also works: 81 MoE check, +12%).
- Sampling/serving semantics must be untouched by a switch: accept A/B gate
  around every toggle (P35/81 discipline — accept, never step-time alone).
- No new training, no new kernels expected; this is plumbing + policy.

## Decision criteria

- E0: a lever with toggle latency > ~1 s or accept perturbation > 0.05 is
  NOT hot-switchable — falls out of the v1 config set (recorded, not hacked
  around).
- E2 gate as stated above; if unmet, the deliverable is the measured reason
  (toggle costs vs dwell times) and the static-per-deployment
  recommendation — which still feeds the paper's discussion and roadmap (d).

## Expected next artifact

`results_switching.md` with the E0 toggle-cost table first; then the policy
module + E2 demo numbers. Feeds: paper follow-on section (outline §9 note),
roadmap (b) via the E3 amortization law, roadmap (d) — an RL post-training
rollout worker is exactly a regime-shifting workload (batch drains as
episodes finish, ctx grows within episodes).
