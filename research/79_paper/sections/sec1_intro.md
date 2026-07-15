# §1 Introduction (draft)

> Framing final; contribution numbers final (incl. the 2.77× promotion).
> ~1 page.

A practitioner serving model M at batch b and context c faces a concrete
question: *should this deployment self-speculate — draft with a cheapened
copy of itself, no trained head, no second model — and if so, with which
cheapening?* The menu is real: weight quantization, windowed attention
over shared KV, draft-only KV-cache quantization, layer skip, restricted
expert routing. Each has a paper arguing for it. None of those papers
answers the practitioner, because the answer changes under their feet: a
lever that wins at batch 1 loses at batch 32; a lever that is free on one
architecture destroys acceptance on another; in whole regions of the space
the right answer is not to speculate at all.

This paper treats that question as a measurement problem. Speculative
payoff factors into two surfaces — a cost ratio R (how much cheaper the
draft's step is) and an acceptance rate β (how often the target keeps the
draft's tokens) — and both are measurable once, offline, per lever and
architecture, with calibration gates against ground truth. From the two
surfaces, selection is free: exhaustive search over priced configurations,
including composed ones and including OFF.

Contributions, each with its number:

- **C1 — the cost map** (§4): serve-mode R for 7–9 levers × 9 regime
  cells × dense/MoE (139 ratios, MLA by transfer), with two clean
  crossovers, an OFF region, and the finding that weight-quant — the
  community default — is a parity band on expert-parallel MoE on both
  kernel stacks.
- **C2 — the acceptance map** (§5): teacher-forced β for 12 levers × 3
  architectures (102 cells, anchor-gated to ≤2%), yielding a portability
  verdict — only fp8 weight-quant carries across architectures — and the
  QK-norm rule: KV-path normalization decides draft-only fp8-K viability,
  with a V-only design rule for un-normed models. Both verified on a
  second prompt distribution.
- **C3 — the selector** (§6): a term-decomposed cost model validated up a
  ladder (held-out lever 2.2%; forward 2.9%; one-anchor architecture
  transfer, R 9.3%), composed into a strategy map that matches 5/5
  end-to-end ground-truth cells — with OFF backed by exhaustive search
  over the full combination space, not lever exhaustion.
- **C4 — the composition law** (§7): β_combo ≈ Πβ_i (median deviation
  ~0.01, 48 paired cells) with two mechanistic exceptions; combo cost
  priced by term edits; and a profiling-budget backtest showing 91
  GPU-minutes of protocol recovers the map within 2.6% of optimal (§7.5).
- **C5 — the floor-free chain** (§8): two execution laws — FA3-family
  captured schedules are replay-safe iff attention geometry is constant;
  compacted decode steps must trim rejected-slot KV — that turn composed
  rooflines into delivered numbers: **1.91× measured exactly at the
  registered roofline (delivery ≈ 100%) at dense b32/16k, growing to
  2.77× at 32k context**, +12% on MoE with the identical stack, and an
  out-of-sample diagnosis of an MLA backend failure the laws were not
  derived from.
- **C6 — profiled levers** (§5.4, §7.6, §8.6): offline profiling changes
  what the map says. Frequency-profiled expert selection (+0.13 β from
  measured routing skew) flips four MoE cells, one delivered end to end
  (1.03×) through a partial-replica loader that ships with the paper;
  layer-set placement doubles dense skip β (the middle-block convention,
  not contiguity, is the naive arm's error); calibration polishes values
  without moving argmaxes. The architecture-split law: the
  decision-changing profiled lever is the one aligned with where the
  architecture's redundancy lives — and even profiling STRATEGY fails to
  port across architectures.

The anti-contributions are load-bearing: three map regions say OFF (and
the exhaustive search plus a measured challenger say it stays OFF); the
envelope-extension property belongs to standalone configs, not shared-KV
self-speculation; and every trap our gates caught is reported as
methodology, because a map built without them inverts real decisions.

Everything is training-free, measured on one serving stack (the R-selector
transfers; absolute TPOT does not — §6), and every claim traces to a
surface measurement or an end-to-end run (§3).
