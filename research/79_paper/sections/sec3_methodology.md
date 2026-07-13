# §3 Measuring the two surfaces (draft)

> Source: Phase 76 (E0 preflight + gates, `results_e0.md`), Phase 77
> (anchor gate, `results_accept.md`). ~1 page + gates table.

## 3.1 Design principle

Every quantitative claim in this paper traces to one of two measured
surfaces — R (cost) and β (acceptance) — or to an end-to-end run. The two
surfaces are measured independently, by different methods, each carrying
its own calibration gate against ground truth; the selector composes them
and is then audited end-to-end. This section describes both instruments
and the traps each gate exists to catch — traps we hit, not hypothesize.

## 3.2 The cost surface R

R = TPOT_lever/TPOT_bf16 per (batch, context) cell, measured in SERVE mode
— one server per arm, decode-step time under steady occupancy — because
serve-mode is what the selector prices: paging, scheduling, and batching
effects are inside the number. Method gates:

- **Infra parity (preflight).** The numerator and denominator must run the
  same attention backend, kernel set, and CUDA-graph mode unless the lever
  itself changes them — and vLLM silently flips these under levers: fp8
  e5m2 KV flips the attention backend to FlashInfer; MLA+fp8-KV flips to
  FLASHMLA; TP-EP has NO dispatch collective at all (so a "local-route is
  free" measurement on TP-EP measures nothing). The preflight asserts the
  full backend/kernel/graph configuration per arm before any cell is
  recorded.
- **Warm pass per cell.** The serving benchmark reuses prompts per seed,
  so a first pass measures decode interleaved with chunked prefill — up to
  4.7× TPOT inflation at b32/16k, large enough to INVERT a kernel
  comparison (§4). Every recorded cell is a warmed re-run.
- **Over-capacity detection.** Cells where the bf16 denominator cannot
  hold the batch are marked, not ratio'd (vLLM v1 logs no preemption
  keyword; the signature is a persistent scheduler wait queue).
- **Cross-method anchor (S1).** Serve-mode TPOT agrees with an independent
  offline two-output-length slope measurement to 2% — two instruments, one
  number.
- **Noise flags (S4).** DP-placement bimodality at low batch on the MoE
  stack: medians of 4 runs, flags retained in the dataset rather than
  scrubbed.
- **Matched denominators.** Skip arms compare dummy-weight to dummy-weight
  (dummy ≈ real validated at 1.00 ± 0.02 dense, 0.95–1.15 MoE).

## 3.3 The acceptance surface β

β = per-token draft acceptance under rejection sampling, measured OFFLINE
by teacher forcing: cache the target's reference continuations and
per-position softmax once per (architecture, context), then score each
lever's draft distribution at the same positions. The draft shares the
target's KV exactly as the lever dictates (window = sliced pages with
original RoPE; draft-only KV-quant = quantized read of the same cache), so
β isolates the lever's distributional damage. Method gates:

- **The anchor gate.** Before the sweep is trusted, measured β composed
  geometrically into an accept length must reproduce real end-to-end
  spec-decoding accept lengths: −1.6% (dense, 16k, K=4) and +0.3% (MoE,
  32k, K=6). A method that cannot reproduce known τ does not get to
  produce new β. (The gate's scope is also a limitation we later paid for:
  it covered dense and MoE, and the un-gated MLA e2e path hid a harness
  defect — §9.)
- **Sample-size floor.** ≥12 prompts × 96 positions (~1,150) per cell:
  per-prompt β spreads 0.91–1.00 over one bank, and a 4-prompt estimate
  missed by −9%. Per-prompt β also jitters run-to-run at numerics
  boundaries (router argmax flips); only the position aggregate is stable,
  and we never quote per-prompt values.
- **Pairing.** All arms within a cell score against the SAME references at
  the SAME positions, so cross-lever deltas cancel prompt-sampling
  variance. The distribution-robustness check (§5) re-pairs everything on
  a second prompt bank.

## 3.4 Composition and audit

The selector composes the surfaces as speedup = τ_β(γ)/(γR+1) — a formula
independently validated at ~90% delivery (γ=3) before this work relied on
it — and §6's validation ladder then tests every use the model is put to:
in-grid fit, held-out lever, forward prediction, architecture transfer.
End-to-end runs audit the composed claims at seven cells across two
architectures (§7–8). The delivery term itself became a measured object
(the launch floor, §8) rather than a fudge factor.

[Table: the gate ledger — gate / what it catches / incident it caught /
where recorded. Fig: two-surface pipeline diagram.]
