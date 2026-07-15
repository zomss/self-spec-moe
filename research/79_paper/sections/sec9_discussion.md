# §9 Discussion, limitations, and open surface (draft)

> Honest-ledger section: every limitation is stated with its mitigation
> or its price. ~0.75 page.

**One box, one stack.** All measurements are single-node H100 ×4 on one
serving stack. The split §6 validated is the portable claim: the
R-selector transfers (one anchor cell, 9.3%), absolute TPOT does not (h,
BW_eff are stack constants). Deployments re-run the 91-minute protocol of
§7.5, not our sweep.

**Noise floors we report rather than hide.** MoE DP-placement bimodality
bounds V0 at ~6.8% (medians of 4, flags retained); the b16/32k spec cells
carry 17% run variance (error bars quoted); the over-capacity detector
false-positives on cold prefill (affected cells kept by manual inspection,
noted in the dataset).

**The anchor-gate lesson (paid for, then repaid).** The β anchor gate
covered dense and MoE; the un-gated MLA end-to-end path concealed a
harness defect that our own capture law later diagnosed (FLASH_ATTN_MLA's
captured chain is not replay-safe; fixed, acceptance 2.00 → 5.92 default).
The episode is a validation asymmetry worth stating as practice: every
architecture whose numbers a map carries needs its own end-to-end anchor,
even when the offline method is calibrated elsewhere. It also left OFF on
MLA measured three ways — priced (≤1.13× optimistic over 53 configs),
mechanistic, and observed (best challenger 0.54×; even β≈1 delivers
0.56× on the eager MLA chain).

**Harness lever fidelity.** The runtime fp8 draft is W8A8 while the map's
q_fp8 β is weight-only — a −2.1 accept gap on MLA at K=5. A weight-only
runtime draft option is plumbing, not research, but until it exists the
map's q_fp8 column prices a config the harness cannot yet run exactly.

**Profiled flips: delivery boundary measured, not closed.** Of the four
map-v5 flips (§7.6), one is delivered (b4/2k, 1.03× — margins ±0.10, a
parity-to-modest win); the 2k band's higher batches are chain-blocked
(0.63–0.64×, needed cycle 27 ms vs measured 44 ms). The acceptance side
transfers exactly everywhere (2.81–2.88 vs offline 0.953), so what
remains is the windowless-MoE instance of the §8 execution program plus
the partial replica's memory rent (~half the expert bytes per rank) —
both quantified. Profiled sets also inherit a monitoring burden a naive
shard does not: routing frequencies can drift with workload; the
distribution-robustness check (§5.2) bounds this for our two banks, not
for all traffic.

**The deferred fabric.** The comm-bound tier (PCIe/multi-node) is where
the refined local-route law predicts its largest wins and where the MoE
comm-free triple (β 0.82–0.83, assembled and measured) is parked. The
original thesis of this project lives there, with its acceptance side now
fully measured and its cost side one sweep away.

**The draft-only KV pool.** Twice motivated and still unbuilt: it is the
lever that would extend the serviceable envelope (§4-F3's scoping — the
only claim shared-KV self-spec structurally cannot reach) and the QK-norm
rule prices exactly where it is safe (fp8-K on normed architectures;
V-only otherwise, 25% of KV bytes for free).

**Remaining headroom on the fixed chain.** Verify idles 31% of its
forward; step-0's residual over a pure chain step is the last ~6% to the
1.98× ceiling at 16k. Neither gates any claim; both are known dials.

**Search scope.** The backtest replays one architecture-as-new (MLA — the
one the record supports honestly); a second replay (MoE-as-new) would
strengthen §7.5. The searched combo space is bounded (≤4 levers, one per
class, fixed lever settings); per-layer assembly à la KnapSpec, per-lever
setting search à la SparseSpec, and online refinement à la Not-a-Bandit
all compose with the maps rather than compete with them.

**Runtime switching.** With delivery ≈ 1 and per-regime winners measured,
the natural system is one that switches levers as its regime shifts —
batch ramps, context growth — using the map as scheduler policy and §7.5's
amortization arithmetic for hysteresis. Scoped as follow-on work; this
paper establishes the maps, the selector, and that the selected configs
deliver.
