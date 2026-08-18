# Status — work measured on the VM box (branch `research/self-spec-moe-vm`)

Written 2026-08-18. Everything here was measured on **cloud-9ezI3Q**, the
QEMU/KVM guest, and is kept on its own branch so a VM number can never be
mistaken for one from the scored h103/h104 campaigns. The main branch is
untouched at `e85b57353`.

## 1. What this box can and cannot measure

The single most useful thing learned here, because it decides which claims are
admissible:

| comparison | round-to-round stability | admissible? |
| --- | --- | --- |
| armed cell vs armed cell | ~1% | **yes** — this is the selector's actual job |
| anything vs OFF | 9–16% | only with a ±10% band |

OFF is the **most** state-exposed cell in the grid, not the least: the clamp is
a fixed per-step host delay, which is enormous against an 8 ms parked step and
modest against a 40 ms armed one. Normalising by OFF was tested and makes
drift ~10x worse, so it is never used.

## 2. Results

**G98-F — the fail-closed rule** (`results_g98_f.md`). Built, tested,
committed, parameter-free (thresholds are the registered arming rent and the
closed G98-C envelope). In-sample it fixes D3's R4 miss and flips the static
clause to PASS. Out-of-sample it **refutes its own premise**: R4 measures
1.129x / 1.131x / 1.196x over OFF across three independent runs here, against
h103's 0.763x, with the R1 control reproducing h103 to 1%. Six lines of
evidence put the anomaly on h103's R4 column. **The rule is not enabled** —
here it would cost 2.6% of aggregate throughput.

**G98-G — the whole selector, end to end on one box**
(`results_g98_g.md`, `results_g98_g_fullgrid.md`). 17 gated cost boots → fit →
prediction map frozen behind a digest → full 31-cell grid, six regimes, two
reversed rounds. **Run twice:**

| | grid 1 | grid 2 |
| --- | --- | --- |
| selector / omniscient | 0.9820 | 0.9839 |
| selector / OFF | 1.415x | 1.426x |

Two independent 62-boot grids agreeing to 0.2 points. D3 on h103 measured
0.951 over the same lattice. The search reproduced h103's picks in 5 of 6
regimes despite a materially different cost surface (quantization is worth 25%
here against 38% on h104).

**R6 misrank — diagnosed and deliberately not shipped**
(`results_g98_g_r6.md`). Not the cost model: `D_hat` under-predicts every cell
by 17.7–21.2%, a near-uniform bias that cannot misrank. It is the acceptance
transfer, optimistic in proportion to skip depth (+3–6% at skip0, +16–17% at
skip8). A 256-token burn-in measuring acceptance on the deployment content
fixes the pick (regret 12% → 0%) and cuts mean per-regime regret 4.4x — but
**costs 0.51% of end-to-end throughput**, because R6 carries 2.2% of the time
in a token-weighted mix while the burn-in's noise costs ~1.5% at R1, which
carries 59.3%. Per-regime regret is the wrong currency for deciding where to
spend measurement; time-weighted contribution is.

**G98-H — MoE breadth** (`results_g98_h.md`). **Tensor parallelism works in
this fork** — the recorded "TP>1 fails" was two config faults
(`draft_tensor_parallel_size` must equal TP; `custom_all_reduce` needs the
NCCL fallback). With them fixed a 30B MoE target and its 15 GB W4A16 draft
boot together under TP=2, which is what makes the quant axis reachable on MoE
at all. **But the measurements are unusable**: median round-to-round
difference 26.3% at R1 against ~1% for the dense grid. No lever conclusions
are drawn. The instability is localised — the two *unlevered* cells are stable
at 0.1–1.8% while every windowed/skipped cell swings 25–41%.

**Artifact layer** (`scripts/w98_artifacts.py`). A filename collision silently
dropped a cell from the first full grid (31 reported, 30 measured), resolved
differently in each round, and a repair-by-filename mislabelled an armed
measurement as OFF. Identity now lives in record content: names carry every
axis, unknown axes are refused, plans prove injectivity before spending GPU
time, resume verifies content before skipping, and coverage is counted from
records. The rerun self-audits clean. 16 tests replay the real failures;
suite at 323.

## 3. Open questions this box cannot settle

* **Is D3's R4 regression real?** Needs 6 replicate boots on h103. Until then
  the phase's "single highest-value fix" is unconfirmed and the fail-closed
  rule stays built-but-disabled.
* **Absolute comparability** with the h103/h104 scored campaigns — this box is
  6–10% slower everywhere and quantization is worth less here.

## 4. Candidate next steps

Listed with cost so they can be chosen between; none are started.

| # | step | cost | what it buys |
| --- | --- | --- | --- |
| A | **MoE at TP=1**, same 8 cells, shared weights | ~45 min | Separates "levered paths interact badly with TP" from "levered paths are more box-exposed". Decides whether MoE breadth is reachable here at all. |
| B | **Round 2 proper on this box**: measure acceptance for the candidate set instead of transferring it | ~1 h | Removes the transfer error at its root rather than correcting it; would likely close most of the remaining 1.6% gap to omniscient. |
| C | **Time-share-weighted burn-in**: measure acceptance only for the slow regimes (R1 alone is 59% of time), long enough to beat its own noise | ~1 h | The R6 work's own recommendation, and the only version of that fix that could pay. |
| D | **h103 R4 replication** | 6 boots, needs h103 | Settles the one open scientific question; unblocks or kills the fail-closed rule. |
| E | **Port to main**: decide what of this branch merges | no GPU | The artifact layer and the TP fixes are box-independent and useful anywhere; the VM measurements are not. |

**Recommendation if a single item is picked: (A)**, because it is cheap and it
decides whether the MoE arc continues on this box or moves; then **(B)**,
which attacks the largest remaining source of selector error with a method the
design already prescribes.
