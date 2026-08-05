# W1 — llama run-to-run variance: root cause matrix

Data: `data/w1/w1_llama_*.json` (first matrix), `data/w1/w1b_llama_*.json`
(W1b pin discriminator). Runner: `scripts/run_w1_boot.py`, drivers
`run_w1.sh` / `run_w1b.sh`, scorer `score_w1.py`. All boots: llama R5,
seed-0 prompts byte-identical across every boot (G6 guarantee), greedy,
3 timed rounds per boot, e0's exact LLM kwargs apart from the toggles.

## First matrix: {off, w2048} x {tune, notune} x 3 boots

| cell | boot toks | between-boot | accept |
|---|---|---|---|
| off / tune | 140.3, 126.3, 142.0 | **+12.4%** | — |
| off / notune | 142.8, 142.0, 126.2 | **+13.2%** | — |
| w2048 / tune | **139.2, 195.3, 193.6** | **+40.3%** | 2.964, 3.124, 3.11 |
| w2048 / notune | 195.6, 194.3, 194.9 | **+0.7%** | 3.124, 3.124, 3.124 |

Within-boot spread is ≤0.6% everywhere except the bad tune draw's first
round (+18.5% — the boot is slow from its first token, not warming up).

## Finding: I4 is TWO sources, not one

### Source A — flashinfer autotune's draft-side kernel lottery. RESOLVED.

With autotune ON (the engine default at O1+, i.e. every phase-88–95 run),
a spec boot draws kernels by boot-time timing: 2 of 3 draws landed at
~195 tok/s, 1 at 139 tok/s — a **40% gap, boot-scoped, stable within
boot**. With autotune OFF the spec arm sits at 194.3–195.6 (0.7%) and
acceptance locks at exactly 3.124 across all three boots. The lottery
even perturbs accept (2.964 vs 3.124): different GEMM kernels, different
rounding, occasionally different draft tokens.

Two corollaries:

1. **notune ≥ best-of-tune.** Autotune never beat the fixed default here;
   its bad draws cost 29%. Turning it off is not a trade, it is free.
2. **Committed llama spec numbers are lottery samples.** Phase-95's
   "S(boot0)=1.395 vs S(boot1)=0.815" was good-draw vs bad-draw. C2's
   llama compile cells were measured under the same default and need a
   notune spot-audit before they feed model validation (W4/W5).

Internal validity: tune boots take ~39 s to engine-up vs notune's ~27 s —
the autotuner demonstrably ran only in the tune arms.

**Decision: every future measurement in this arc sets
`kernel_config={"enable_flashinfer_autotune": False}`.** (Verdict on the
map: P-W1a as literally pre-registered is REFUTED — notune does not bring
BOTH arms under 5% — because a second source exists:)

### Source B — engine-wide bimodal AR mode (~126 vs ~142, ±13%). PENDING W1b.

The AR anchor flips between two discrete levels **independent of
autotune** (slow mode hit tune/b1 AND notune/b2), tight within boot
(≤0.6%). Per-boot snapshots exonerate the obvious suspects:

- **Not co-tenant GPU load**: fastest boot (142.8) ran beside a neighbor
  at 74 GB/util 100%; a slow boot (126.2) ran beside an idle neighbor.
- **Not clocks/thermal**: GPU1 held 1980 MHz, 45–53 °C, every boot.
- **Not NUMA sockets**: the box is single-node (192 CPUs, all GPUs node 0).
- Lane asymmetry: 6/6 GPU0 (spec-lane) boots fast; GPU1 (AR lane) 2/6 slow.

This matters because **AR is the denominator of every S** — a ±13% anchor
poisons every ratio regardless of how stable the spec arm is.

W1b (pre-registered in `run_w1b.sh`): AR-only, notune, both lanes,
taskset-pinned (disjoint quiet core sets) vs unpinned, 3 boots each.

### W1b result: pinning does not prevent it — the mode is an external EPISODE

Per-round analysis (boot medians hide it; episodes lasted 1–2 of 3 rounds
in this matrix, vs whole boots in the first — episode LENGTH varies):

| cell | episode rounds | fast cluster | spread |
|---|---|---|---|
| g0 / nopin | 1/9 | [140.7, 141.6] | +0.6% |
| g0 / pin | **2/9** | [140.9, 141.9] | +0.7% |
| g1 / nopin | 2/9 | [126.1→] [141.8, 142.4] | +0.4% |
| g1 / pin | 0/9 | [141.8, 142.7] | +0.6% |

**P-W1b1 REFUTED** — a boot pinned to 16 quiet cores spent two rounds in
the slow mode. **P-W1b2 REFUTED** — both lanes exhibit it. New facts from
the round granularity: the episode always begins at boot and resolves TO
fast (never fast→slow mid-boot here); levels are two discrete points
(~126 / ~142, −11.3%) with fast-cluster spread <1% — an 8x-wide empty gap
between modes.

Eliminated: autotune, CPU pinning, GPU lane, co-tenant GPU utilization
(fastest boot ran beside util-100% neighbor; slow beside idle), SM clocks
(1980 MHz throughout), thermal (45–53 °C), NUMA (single-node box),
system load (1-min loadavg 39–44 across every boot, slow and fast).
Remaining suspect: **co-tenant fabric/DRAM contention** (GPUs 6–7 run a
third party's job on the same NV18 mesh; a steady collective would cut
bandwidth by a quantized amount — matching the discrete level — in
episodes we cannot see from nvidia-smi). On a shared box this is
external: detectable, not controllable. Suspected, not proven (T11:
recorded as such).

### Protocol decision (P-W1b3 branch): episode rejection, not prevention

For every timed measurement in this arc:

1. Reference = max round across the configuration's boots.
2. Reject rounds >5% below reference (the threshold sits in the empty
   gap between modes; nothing legitimate lives there).
3. A boot must keep >=2 rounds to score; otherwise re-run it.
4. Rejected-round counts are REPORTED with every artifact.

Retrospective check on the first matrix: under this rule the spec/notune
cell keeps all 9 rounds (worst round −4.0% off reference, below
threshold) — the 0.7% swing stands without rejection.

## Gate status: MET, with disclosed protocol

- Spec arm: notune, no rejections needed — between-boot swing **0.7%**,
  accept locked at 3.124. The throughput model validates against this.
- AR anchor: notune + episode rejection — fast-cluster spread
  **0.4–0.7%** per cell.
- Both are far under the 5% gate. llama is ADMITTED to model validation
  under: `enable_flashinfer_autotune=False` + episode rejection.
  (Pinning: neither helps nor hurts; not adopted, keeps the boot path
  minimal.)

## Consequences for the rest of the arc

1. Every future run: notune. The W2 driver defaults accordingly.
2. C2's llama compile cells were measured under autotune's lottery — a
   notune spot-audit of llama R cells is REQUIRED before W4/W5 uses them.
3. Phase-95's llama E0 numbers (both "seeds") are lottery samples; the
   llama window-envelope refutation (P2) should be re-read after W2
   reruns it notune.
4. AR anchors everywhere inherit episode rejection; ITERS should be >=4
   so a 1–2-round episode still leaves >=2 scoring rounds.
