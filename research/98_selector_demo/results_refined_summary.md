# The refined grid: the selector measured against a deployment baseline

Consolidated record of sections 29-44 of `results_refined_lo.md`, written
2026-08-20. Every number is measured and traceable to a committed artifact;
that document is the authority and this is the reading. Nothing here is new
measurement.

**What this arc is.** `results_phase98_summary.md` records the phase's
registered claims on the six-regime R-grid, scored against OFF. This arc
moves to Phase 100's length-quadrant grid (LI/LO/LIO/SS), scores against
**stock vLLM** rather than against our own parked runtime, and ends with the
selector's own decision rule run end to end. It supersedes nothing in that
document; it measures a different grid and a different baseline.

---

## 1. The headline

| | |
| --- | --- |
| selector, share of omniscient | **99.61-99.62%** |
| over the best single static configuration | **+1.97% to +2.30%** |
| over stock vLLM | **~1.20-1.24x** |
| points picked exactly | **6 of 8** |
| switching value available | **+2.37%** |
| switching value captured | **83%** |

Eight `(cell, batch)` points, four arms common to all of them, equal token
mix, time-weighted, instrument-free, quantized draft, natural EOS. Ranges
rather than point values because section 44's replication found one cell's
baseline unstable; the two ratio claims are insensitive to it and the
absolute multiple is not.

### The measured grid

Best arm at each point, against that point's own stock boot:

| cell | batch | best arm | vs stock | `off` vs stock |
| --- | --- | --- | --- | --- |
| LI | 8 | `woff/skip4` | 1.063 | 0.915 |
| LI | 16 | `w1024/skip4` | 1.128 | 0.919 |
| LIO | 8 | `w1024/skip4` | 1.232 | 0.930 |
| LIO | 16 | `w1024/skip4` | 1.331 | 0.944 |
| SS | 8 | `woff/skip4` | 1.055 | 0.751 |
| SS | 32 | `woff/skip4` | 1.344 | 0.812 |
| LO | 8 | `w1024/skip4` | 1.358 | 0.856 |
| LO | 16 | `w1024/skip4` | 1.664 | 0.910 |

---

## 2. What the arc established

### 2.1 The prediction transfers; the selection is worth about 2%

One coefficient set, calibrated at one context and two batches, ranks a grid
spanning **75 to 19,147 prompt tokens** and batches 8 to 32 well enough to
pick the omniscient arm at six of eight points. Both misses are near-ties —
2.4% and 0.7% — at or below this box's 1.2% replicate spread.

That is the two-round design's premise holding: cost predictable offline from
single-lever profiles, acceptance measured where it must be.

The switching value it unlocks is **+2.37%**, of which the selector captures
83%. On a single-batch slice of the same grid the value was **+0.15%**
(section 39), because one arm sat at or near the top of every cell. Sweeping
batch is what created a selection problem at all.

### 2.2 The value is in arming and in not choosing badly

| decision | worth |
| --- | --- |
| arming at all, over the parked runtime | **+35.5%** |
| not picking a bad lever (best static over worst) | **+11.3%** |
| choosing between good levers per point | **+2.0%** |

A deployment that picks `w1024/skip4` once and never revisits it gets most of
the available benefit on this workload family. The selector earns its place
by *finding* that arm and being right about why — which is what makes it
transfer to a grid where the answer is not constant — more than by switching.

This is the Phase-82 dwell-time law arriving a third time, now against a
stock baseline.

### 2.3 Quantization is load-bearing, not a convenience

The quantized draft wins every arm at every cell by **6.5% to 34.9%**, and at
LI and SS it decides whether speculation is worth doing at all: every bf16 arm
falls below stock at those cells (0.796-0.925 and 0.772-0.787) and every
quantized arm clears it.

Section 27 gives the mechanism and its limit. Quantization moves exactly one
bucket of the step — draft body GEMMs, 20.253 ms to 10.827 — and leaves
attention (+0.087), verify GEMMs (-0.005) and `lm_head` (+0.011) inside
noise, because kernels serialize and each lever shortens only what it
touches. It delivers **30% of the speedup its bytes promise**: Marlin
sustains 39.5% of peak bandwidth against the bf16 GEMM's 81.9%, so 3.88x
fewer bytes buys 1.87x less time.

### 2.4 The window's price, and where levers interact

Acceptance measured per cell (section 35): the 1024 window costs **31% of
acceptance at LI**, 19% at LIO, and **nothing at SS** — where `w1024/skip4`
and `woff/skip4` read 4.779 to three decimals, because a 1024 window cannot
bind on a 75-token prompt. At LI the same window wins on throughput anyway:
at 12-17K of context the KV traffic it removes outweighs the tokens it gives
up.

Sections 26 and 33 located why the levers are not separable. The window's
value is **(attention time removed) - (host time exposed)**, and only the
first term is predictable from bytes. Quantization removes 10.8 ms of device
work without touching attention, which shrinks the pool of device work
available to hide host work behind — so **the levers interact through the
host, not through the GPU.**

---

## 3. The measurement findings, which may outlast the numbers

Four published conclusions in this arc were reversed, each by the same
question: **what does this baseline pay that the other does not?**

| section | the asymmetry | size |
| --- | --- | --- |
| 21, 25 | different generation lengths under natural EOS | up to 22 points |
| 30 | scoring against `off` (our parked runtime) instead of stock | 17-22 points |
| **32, 33** | **the profiler and koff trace, on our side only** | **5-58 points** |
| 34, 44 | a co-tenant process, on unreplicated single boots | 47.6% |

The instrument case is the sharpest. `boot_environment` set
`VLLM_SELF_SPEC_PROFILE=1` and a trace path on every armed boot and on `off`,
while stock had neither. Removing it **reordered** the arms rather than
shifting them: at LI b16 the windowed arms gained 56-58% where everything
else gained 11-14%, moving them from below the parked arm to above stock. An
instrument that costs host time reorders levers that differ in host work,
which is exactly what section 26 said the window is.

The confirmation that the diagnosis was right is external: instrument-free,
our parked overhead reads 0.915 and 0.930 against Campaign 1's 0.933 and
0.925 on different hardware — **agreement to 2% where the instrumented
numbers were 10-15 points apart**, with nothing tuned to make them meet.

### Two registered protocol consequences

* **Calibrate the cost model at the longest context and largest batch the
  deployment will see.** A coefficient unidentified at the calibration point
  is not merely imprecise elsewhere — it is *unbounded*, because nothing
  constrains its sign. Fitted on 156-token prompts, `kappa_kv` came out
  negative and the model claimed reading 15,000 KV positions is 4.2 ms
  cheaper than reading 4,252 (section 36). Calibration at the long end
  transfers *down* to a cell with 60x less context; the short end does not
  transfer up (section 38).
* **An acceptance curve must be truncated at the cell's natural output
  distribution.** Under `ignore_eos`, past the natural stopping point the
  draft predicts filler almost perfectly: at SS, whose median output is 153
  tokens, four of seven arms read `tau_k4` = 5.000 in the bucket covering
  u >= 256 — every drafted token accepted, every step (section 35).

---

## 4. What is not established

* **One box, one model, one stack.** Qwen3-8B dense, vLLM V1, TP=1, w4a16
  Marlin, a KVM guest with a documented host-side clamp. Section 28 records
  what transfers: the method and the mechanisms, not the coefficients, whose
  re-fit costs one singles sweep (~19 boots).
* **The armed discrepancy with h103 is unexplained.** Their `w4a16` at LI b8
  reads 0.708 against our comparable arm at 1.003 — a factor of **1.42**. The
  instrument closed the *parked* gap to 2% and closes none of this. G98-F
  asked for six replicate boots on h103; they have not been run, so two
  campaigns disagree and only one side has been re-measured.
* **SS b32's baseline is unstable.** Greedy is not bit-deterministic across
  boots, and at a 2.4-3.1 s run with a draining batch the *longest* request
  sets the wall clock: stock moved 454 tokens to 350 between boots, a 26%
  swing. The remedy is Campaign 1's `n = max(16, 2b)` requests rather than
  `n = batch`; it is not applied here.
* **The bf16 family has no prediction map**, only measurements. It is needed
  to verify the selector *rejects* bf16 at LI and SS, which section 34 shows
  it should.
* **The window-size axis is at the design's resolution limit.** Sections 37,
  42 and 43 each improved a coefficient without moving the LIO b16 miss; the
  remaining error is 3.4 points on a near-tie between two arms of the same
  lever family.

---

## 5. Where it leaves the design

The two-round selector works, and the honest statement of what "works" means
has three parts:

1. **Round 1's elimination rule is sound by construction** and is not what
   was ever at risk. It kills only what cannot pay at perfect acceptance.
2. **Round 1's cost model transfers**, but only if calibrated where its
   dominant terms dominate. Every failure in this arc was a coefficient
   identified at the wrong operating point, and each was fixed by moving the
   calibration rather than by changing the model's form.
3. **Round 2's acceptance must be measured per workload**, truncated at the
   natural length, and is stable across batch to within 4% — section 41
   refuted `tau(u, B)` as a source of ranking error at the one point where it
   could have mattered.

What the arc does not support is the framing that per-workload *switching* is
where the value lies. On this grid it is worth 2.4%, the selector collects
2.0%, and arming at all is worth 35%.
