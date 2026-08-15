# Preregistration amendment 2 — the instrument, for D3 currency

Date: 2026-08-14
Status: **DRAFT — awaiting approval** (protocol per amendment 1: an explicit
APPROVED stamp before it binds anything).

Amends: the measurement instrument for **future timing-currency
measurements only** (D3 and any later cost work). Rounds 1 and 2 are closed
and stay exactly as measured — they carry the same instrument and remain
comparable to each other. **D2 is untouched by design**: acceptance is a
token-counting measurement (per-position reached/accepted counters,
bootstrap over requests), which no timing instrument can bias.

## 1. What is now known, with both rounds in hand

1. **X25 (section 7 of the clamp doc):** 8 of the profiler's 12 per-step
   `cudaDeviceSynchronize` calls sit INSIDE the `draft_chain` region being
   measured, costing **+2.4 ms/step of real time destroyed** (not
   misattribution — the profiler-independent `mean_armed_step` moves by the
   same amount). Removal was demonstrated safe for CUDA graphs across four
   interleaved boots.
2. **Round 2's fit does NOT show simple absorption.** Section 7
   hypothesised the fitted `F` would absorb the instrument cost. Fitted `F`
   per regime: R1 3.256, R4 3.577, R5 2.432, R5cot 2.751, R6 1.932, R8
   3.331 ms, against 3.66 ms registered from checkpoint bytes. If a
   constant exposed instrument cost dominated `F`, hiding-ordered regimes
   would order `F` accordingly; R4 (most hidden) carries the LARGEST `F`.
   The floor is regime-structured for physical reasons, and the instrument
   share cannot be cleanly subtracted after the fact — which is precisely
   why the instrument should stop paying it going forward.
3. **X29b:** ~150-168k clock reads per batch-1 engine step (~4 ms at
   healthy vDSO speed, ~10% of the step), independent of the profiler.

## 2. The amendment

For every timing measurement taken after this amendment is approved:

* `SelfSpecProfiler._fine_only()` also gates `draft_forward`, so the
  default-on sync count per step drops from 12 to 4 and no sync remains
  inside the `draft_chain` region. (The X25 monkeypatch, made permanent.)
* The change is recorded in the authorization of any campaign that uses it,
  and no number measured under the new instrument is compared to a Round-1
  or Round-2 absolute without naming the +2.4 ms difference.

**Why before D3:** D3's claim is comparative in decode currency — the
selector against static configurations **including OFF**. The inner-sync
cost taxes only armed configurations (OFF has no draft chain), so the
current instrument biases exactly the comparison D3 scores. Removing it is
conservative bookkeeping, not tuning: it removes a known artifact that
today works AGAINST armed selections in marginal K/OFF decisions.

## 3. Explicitly out of scope

The ~160k clock reads/step are an ENGINE property, not an instrument
property: reducing them changes the system under measurement everywhere.
That optimisation (with the VM-only mmap storm of section 6) is registered
as post-phase engineering, not amended into the campaign midstream.
