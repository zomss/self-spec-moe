# Preregistration amendment 2 — the instrument, for D3 currency

Date drafted: 2026-08-14. Revised 2026-08-15 (§3 framing corrected, dual-arm
option added).
Status: **DRAFT — awaiting approval** (protocol per amendment 1: an explicit
APPROVED stamp, dated, before it binds anything). The researcher stamps this,
not the agent that drafted it.

Amends: the measurement instrument for **future timing-currency
measurements only** (D3 and any later cost work). Rounds 1 and 2 are closed
and stay exactly as measured — they carry the same instrument and remain
comparable to each other. **D2 is untouched by design**: acceptance is a
token-counting measurement (per-position reached/accepted counters,
bootstrap over requests), which no timing instrument can bias. D2 may
therefore proceed before this amendment is settled.

## 1. What is now known, with both rounds in hand

1. **X25 (section 7 of the clamp doc):** 8 of the profiler's 12 per-step
   `cudaDeviceSynchronize` calls sit INSIDE the `draft_chain` region being
   measured, costing **+2.4 ms/step of real time destroyed** (not
   misattribution — the profiler-independent `mean_armed_step` moves by the
   same amount). Removal was demonstrated safe for CUDA graphs across four
   interleaved boots. **Identified and documented before Round 2 ran.**
2. **Round 2's fit does NOT show simple absorption.** Section 7
   hypothesised the fitted `F` would absorb the instrument cost. Fitted `F`
   per regime: R1 3.256, R4 3.577, R5 2.432, R5cot 2.751, R6 1.932, R8
   3.331 ms, against 3.66 ms registered from checkpoint bytes. If a
   constant exposed instrument cost dominated `F`, hiding-ordered regimes
   would order `F` accordingly; R4 (most hidden) carries the LARGEST `F`.
   The floor is regime-structured for physical reasons, so the instrument
   share cannot be cleanly subtracted after the fact.
3. **X29b:** ~150-168k clock reads per batch-1 engine step (~4 ms at
   healthy vDSO speed, ~10% of the step), independent of the profiler.

## 2. The change proposed

For every timing measurement taken after this amendment is approved:

* `SelfSpecProfiler._fine_only()` also gates `draft_forward`, so the
  default-on sync count per step drops from 12 to 4 and no sync remains
  inside the `draft_chain` region. (The X25 monkeypatch, made permanent.)
* The change is recorded in the authorization of any campaign that uses it,
  and no number measured under the new instrument is compared to a Round-1
  or Round-2 absolute without naming the +2.4 ms difference.

## 3. The direction of the bias, stated plainly

An earlier draft of this amendment called the change "conservative
bookkeeping, not tuning." **That was wrong and is corrected here.**

D3's claim is comparative in decode currency: the selector against static
configurations **including OFF**. The inner-sync cost is paid only by ARMED
configurations, because OFF runs no draft chain. The current instrument
therefore makes the selector look WORSE than it is, and removing it makes
**D3's claim easier to satisfy, not harder**.

That is the opposite of conservative, and it is the whole reason this needs
a dated stamp rather than a silent edit. The reviewer's challenge writes
itself: *you removed an instrument cost after discovering it was hurting
your result.* The honest answers:

* The cost was identified, measured, and documented **before Round 2 ran**,
  not after seeing an unwelcome D3 number (X25; §7 of the clamp doc).
* It is an artifact of the PROFILER, not a property of the system under
  test. The deployed selector runs no profiler, so an unamended D3 charges
  the selector for instrumentation that production never pays.
* The A/B is cheap and already demonstrated (four interleaved boots), so
  the claim need not rest on the correction being taken on trust.

## 4. The three options

* **(A) Approve.** D3 measures the system rather than the instrument.
  Accepts an instrument discontinuity with Rounds 1-2 and the critique
  above.
* **(B) Defer or reject.** All three rounds share one instrument; the bias
  runs against our own hypothesis, which is unimpeachable. Cost: D3's
  headline understates the selector by up to ~2.4 ms/step on armed arms.
* **(C) Approve with a dual-arm requirement — RECOMMENDED.** Take D3's
  decisive comparison under BOTH instruments. Report the corrected
  instrument as primary and the legacy instrument as a registered
  conservative bound. If the claim holds under both, the critique in §3 is
  answered by measurement rather than by argument; if it holds only under
  the corrected instrument, that dependence is itself a reportable result
  and must be named, not buried.

Under (C) the registered reporting rule is: **both numbers appear wherever
the D3 result appears.** A dual-arm run is not licence to quote whichever
arm is kinder.

## 5. Explicitly out of scope

The ~160k clock reads/step are an ENGINE property, not an instrument
property: reducing them changes the system under measurement everywhere.
That optimisation (with the VM-only mmap storm of section 6) is registered
as post-phase engineering, not amended into the campaign midstream.

## 6. Stamp

```
Status:  [ ] APPROVED (A)   [ ] APPROVED (C, dual-arm)   [ ] DEFERRED   [ ] REJECTED
Date:
By:
Note:
```
