# Results: overlapped spec gives World A a role -- at inter-node

Date: 2026-06-29. Analytic overlap model (`overlap_model.py`); k=1 (comm-bound optimum).
Overlapped cycle = max(C_v + C_d, M_v); baseline (DBO, no spec) = max(C_v, M_v);
speedup = (beta+1)*baseline/cycle. C_v=1 (verify compute), M_v=R (verify comm),
C_d = draft compute (World A full forward = 1; EAGLE head = 0.1).

| regime | R=comm/compute | f | EAGLE (b.80) | World A local (b.85) | World A FP4 (b.92) |
| --- | ---: | ---: | ---: | ---: | ---: |
| PCIe serving | 1.2 | 0.55 | **1.80** | 1.11 | 1.15 |
| PCIe high-batch | 1.9 | 0.65 | 1.80 | 1.76 | **1.82** |
| crossover | 2.0 | 0.67 | 1.80 | 1.85 | **1.92** |
| inter-node IB | 4.0 | 0.80 | 1.80 | 1.85 | **1.92** |
| inter-node | 9.0 | 0.90 | 1.80 | 1.85 | **1.92** |

## The draft-cost-vs-quality tradeoff FLIPS at R=2 (f~0.67)

- **Single-node PCIe (R<2):** the draft COST dominates the cycle (overlap can't hide the
  draft because comm isn't big enough). A CHEAP draft wins -> **EAGLE** (World A's
  full-forward draft adds compute -> 1.1-1.8x, below EAGLE).
- **Inter-node (R>2):** the verify's comm is large enough to fully hide the comm-free
  draft -> the draft is FREE -> only beta matters -> **World A's full-model draft (beta
  0.92) beats EAGLE's small head (beta 0.80)**, and it is **training-free**.

This is the key insight: **at single-node you want a cheap draft (EAGLE); at inter-node you
want a high-fidelity draft (World A full-model), because the draft is hidden for free and
only its acceptance counts.** The expensive full-forward draft -- a liability everywhere
else -- becomes the right choice once comm >> compute.

## Why this matters

It is the first composition where **World A both works WITH conventional spec and BEATS it**:
the spec module runs continuously and World A's comm-free local-routing/FP4 draft is hidden
behind the inter-node all-to-all, delivering a higher-beta draft for free, training-free.
And it lands exactly in the **inter-node regime** -- the project's real-win territory
(Phase 20: f=0.6-0.8) that single-node PCIe only approximates.

## Caveats / what must be confirmed (hardware-gated)

1. **Perfect-overlap assumption** -- real DBO overlap is ~70-90% efficient; the crossover R
   shifts up a little and the inter-node win shrinks modestly. Still positive for R>>2.
2. **Data dependency** -- draft i+1 needs verify i, so the overlap requires
   speculative-ahead drafting (draft assuming acceptance; discard on rejection). Free when
   overlapped, wasted on rejection -> net-positive at high beta; needs a real scheduler.
3. **beta gap** -- assumes World A draft beta (0.85 local / 0.92 FP4) > EAGLE (~0.80). If a
   given EAGLE head exceeds 0.85, World A-local ties and only World A-FP4 (0.92, needs the
   full-coverage cache, ~16 GB) still edges. Per-model beta needed.
4. **Inter-node hardware** -- the whole win is at R>2 (f>0.67), i.e. a network hop. Needs a
   real multi-node IB box to measure (single-node PCIe sits just below the crossover).

## Verdict

The user's "spec every step" (overlap) is the right idea for composing World A with
conventional spec -- but its advantage is **inter-node only**. There it flips World A's
expensive draft into an asset and lets a **training-free** self-spec **beat a trained EAGLE
head**. This is the strongest positive direction for the project, and it is gated on a
multi-node measurement.
