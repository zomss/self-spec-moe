# Results (Stage A): Comm-Bound Decode -- Regime + Composed Lossless Speedup

Date: 2026-06-26. Real Qwen3-30B-A3B, attention-DP(8) + EP, dummy weights,
`enforce_eager=False`. Decode step time measured on two **real** transports (no
injection): NVLink (`NCCL_P2P_DISABLE=0`) and forced sockets
(`NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_NET=Socket`).

## 0. Blocker: the PCIe-exact knob hangs this box (proven)

`NCCL_P2P_DISABLE=1` -- the knob that would force the EP all-to-all onto host-staged
PCIe -- **hangs even a trivial 2-GPU all-gather** here (60s timeout; P2P-on completes
instantly). This is the same transport wall as Phase 02 (forced IB) / Phase 20
(real-IB loopback): on this NVSwitch box you cannot create a PCIe-exact comm-bound
regime. **Forced sockets DO work** (TCP loopback), so they serve as a *real*
comm-heavy endpoint -- pessimistic (loopback is slower than PCIe-P2P), but real data
movement, not injection. NVLink (comm ~free) and sockets (comm-heavy) bracket the
real PCIe point.

## 1. Regime confirmed: EP decode is heavily comm-bound off NVLink

| global batch | S_nvlink (ms) | S_socket (ms) | f = (S_socket-S_nvlink)/S_socket |
| ---: | ---: | ---: | ---: |
| 8 | 14.71 | 92.62 | **0.841** |
| 32 | 15.19 | 117.56 | **0.871** |
| 128 | 15.33 | 138.52 | **0.889** |

NVLink step is ~flat at ~15 ms (weight-read-bound compute; comm ~free, confirming
Phases 12/16). Off NVLink, **84-89% of the step is communication**, and f **grows
with batch** -- exactly the machine-balance prediction (comm scales with batch while
low-batch compute is weight-read-bound). This is the empirical, real-transport
version of the positioning table: the comm-bound regime is real and large.

## 2. Composed lossless speedup (local-routing draft, comm eliminated)

Compute is identical with comm on/off, so a comm-free local-routing **draft** runs at
`S_draft ~= S_nvlink` even on the comm-bound box, while the bf16 full-EP **verify**
pays the comm-bound step `S_verify`. With measured acceptance `beta` (Phases
18/22/23):

```
speedup = T(k,beta) * S_verify / (k*S_draft + S_verify),  T = 1 + beta*(1-beta^k)/(1-beta)
```

Best-k speedup over the comm-bound baseline (`pcie/Nx` = socket comm deflated N* as a
PCIe-P2P estimate; real point is between NVLink=1x-no-win and socket):

| global B | beta | socket (measured) | pcie/2x | pcie/4x (conservative) |
| ---: | --- | ---: | ---: | ---: |
| 8 | 0.82 | 2.14x | 1.67x | 1.34x |
| 8 | 0.92 | 2.91x | 2.09x | 1.57x |
| 32 | 0.82 | 2.35x | 1.83x | 1.43x |
| 32 | 0.92 | 3.24x | 2.33x | 1.71x |
| 128 | 0.82 | 2.51x | 1.95x | 1.50x |
| 128 | 0.92 | **3.50x** | 2.54x | 1.85x |

**Decision: GO.** The comm-bound regime is real (f=0.84-0.89, real socket transport),
and the composed lossless speedup is **>= 1.3x across the entire plausible PCIe
operating range** (even deflating socket comm 4x at the low beta=0.82), rising to
**1.6-1.9x** at beta=0.92 and **2-3.5x** at the measured socket point. Win grows with
batch (the throughput regime). Proceed to Stage B (integrated scheduler).

## 3. Honest caveats

- **Socket is a pessimistic comm proxy.** TCP loopback is slower (more latency-bound)
  than real PCIe-P2P, so the socket column over-states comm. The `pcie/Nx` columns
  deflate it; the true PCIe point needs a real PCIe-only box (the deflation factor is
  an estimate, not measured -- the one thing this hardware cannot pin down).
- **Component-measured, not integrated.** S_draft uses the NVLink step as a
  compute-only proxy for the local-routing draft; the real lockstep scheduler
  (Stage B) measures end-to-end tokens/s and verifies losslessness directly.
- **No DBO-overlap baseline** (needs DeepEP) -- the key reviewer comparison is still
  open. The speedup here is over plain comm-bound EP, not over an overlapped EP.
- **beta is one-step (Phases 18/22/23)**; multi-token compounding is in the T(k,beta)
  term. Verify is exact bf16 -> losslessness by construction.
- Dummy weights (timing only); FP4 *compute* speedup not on H100 (the win measured
  here is the *communication* axis, which is the point).

## 4. Bottom line

On real transports, MoE EP decode off NVLink is **84-89% communication**, and a
lossless comm-free draft + exact verify nets **>=1.3x (conservative PCIe) to 3.5x
(measured sockets)**, growing with batch. The central *benefit* claim -- modeled
until now -- is, for the first time, grounded in measured step times on real fabric.
The PCIe-exact operating point and the DBO baseline remain the two things this box
cannot produce; everything else says GO for the integrated scheduler.
