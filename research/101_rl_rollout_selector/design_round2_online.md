# Round 2 online: observe, explore, select — with maximum exploration first

Date: 2026-08-22. Design, with its costs computed from the phase-98 lattice
(measured data, no new boots). Registered direction from the researcher:
**Round 2 must be cheap, the repetition of rollouts is the resource, and the
opening iterations do maximum exploration.**

## 1. The structural advantage this design stands on

**Losslessness makes exploration free in tokens.** Every arm is
distribution-preserving, so tokens generated under a suboptimal arm are exact
policy samples and fully valid training data. Exploring arm B on a slice of a
rollout costs only the throughput difference between B and the best arm --
never data. The bandit over Round 1's shortlist has zero data cost.

**And the measurement is already paid for.** `num_accepted_tokens` crosses to
the host every step (sections 46-47 located that round trip); accumulating it
into per-(arm, u-bucket) counters is O(1) host work on a value that is
already there, and the per-request u-binning machinery exists from G98-D. No
new sync, no instrument on the measured side (the section-32 trap). The
rollout IS the acceptance campaign: production scale, true content, true
temperature, natural stopping — better data on exactly the two axes the
phase-98 calibration is weakest (T=1, no `ignore_eos` inflation).

## 2. The loop

```text
iteration 1  (COLD START, MAXIMUM EXPLORATION)
  split the rollout uniformly across ALL Round-1 survivors
  harvest per-(arm, u-bucket) acceptance counters from every slice
  -> after one iteration, every arm has a measured curve on real traffic

iteration i  (EXPLOIT + EXPLORE)
  Round 1 re-ranks analytically from the updated counters      [free]
  top-3 schedules run on sub-batch splits; winner takes the bulk
  epsilon-slice keeps exploring: shortlist alternatives + jitter
  of the switch points u* to refine crossover boundaries

every iteration (DRIFT MONITOR)
  observed acceptance vs the values the schedule was selected on;
  re-shortlist when the gap exits the envelope
```

## 3. The cost of exploration, measured

Uniform split over the top-N measured arms, regret vs running the best arm
alone (harmonic mean over the 14 lattice points, from `g98_lat_*`):

| breadth | mean regret | worst point |
| --- | --- | --- |
| N=3 (top-3 confirm) | **2.4%** | 5.5% |
| N=5 | 5.1% | 13.0% |
| N=10 | 8.7% | 20.8% |
| **N=16 (all survivors; skip16 out)** | **15.4%** | 25.2% |
| N=20 (everything) | 23.9% | 32.8% |

**Maximum exploration costs ~15% of ONE iteration.** Over a training run of
hundreds of iterations that is **~0.05% of total rollout compute**, in
exchange for measured acceptance curves for every arm on true traffic. The
skip16 arms are excluded even from maximum exploration: section 45 measured
them never entering a top-5 anywhere, and D2(b) explains why (additivity
inverts at k=16), so their slices would buy nothing.

Steady state: top-3 sub-batch confirmation (2.4%) applies only to the
confirmation slices, and an epsilon of ~1-5% keeps boundaries honest at
0.1-0.3% total regret.

## 4. Statistical sufficiency, measured

Phase 98's acceptance curves rested on **277-22,699 armed steps per (arm,
u-bucket)**, median ~2,500. One rollout iteration at 512 requests x 10K
tokens produces ~1.3M armed steps:

* a **1% slice** yields ~3K steps per bucket — already above the phase-98
  median;
* under maximum exploration (uniform over 16 arms), every arm receives
  ~80K steps per bucket — **~30x the phase-98 median, for every arm, in one
  iteration.**

So the counters converge faster than the policy drifts, which is the
condition the whole design needs.

## 5. Registered parameters

| parameter | value | rationale |
| --- | --- | --- |
| exploration schedule | iteration 1: ALL survivors uniformly; anneal to top-3 + epsilon by ~iteration 3 | researcher directive; one-time 15% amortizes to ~0.05% |
| epsilon (steady state) | 1-5% of requests, on shortlist arms + u* jitter | regret 0.1-0.3%, coverage stays live |
| cold start | maximum exploration IS the cold start | replaces the one-time singles campaign; better data, comparable cost |
| drift trigger | observed-vs-selected acceptance outside the fit envelope | converts gap 6 from assumption to monitored quantity |
| counter granularity | (arm, u-bucket, coarse B) with the registered u-edges | matches the tau(u) structure the model consumes |

## 6. What this design cannot do

* **Discover an arm Round 1 never listed.** Exploration is over the
  survivor set; Round 1's shortlist quality (top-5 contains the true best,
  s48) is still load-bearing.
* **Attribute across arms within one slice.** Each slice observes its own
  arm only; comparisons are across slices of the same iteration, which share
  content distribution and policy version by construction.
* **Escape the one-iteration lag.** Counters describe the previous policy.
  The drift monitor bounds the harm; it does not remove it.

## 7. Open engineering

The counter path (per-arm, per-u-bucket accumulation in the worker, dumped
per iteration) does not exist yet; it assembles existing parts (upstream
per-position counters, G98-D u-binning) and must stay off the hot path per
section 32/47's instrument lessons. Sub-batch arm assignment needs the
scheduler to pin an arm per request group, which the K/OFF ladder does today
for K only.
