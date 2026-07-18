# E2 — switching demo on a real-data regime-shifting trace (2026-07-18; DRAFT)

Trace (Qwen3-8B + W4win draft, 1xH100, continuous phases, real datasets):
P1 interactive b1 x AIME math (8 requests, 256 tok), P2 serving b8 x C4
web docs ~6k-tok prompts (512 tok), P3 burst b32 x AIME (512 tok).
Arms: OFF, static K4, static K6, policy. All engines async-scheduled.

## Final numbers (tok/s; winner bold)

| arm | P1 b1 math | P2 b8 docs | P3 b32 burst | aggregate |
|---|---|---|---|---|
| OFF | 149.9 | **689.9** | **4180.1** | **957.9** |
| k4 static | **173.1** | 516.6 | 3728.8 | 932.8 |
| k6 static | 168.1 | 475.6 | 3539.5 | 886.1 |
| policy | 157.3 (-9%) | 565.5 (-18%) | 4118.1 (-1.5%) | 929.2 (-3%) |

## Gate verdict: FAIL-HONEST (as the README pre-registered)

Policy beats both spec statics (+0/-0.4% vs k4, +4.9% vs k6) but loses
to static-OFF by 3%. The decisive arithmetic: the OMNISCIENT per-regime
switcher scores ~974 = only **+1.7% over static-OFF**, because the one
spec-favorable regime (P1: +15% for spec) carries 9% of trace tokens.
No detector can pay for itself inside a 1.7% ceiling. **The dwell-time
law, measured**: switching pays iff sum(regime_volume x spec_delta)
exceeds detection cost; toggle cost itself is ~0 (E0), so the binding
constraint is DETECTION, and the win condition is a workload where
spec-favorable regimes carry volume. On this trace, static-per-
deployment (OFF) is the right answer -- the phase's honest finding.

## What the demo DID establish

1. **The policy compiles into per-step scheduler primitives** with zero
   switch latency (E0): per-batch-size K schedule (fork-native), the new
   accept-feedback OFF gate (content axis), and the new short-ctx OFF
   rule -- signal = (running batch, mean effective ctx, accept EMA),
   exactly the E1 README detector.
2. **OFF-mode parity while armed**: P3 policy = 4118 vs true-OFF 4180
   (-1.5%) with the draft resident -- switching keeps ~99% of AR
   performance in OFF regimes. This required a real engine fix.
3. **The content axis is real and drifts**: doc-summarization accept
   starts at frac ~.81 (predictable openings) and decays to ~.65 --
   detection latency against a drifting signal with small margins
   (math .87-.91 vs docs-early .81) is the residual cost (P2 -18%).

## Engine work that landed (commits in this phase)

- K=0 draft short-circuit in the runner: proposing 0 tokens previously
  ran the ENTIRE draft chain (measured: b32 "OFF" at 48% of AR). The
  short-circuit must preserve the async bookkeeping (next_token_ids /
  prev_sampled_token_ids) -- skipping it crashes bookkeeping_sync.
- Accept-feedback gate in the scheduler: volume-weighted EMA (a 4-token
  b1 step must not flap the gate), regime-band reset (stale content
  signal from one regime must not gate another OFF irrecoverably),
  probe bursts with a floor weight (async pipelining delivers probe
  verify data 2 steps late; without the floor, b1 probes fold at w=.04
  and the gate locks OFF permanently), dual-threshold hysteresis.
- ASYNC TRAP (affects any draft_model benchmark): vLLM silently
  disables async scheduling for draft_model spec -- the spec engine
  loses 7-15% AR-side throughput vs a nospec engine unless
  async_scheduling=True is forced. Any spec-vs-nospec comparison
  without it inflates OFF's advantage.

## Iteration ledger (fail-honest record)

v1 K-only policy: OFF absent from action set -> loses OFF regimes.
v2 K=0 by schedule: no short-circuit -> "OFF" at 48% AR; b1 EMA flap.
v3 short-circuit (sync): works; async off -> all arms handicapped.
v4 async: crash (bookkeeping invariant) -> in-branch fix; P2 -23%.
v5 higher threshold: P2 fixed, P1 locked OFF (warmup pollution +
   unrecoverable probes) -> regime reset + probe floor.
v6 resets: exposed missing ctx axis (b16-24 K6 fired at 2k ctx).
v7 shortctx rule: P1 flap at single threshold -> hysteresis.
v8-9 hysteresis + async probe-lag fix: mechanism complete.
v10 continuous trace (round-drain artifact removed): final table.

## Root-cause analysis (gate-transition log, VLLM_SELF_SPEC_GATE_DEBUG)

Why the policy loses to OFF (-3%) and even ties/loses static-k4:

Per-phase time vs k4: policy GAINS at P2 (+0.69s) and P3 (+0.41s) and
gives it all back at P1 (-1.19s) -- where policy and k4 run the SAME
config on paper. The log (11 transitions) shows why:

1. **P1: intra-regime content variance flips the gate one request too
   late.** OFF stretches of 97/98/130/124 steps at band=1 -- the gate
   latches OFF ~30 steps into a HARD math problem (per-request accept
   frac ~.75-.8 vs easy ~.9+), then stays OFF ~128 steps (the probe
   interval) into the NEXT, usually easy, request. Per-request dwell
   (~60 steps) is SHORTER than detect (~30, w=.04 at b1) + recover
   (~128) -- a phase-lag classifier on a signal whose dwell is shorter
   than its time constants. Measured blend: ~45% OFF x 149.9 + 55% x
   173.1 = 162.7 (~ the observed 155-157 with probe cost). The
   per-request OFF decisions are individually CORRECT (break-even
   accept at this cell is tau ~4.06; hard problems sit at ~3.1) -- they
   just arrive one request late, converting a correct policy into a
   net loss vs always-ON.
2. **P2 (-18% vs OFF): detection latency against a drifting signal.**
   Docs accept starts at frac ~.81 (predictable summary openings, just
   under the .84 bound) and decays to ~.65 -- the EMA crosses slowly,
   so ~15-25% of the phase runs at the spec rate (517) before OFF.
3. **P3 (-1.5%): the arming rent.** Zero drafts scheduled, yet the
   armed engine pays per-step propose bookkeeping, K6-sized lookahead
   allocation, and gate CPU. This is the true cost of keeping the
   switch available.

So the policy pays three taxes no static pays -- detection lag, probe
cost, arming rent -- against an omniscient dividend of only +1.7% on
this trace. Negative margin is structural here, not a tuning miss.

**Fix directions (next iteration, not run):** (a) per-REQUEST accept
signal (reset/track at request boundaries -- the regime unit at b1 IS
the request), (b) probe cadence scaled by batch (128-step recovery is
2 requests at b1, 2 seconds at b32), (c) map-prior K by prompt features
(task type predicts accept before any token is drafted -- the offline
map re-enters as the prior the online signal corrects).

## Fix round: per-request signal + gate hierarchy (2026-07-18, late)

Applied the root-cause fixes and re-measured (policy arm only; statics
unchanged):

| policy variant | P1 | P2 | P3 | aggregate |
|---|---|---|---|---|
| global EMA (v10, baseline) | 157.3 | 565.5 | 4118.1 | 929.2 |
| per-request EMA alone | 152.6 | 554.6 | 4120.6 | 909.0 (45 flaps: b1 signal is NOISE at half-life 24) |
| + gate min-batch 4 (b1 -> map prior) | 163.6 | 563.2 | 4141.7 | **948.6** (2 transitions, both correct latches) |
| + ctx-cell OFF 8000:8 (map boundary) | 157.8 | 579.0 | 4130.3 | 938.0 |

Diagnostic (gate forced always-OFF = pure ARMING RENT): P1 145.6 /
P2 683.4 / P3 4146.8 -> rent is 0.9-2.9% per phase, NOT the gap.

**The layered causal chain (all measured):**
1. b1: accept signal has no statistical power (4 quantized tok/step;
   per-step noise > hard/easy content gap) -> per-request tracking
   FLAPS (45 transitions). Correct resolution: gate only where volume
   gives SNR (min-batch 4); at b1 follow the map prior. Residual b1
   deficit ~3-6% vs k4-static = the PADDED-VERIFY TAX: a K6-sized
   engine (needed for the b13-24 band) running K4 -- R(K4|Kmax=6) >
   R(K4|Kmax=4), the measured price of K-flexibility.
2. b8/6k-ctx docs: NOT a content cell -- break-even accept tau* =
   3.87/0.75 = 5.16 > 5 = max at K4: spec loses AT ANY ACCEPT. An
   accept gate is structurally the wrong detector; the (b, ctx) MAP
   is the right one (the same b8 cell at 16k wins 1.81x). The ctx-cell
   rule catches it in 0 steps (veto log: n=8, ctx 4839 -> OFF).
3. OPEN ANOMALY (flagged): P2 under either OFF mechanism inside the
   full trace runs 555-580, while the forced-OFF diagnostic ran 683
   and the nospec engine 690. Same suppression path (num_spec=0),
   ~15% unexplained -- hypotheses: warm-state asymmetry after a
   spec-active P1, or prefill-time variance. Needs per-step tracing.

Best policy aggregate 948.6 = -1.0% vs best-static (OFF 957.9), +1.7%
over k4, +7.1% over k6, on a trace whose omniscient switching ceiling
is +1.7%. Transition count 45 -> 2. The demo's residual gap is fully
attributed except item 3.

## Next (per DIRECTION): the RL-rollout trace is the natural win stage
-- content drift over training is the star axis there, and rollout
batches are large+long-decode (spec-favorable volume). Rollout regime
dwells are LONG (a training batch is one content regime), which is
precisely the condition the root-cause analysis says the detector
needs.
