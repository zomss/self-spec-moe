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

## Next (per DIRECTION): the RL-rollout trace is the natural win stage
-- content drift over training is the star axis there, and rollout
batches are large+long-decode (spec-favorable volume).
