# W2 — both currencies, measured: the map's currency vs the deployment's

Data: `data/w2/` (dense + llama × {off, 512, 2048}, notune, ITERS=4,
seed 0, episode-rejection protocol from `results_w1.md`). Scorer:
`scripts/score_w2.py`. All numbers below are episode-rejected medians;
rejection counts are in the scorer output and the JSONs.

Definitions (engine-core timestamps, `vllm/v1/metrics/stats.py:444-448`):
`prefill_time = first_token_ts − scheduled_ts`, `decode_time =
last_token_ts − first_token_ts`, per finished request.

- **S_e2e** — wall-clock rate ratio vs the arch's AR anchor (deployment
  currency; what phase 95 scored).
- **S_dec** — per-request decode-rate ratio (TPOT currency; the closest
  live analogue of C2's compile-cell currency).
- **φ** — prefill share of request time. **other** — request time in
  neither phase timer (queue/serialization), derived `e2e_lat − prefill −
  decode`.

## The table

| arch | regime | AR e2e | S_e2e w512 | S_dec w512 | S_e2e w2048 | S_dec w2048 | gap (dec−e2e, w512) |
|---|---|---|---|---|---|---|---|
| dense | R4 | 591.2 | 0.970 | 0.976 | 0.977 | 0.971 | +0.6% |
| dense | R5 | 357.6 | **1.264** | 1.308 | 1.252 | 1.272 | +4.4% |
| dense | R5cot | 579.4 | 1.518 | 1.682 | **1.584** | 1.649 | **+16.4%** |
| dense | R8 | 2025.5 | 0.945 | 1.051 | 0.934 | 1.003 | **+10.7%** |
| dense | R1 | 153.5 | 1.078 | 1.108 | 1.076 | 1.104 | +3.0% |
| dense | R6 | 2747.9 | 1.010 | 1.025 | 1.010 | 1.032 | +1.5% |
| llama | R4 | 601.7 | 1.041 | 1.065 | 1.014 | 1.027 | +2.4% |
| llama | R5 | 142.4 | 0.992 | 1.009 | 1.103† | 1.269† | +1.7% |
| llama | R5cot | 626.4 | 1.504 | 1.637 | 1.525 | 1.597 | **+13.3%** |
| llama | R8 | 2157.5 | 0.967 | 0.999 | 0.963 | 0.996 | +3.2% |
| llama | R1 | 156.9 | **0.723** | 0.745 | **1.085** | 1.116 | +2.2% |
| llama | R6 | 3546.5 | 0.963‡ | 0.989 | 0.833 | 0.809 | +2.7% |

† order-contaminated, see W2b below (fresh-boot value is ~1.37, W1).
‡ 3 of 4 rounds rejected; raw median 0.631 — see "protocol limits".

Post-hoc certified values for llama R5 (fresh boots, W2b/W2c/W2d):
policy-run w512 **0.99** / w2048 **1.367**; unconditional K4 w512
**1.338** (accept 3.463) / w2048 **1.395** (accept 3.765).

## Findings

### F1 — the currency gap has TWO mechanisms, and prefill is the smaller one

Where prefill share is high (R4/R5, φ 0.09–0.27) the gap is +0.6–4.4%.
The LARGE gaps are at long-generation cells where φ≈0.02–0.003:
R5cot +13–16%, dense R8 +10.7%. Decomposed (dense R5cot, per request):
spec compresses decode 42.4 s → 24 s (1.68×) but non-decode request time
GROWS 2.1 s → 3.9 s, diluting e2e to 1.52×. **SD buys TPOT only; every
fixed term — prefill, queue, straggler — dilutes it**, and the dilution
scales with the decode speedup itself. The map's currency error is
therefore largest exactly where the lever is strongest.

The queue audit makes the missing term explicit: at llama R5, **25.6% of
request time sits in neither phase timer** (chunked-prefill serialization
of 14k prompts at b8); dense R5 13.4%; ≤7.4% elsewhere. Any theory-precise
model (C-A) needs wall = prefill + decode + queue with all three measured.

### F2 — llama under notune is not flat, and the compiled policy is now the broken layer

Phase 95 (lottery-contaminated) called llama's envelope zero. Clean llama:
R5cot **+50%**, R1 w512 **−28%**, R6 w2048 **−17%** — a 78-point spread.
The sharpest cell: R1 (b1) w512 0.723 vs w2048 1.085 **at equal accept
(4.61/4.56)**. Physics cannot produce that sign — w2048's scratchpad
gathers MORE columns at ctx≈1087 — but the policy layer can: the two arms
load different compiled tables whose lottery-era R constants drive
opposite arm/park decisions at b1. Same story at R6 (b32, ctx≈318,
neither window binds; spread 0.963 vs 0.833).

**Consequence (hard prerequisite for W4/W5): C2's llama cells must be
re-audited under notune before anything validates against them, and the
compiled policy tables must be regenerated from the audited cells.**

### F5 — the live f EMA is inflated by optimistic pooling (W2b kpick)

W2b's GATE_DEBUG trace (2825 kpick steps): the scheduler's f EMA ranges
0.591–1.000, median **0.647**, while true per-token acceptance is
**0.531** (accept 3.124 at K4). The pooled accept-EMA defaults unseen
requests to 1.0 (`scheduler.py:1324-1326`); with requests finishing and
restarting every round, the default is re-injected continuously — a
standing +0.1–0.12 bias, not just an optimistic INIT. K histogram over
the boot: K=0 52%, K=2 29%, K=4 18% — mixed duty at a cell whose table,
under the TRUE f, commands park.

### F6 — true R vs table R: the lottery-era tables are ~1.5× too costly, and the two errors interact

Unconditional K4 fresh boots (no policy file, `W1_UNCOND=1`) close the
loop by pure identity arithmetic `R = (tau/S − 1)/K`:

| window | uncond S_e2e | accept | **true R** | table R (K4) | inflation |
|---|---|---|---|---|---|
| w512 | 1.338 | 3.463 | **0.397** | 0.5809 | 1.46× |
| w2048 | 1.395 | 3.765 | **0.425** | 0.6187 | 1.46× |

The complete llama-R5 mechanism chain, all measured:

1. **Physics is good on BOTH windows** (uncond 1.34 / 1.40; w2048 also
   buys +0.30 accept at pure K4 — the I2 direction).
2. The tables' R (lottery-era) says park at the true f.
3. The inflated EMA (F5) partially counteracts: w2048 armed enough to
   reach 1.367; w512 stayed mostly parked at 0.99 — **−35% below its own
   physics**.

Two wrong estimates (R too high, f too high) compensating with chaotic
sign IS the current deployed selector. Fixing either alone can make
behavior WORSE (correct f + inflated R ⇒ park everywhere on llama). They
must be fixed together: notune re-audited R (W4/W5) + an unbiased accept
estimator (the pooling default belongs at the current EMA, not 1.0 —
design input for W4/W8).

### F3 — dense's map survives in deployment currency where its margins are real

Dense window ORDERING reproduces: R5 prefers w512 (1.264 vs 1.252), R5cot
prefers w2048 (1.584 vs 1.518, accept +0.21) — the I2 pattern — and
gate-worthy cells (R4, R8 below 1.0) are correctly below 1.0 in both
currencies. The map's failure mode remains levels, not orderings, when
margins exceed noise (the phase-95 scope statement, now in clean data).

### F4 — the W1/W2 R5 gap is NOT a regime-order effect (W2b), and it broke the rejection protocol

W1 (fresh boot, R5 only): 194–196 tok/s. W2 (same config, R4 ran first):
148–157, flat across 4 rounds, accept IDENTICAL (3.124). Two mechanisms
were pre-registered (`run_w2b_order.py`): policy-state pollution (H-duty)
vs KV-block scatter (H-scatter).

**W2b verdict (single boot, R5 → R4 → R5-again, GATE_DEBUG):**

| phase | rates | accepts |
|---|---|---|
| R5 first | 187.4, 194.6, 194.7 | 3.124 × 3 |
| R4 mid | 611.6, 610.7, 615.1 | 2.28–2.38 |
| R5 again | **194.8, 194.6, 194.7** | 3.124 × 3 |

P-W2b1 CONFIRMED (fresh boot reproduces W1). **P-W2b4: no order effect —
R5-again is identical to R5-first to the decimal.** H-duty AND H-scatter
both refuted.

Reattribution (suspected, not proven — T11): W2's llama lane ran on GPU1;
W1's spec lane and W2b ran on GPU0. W2's R4 rounds BEFORE the slow R5
were fast (611 — matching W2b exactly), so the suppression began
mid-boot and covered R5's whole ~3-min window — within observed source-B
episode lengths, at −22% on the spec path vs −11% on AR (the scratchpad
gather is more bandwidth-sensitive than an AR step). This is consistent
with source B hitting a whole measurement cell.

**Protocol hole exposed: within-cell max-anchored rejection cannot catch
an episode that covers every round of a cell** (W2's R5 rounds 147.6–157.0
looked mutually consistent; ref=max flagged nothing). Certification
against an external reference is required — here W2b (194.7, S=1.367)
supersedes the contaminated w2048 cell, and a fresh-boot w512 R5 run
(`w2c_llama_w512_r5_fresh.json`) certifies or corrects the w512 cell.

### Protocol limits found (feed into W3)

- Max-anchored episode rejection is too aggressive at b1 (llama R1 w512
  kept 1/4 rounds — below the ≥2 rule; the b1 sequential-prompt protocol
  has legitimate round-to-round spread). W3 must define a b1-specific
  rule (median-cluster reference) before any scored comparison.
- Rejection is also too WEAK against long episodes: a cell whose entire
  round set sits inside one episode self-certifies (F4). W3's protocol
  needs an external reference per cell — a replicate boot, or a
  cross-boot anchor — not just within-cell consistency.
- Sequential regimes share scheduler state (EMA, probe phase, policy
  cell): W2b showed the STEADY-STATE is order-independent for R4→R5, but
  the EMA/probe transients around regime boundaries remain unmeasured;
  map-validation runs still prefer per-regime fresh boots.
- llama w2048 R5 in the table above (†) is superseded by W2b's fresh
  measurement: S_e2e = 1.367 (194.7 / 142.4).

## Currency decision (feeds W3's pre-registration)

The arc's declared currency, per C-A and these measurements:

- **Model validation (W4/W5)**: TPOT/decode currency (S_dec) — it is what
  `S = (1+fK)/(KR+1)` predicts, now measurable live per request.
- **Deployment claims (W6+)**: S_e2e, always reported next to S_dec with
  the bridge terms (φ, queue, straggler) measured, never inferred.
