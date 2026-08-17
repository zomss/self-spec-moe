# Preregistration — the fail-closed arming rule (G98-F)

Issued 2026-08-17, after D3 scored and before the rule was evaluated against
any measured grid. Amends nothing: D1', D2 and D3 are scored and closed, and
their artifacts are untouched.

## 1. The defect this addresses

D3 scored the two-round selector at **95.1% of omniscient** (corrected
instrument) with a single dominant miss: at **R4** (batch 8, 8k context,
summarization) every lever family loses to no speculation, and the selector
armed anyway — **522.3 against OFF's 684.9 decode tok/s, a 24% regression on
that regime**. Its other five picks cost under 1% combined
(`results_g98_e.md`).

The selector ranked the armed candidates correctly at R4. What it lacked was
a rule for *declining to arm*, and its own prediction already carried the
warning: R4's predicted margin over OFF was **1.026**, the thinnest in the
map, in the regime whose cost fit was the loosest.

## 2. The rule

For each regime, with `predicted` the D3 prediction map and `envelope` the
D1' combined envelope:

```text
margin(R)    = predicted[best armed cell][R] / predicted[off][R]
threshold(R) = max(EPSILON_ARM, envelope(R))
arm(R)  iff  margin(R) - 1 > threshold(R)      # else park (OFF)
```

**No free parameter is introduced.** Both inputs are registered quantities
that predate this rule:

| input | value | provenance |
| --- | --- | --- |
| `EPSILON_ARM` | 0.015 | Round-1 preregistration arming rent |
| `envelope(R)` | 0.017–0.136 | `d1p_fits.json`, G98-C (scored, closed) |

A regime with no fitted envelope raises rather than arms: an unresolved
regime is a reason to decline, not to guess.

**The comparison is deliberately conservative.** `envelope` bounds the
relative error of the draft-chain cost `D`, while `margin` is a throughput
ratio; an armed step costs `D + verify`, so a relative error `e` in `D` moves
throughput by `e·D/(D+verify) < e`. Using the full envelope overstates the
uncertainty and therefore declines slightly more often than a tight
propagation would — the right direction for a rule whose purpose is to
decline when unsure, and it keeps the threshold traceable to one registered
number.

## 3. What the rule predicts, committed before scoring

Computed from the committed prediction map and the closed D1' fits alone —
no measured D3 grid data — and sealed in `data/g98_f/commitment.json`
(digest `bea15a46d26c8b84569ebd093b03d47975f6a29aa16dd7edfa739ab3728740c8`):

| regime | margin | threshold | decision |
| --- | --- | --- | --- |
| R1 | 1.335 | 0.0254 | arm |
| **R4** | **1.026** | **0.0724** | **park** |
| R5 | 1.632 | 0.1362 | arm |
| R5cot | 1.805 | 0.0956 | arm |
| R6 | 1.228 | 0.0323 | arm |
| R8 | 1.180 | 0.0174 | arm |

**Firing set: {R4}.** Every armed regime clears its threshold by more than
3x, so the non-firing is not a knife edge (asserted in
`test_w98_failclosed.py`).

## 4. Claims

**C-FC1 (in-sample, arithmetic).** Applying the rule to the scored D3 grid
raises the selector's share of omniscient and removes the R4 regression,
under both instrument arms.

**C-FC2 (out-of-sample, G98-F).** The rule's two empirical premises
reproduce on fresh boots on different hardware:

* **Premise 1** — at R4 the armed pick measures below OFF.
* **Premise 2** — at R1 the armed pick measures **above** OFF by >5%.

Premise 2 is the load-bearing one. A rule that parks is trivially safe; what
makes this a fix rather than a retreat is that it declines *only* where the
margin is invisible. Both premises are measured in one interleaved session
with arm order reversed between rounds, and both regimes share each boot.

## 5. Stated limitations

* **C-FC1 is in-sample by construction.** The rule was written after
  observing R4's failure in the D3 grid. It is reported as the consequence
  of a pre-committed rule, never as independent confirmation.
* **The validation box is the KVM guest** whose host-side clamp gated the
  original campaign. The bias is directional and known: a clamp adds fixed
  per-step host cost, and armed steps carry more host work than parked ones,
  so a clamped boot penalises the armed arm. That inflates premise 1 and
  works against premise 2 — which is why premise 2 carries the verdict.
  Startup signatures are recorded per boot.
* **The rule is evaluated at the selector's pick level**, which is where D3
  scores it. Wiring it into the live serving ladder is a separate change and
  is not claimed here.
* **One threshold shape.** `max(rent, envelope)` is not shown to be optimal,
  only sound and parameter-free. A tighter propagation would decline less
  often; that is a refinement, not a correction.
