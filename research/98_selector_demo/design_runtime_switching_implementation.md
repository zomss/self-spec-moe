# Runtime switching implementation plan — all three levers

Date: 2026-08-13
Status: implementation plan. Nothing here is built. Sequenced behind the
W98-R2 cost campaign and D2; see §7.

Supersedes nothing. Specialises
`research/97_composition_runtime/design_p5_runtime_layer_skip.md` to the
selector model stated by the user: **the layer set is fixed by offline search;
what varies at runtime is how many of those layers are skipped.**

## 1. Corrected status — what is live today

An earlier reading of the env vars suggested all three levers were boot-time.
That is wrong, and the correction matters because it changes the size of the
work:

| lever | mechanism | measured cost | live today? |
| --- | --- | --- | --- |
| **K / OFF** | `KOffAction` registry, per step | free | **yes** — driven by the selector |
| **window** (≤ init cap) | mutate `drafter._kv_window` | **~2 ms RPC**, restore exact (P82 E0) | mechanism yes, **not selector-driven** |
| **quant / draft weights** | in-place `copy_` from pinned DRAM | **113 ms @ 54 GB/s** for an 8B draft (P89) | mechanism yes, **not selector-driven** |
| **skip** | `_apply_skip_layers`, load-time structural | n/a | **no mechanism** |

`KOffAction` (`koff_runtime.py:702`) carries exactly `action_id, k,
target_graph_id, target_query_width, draft_graph_id, draft_query_width`, and
`ACTIONS_BY_ID` holds only `OFF` and `K4`. So even for window and quant, where
the mutation primitive exists and is measured, **the selector has no way to name
the setting it wants**. Wiring them is the same edit as for skip.

## 2. Switching cost sets switching cadence

The three levers differ by ~5 orders of magnitude in switch cost, against a
~25 ms step. The selector cannot treat them as one decision:

| lever | switch cost | usable cadence |
| --- | --- | --- |
| K / OFF | free | **per step** (already so, via `dynamic_sd_lookup[batch]`) |
| skip (after this work) | dispatch-key change | **per step**, target-step boundary only |
| window | ~2 ms | **per segment** — 8% of a step; not per-step |
| quant weights | 113 ms | **per regime shift at most** — 4–5 steps of stall |

**This is a design constraint, not a detail.** A selector that switches
quantization per step would spend more time swapping than serving. The policy
must carry a per-lever cadence and a hysteresis rule, and D3's composite must be
scored against the switch cost actually paid, not an idealised free switch.

Registered consequence: quant is effectively a **segment-level** lever. If the
regime mix changes faster than ~113 ms, quant should be pinned boot-static and
only window/skip/K vary. Whether the declared workload mix is that fast is a
D3 input.

## 3. The skip ladder — the user's model

Search fixes one **priority order**; the runtime variable is the **cut point**.
The sets already committed for W98-R2 are exactly that ladder, and are nested by
construction:

```text
skip4  = {2,4,7,16}
skip8  = skip4  + {11,20,25,30}
skip16 = skip8  + {1,6,9,13,18,23,27,33}
```

This is a strict simplification of P5, which assumed a pool of 2–3 arbitrary
knapsack identities. A nested ladder means:

* switching is monotone along one ordering, so `keep_frac` stays a clean scalar
  (the same property that keeps the cost model identifiable);
* the captured-graph pool is **|cut points|**, not |identities| × |counts|;
* "skip off" is the empty prefix — no special case.

**What must be tested, not assumed:** that one ordering serves every regime.
D2 registers exactly this question. If orderings split by regime, the pool grows
back toward P5's general case and §4's graph budget must be re-derived.

## 4. Implementation

### S1 — retain the full layer and alias union (P5 D1)

Stop removing layers at construction. Build the draft with **every** layer
present and every attention registration intact, so a skipped layer is *unused*
rather than *absent*, and the KV spec covers the union.

Correctness carried over unchanged: skipped layers keep their original
ModuleList indices so each remaining draft layer still name-matches its target
twin — the property the current passthrough exists to protect.

**Measure, do not assume:** KV capacity is then sized for the union. G98-A
measured the quantized draft at 22,190 blocks against the 21,682-block floor —
**~2.3% headroom**. The union must be re-derived against that floor before
anything else is built. This is the step most likely to fail on a hard resource
ceiling rather than a software fix.

### S2 — one captured graph per cut point (P5 D3)

Capture draft decode graphs once per cut point at init; switching is a
dispatch-key change with no recapture. Graph HBM per set is a **deliverable, not
an estimate** — the one available data point is 0.55 GiB for the full
target+draft pool at boot, and a skip set changes only the draft.

### S3 — extend the action registry for all three levers (P5 D4, widened)

`KOffAction` gains:

```text
draft_skip_cut     int          index into the fixed ladder (0 = off)
draft_kv_window    int          0 = off; must be <= the boot cap
draft_weight_set   str | None   None = target-matching
```

`ACTIONS_BY_ID` stays a **closed registry** — the existing fail-closed
validation extends unchanged, and an action naming an unregistered cut point,
window, or weight set is refused at boot rather than at step time.

Selection keys off the existing `dynamic_sd_lookup[batch]` path (P5 D5), which
Phase 82 measured as free per-step, rather than introducing a second policy
mechanism.

### S4 — cadence and hysteresis in the policy

Per §2 each lever carries a minimum dwell time. The policy must refuse a switch
whose amortised cost exceeds its predicted gain over the expected dwell. This is
where the cost model is consumed at runtime, and it is why S0 blocks everything.

### S5 — switch-correctness proofs

Unchanged from P5 and none may be relaxed:

* a set change takes effect **only at a target step boundary**, never mid-chain;
* **token correctness across a switch is proven, not assumed**;
* per-setting measurement of graph HBM, switch latency, draft cost and
  acceptance.

## 5. The runtime choice constrains switching cost

A tension surfaced by X6–X14 that belongs in this plan:

* Under **piecewise** (the W98-R2 scored runtime), window is a *data* mutation —
  paged attention just reads `min(context, window)` — hence ~2 ms.
* Under **whole-chain** (`FULLCG=1`), the scratchpad is **sized by the window**,
  so changing window changes captured shapes and would require a graph per
  window, multiplying the S2 budget.

So whole-chain buys steady-state speed (1.21x at batch 1, ~1% at batch ≥ 32) and
pays for it in switching flexibility. Which side wins depends on switch
frequency, which is a D3 input. **Do not fix the runtime for the switching
system until D3 measures how often it actually switches.**

## 6. Gates

| gate | content |
| --- | --- |
| RS-0 | CPU: union build preserves every alias and KV spec; registry accepts the widened action fields; fail-closed on unknown cut/window/weight-set |
| RS-1 | one boot: capture all cut points, report graph HBM per set **against the 21,682-block shared-KV floor** |
| RS-2 | switch latency per lever, and a token-correctness proof across a target-step-boundary change |
| RS-3 | acceptance and draft cost per setting, matched against the boot-static equivalent |
| RS-4 | policy: cadence/hysteresis honoured, and D3 composite scored against **paid** switch cost |

RS-1 is the gate most likely to fail, for the reason given in S1.

## 7. Sequencing, and what blocks what

```text
S0  W98-R2 cost campaign  ->  identified cost model      [BLOCKING]
D2  knapsack + controls   ->  candidates per regime,
                              and whether ONE ordering serves all regimes
D3  end-to-end            ->  sizes the runtime lever
RS  this plan             ->  built only for the levers D3 shows are worth it
```

**S0 is blocking and not optional.** The policy in S4 consumes cost predictions
to decide whether a switch pays. The current model under-predicts quantized
compositions by 4–23%, one-directional, and 2 of 6 regimes are NOT RESOLVABLE.
A selector built on it would switch toward configurations it systematically
believes are cheaper than they are.

P5's own recommendation stands and is inherited: build after D3, because if one
setting serves every regime the runtime lever is worth little and boot-static
suffices.

## 8. Open items to measure, not assume

1. **Union KV cost** against the 21,682-block floor (S1) — the hard ceiling.
2. **Graph HBM per cut point** (S2) — one log line is not a measurement.
3. **Whether target-matching ↔ quantized is a copy or a re-bind.** Phase 89's
   113 ms is a DRAM→GPU copy between two *owned* weight sets. A target-matching
   draft **aliases** the target's weights rather than owning a copy, so that
   transition may be a re-binding with a different cost profile. Not yet checked.
4. **Whether only the quantized body needs swapping.** `lm_head` and embeddings
   are bf16 and hold identical values in both checkpoints (2.49 GB of the
   6.07 GB); if they can be left resident, the swap moves ~2.79 GB rather than
   the full set, and 113 ms should fall proportionally.
5. **Window cap at boot.** Window is switchable only *below* the init cap, so
   the boot cap must be the largest window any regime may select.
