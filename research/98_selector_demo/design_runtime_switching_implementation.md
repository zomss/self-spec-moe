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
| **quant** | select between resident draft variants | see §2a — a re-bind, not a copy | **no**: the engine boots exactly one draft |
| **skip** | `_apply_skip_layers`, load-time structural | n/a | **no mechanism** |

### What `target-matching` means — weights, not KV

Verified from the boot environment rather than assumed:

| arm | `VLLM_SELF_SPEC_SHARED_KV` | `VLLM_SELF_SPEC_SHARE_WEIGHTS` |
| --- | --- | --- |
| `target-matching` | **1** | **1** |
| `w4a16-quantized` | **1** | 0 |

**Shared KV is on in both arms.** The draft reads and writes the target's KV
cache whether or not it is quantized — the Phase-66 property — which is why the
quantized boot still had to pass shared-KV alias and true-slot proofs in G98-A.

What `target-matching` adds is that the draft's *parameters are the target's
tensors*, aliased. That is exactly why G98-A replaced the alias proof with
`validate_independent_draft_weights` for the quantized arm: the mirror
assertion that **no** draft parameter shares target storage.

Consequence for this plan: switching quant changes **which weights the draft
uses and never touches the KV binding**, so the shared-KV alias and true-slot
proofs hold across a quant switch by construction. That removes a class of
correctness risk from the lever.

`KOffAction` (`koff_runtime.py:702`) carries exactly `action_id, k,
target_graph_id, target_query_width, draft_graph_id, draft_query_width`, and
`ACTIONS_BY_ID` holds only `OFF` and `K4`. So even for window and quant, where
the mutation primitive exists and is measured, **the selector has no way to name
the setting it wants**. Wiring them is the same edit as for skip.

## 2. Every lever can be free at switch time

Two earlier drafts of this section were wrong, and the correction shrinks the
work rather than growing it.

### Legality is uniform

A switch may take effect **only at a target step boundary**, never mid-chain.
P5's registered requirement, identical for every lever. No lever switches more
freely than another, and the first draft's invented "per segment" granularity
does not exist.

### The 2 ms window cost is wiring, not physics

Phase 82 E0 measured, and the mechanism explains the difference:

| lever | mechanism | cost |
| --- | --- | --- |
| OFF ↔ on | scheduler per-step primitive (`disable_by_batch_size`) | **0** |
| K | scheduler per-step `dynamic_sd_lookup[batch]` | **0** — "CG widths for the K set captured at init" |
| window | `drafter._kv_window` mutation | **~2 ms RPC** |

`self._kv_window` is an int on the **worker-side** proposer, initialised from an
env var, so changing it needs an out-of-band control-plane message. The mutation
itself is an attribute assignment; the forward path just reads the int.

K and OFF are free because the **scheduler** owns the decision and it rides the
per-step scheduler output that already flows to the worker. The difference is
which plane carries the decision, not the nature of the value.

**So the fix is S3 itself**: put `draft_kv_window` in `KOffAction` and let the
action ride the existing per-step path. The RPC disappears and window costs what
K costs — zero. The K row even states the pattern: prepare every pool member at
init, select per-step by key.

### The one real cap, and where it lives

The window's persistent buffers (`_win_block_table`, `_win_seq_lens`,
`_win_col_arange`) are allocated **once at `max_model_len`** and used as
left-slices. That is a deliberate Phase-93 IMA fix: *"the buffers are read by
CAPTURED graphs through capture-time pointers, so they must NEVER be
reallocated after the first capture (data change, not shape change)."* They
impose **no cap**.

The cap comes from the FULLCG scratchpad:

```python
n_kept = self._scratchpad_n_kept_blocks()   # n_sink + n_last, derived from _kv_window
cap    = n_kept * self.block_size
if self._sp_col_arange is None or self._sp_col_arange.shape[1] != cap:
    self._sp_col_arange = torch.arange(cap, ...)   # REALLOCATES
```

`_sp_col_arange` is sized from the window and **reallocates when the window
grows** — precisely what the Phase-93 fix forbids for a buffer a captured graph
points at. So:

* **under piecewise** (the W98-R2 runtime) there is no cap at all;
* **under whole-chain** the cap is this one index vector, and the fix is to
  allocate it at the same max as the buffers it indexes.

An earlier draft claimed whole-chain needs a captured graph per window because
"the scratchpad is sized by the window". That is wrong: X13 measured the
scratchpad copy **flat** across windows 128→1024 (0.526–0.528 ms at batch 1),
because `_sp_col_arange` is only an index vector over max-sized data buffers.
Max-allocating it removes the cap without a graph per window.

### Consequence

With S3 wiring window into the per-step action and `_sp_col_arange`
max-allocated, **all four levers are free at switch time**:

| lever | switch cost | paid instead at |
| --- | --- | --- |
| K / OFF | 0 | init (CG widths captured) |
| window | 0 | init (max-sized buffers) |
| skip | 0 (dispatch key) | init (one graph per cut point, S2) |
| quant | 0 (re-bind) | init (dual residency, S3b) |

The costs move to **initialisation memory**, which is one budget against one
ceiling — the 21,682-block shared-KV floor — rather than four different runtime
penalties. That is a much simpler design problem, and it is why §6's RS-1 and
RS-5 are the gates that matter.

A switch-frequency term in the policy is therefore **not required** for the
levers themselves. It is still required for the RL weight refresh (§2a b),
which is the only operation left with a real per-event cost.

### 2a. Quant as a lever is NOT the RL weight refresh

These are two different operations with different triggers, costs and owners,
and an earlier draft collapsed them:

**(a) Quant as a selector lever** — choose between draft variants that are
already resident. With both resident this is a **re-bind**: a dispatch-key
change like skip, not a data copy. The memory story is favourable, because
`target-matching` *aliases* the target's weights and costs almost nothing extra,
while the quantized draft is 6.07 GB.

  *What is missing is not the switch, it is the residency.* The engine boots
  exactly one draft (`speculative_config.model`). Holding two resident is
  unbuilt and is the real work item for this lever.
  (`VLLM_SELF_SPEC_DRAFT_RESIDENT_SETS` is unrelated — it is MoE expert
  residency in `local_route.py`, not draft weight sets.)

**(b) RL weight refresh** — the target's policy updates during rollout, so the
draft must be re-derived to match. This is **mandatory, not a selector choice**,
and it is where Phase 89's **113 ms @ 54 GB/s** pinned-DRAM copy belongs. Phase
92 measured the staleness curve that decides how often it must be paid
(accept 3.931 at drafter@0 against 3.834 fresh at policy step 8).

Attributing 113 ms to "the quant lever" was wrong in both directions: the lever
is cheaper than that, and the refresh is not a lever at all. The plan must carry
both, separately.

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

### S2b — max-allocate the scratchpad index vector

`_sp_col_arange` is sized from `_kv_window` and reallocates when the window
grows, which is the FULLCG init cap (§2). Allocate it at the same maximum as the
window buffers it indexes, so the cap disappears and a captured graph never sees
a moved pointer. Small change; it is what makes window free under whole-chain
rather than only under piecewise.

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

`draft_weight_set` names a **resident** variant. It does not trigger a load.

Selection keys off the existing `dynamic_sd_lookup[batch]` path (P5 D5), which
Phase 82 measured as free per-step, rather than introducing a second policy
mechanism.

### S3b — dual draft residency (the real quant work item)

Per §2a the quant *switch* is a re-bind; what is missing is holding two drafts
resident at once. The engine boots one (`speculative_config.model`).

Required: build both variants at init, sharing the KV binding (which is shared
in both arms — see §1), and keep the target-matching variant as an alias of the
target's weights so only the quantized set costs new memory (6.07 GB, of which
2.49 GB is bf16 `lm_head` + embeddings identical in both).

**Measure, do not assume:** whether the two variants can co-reside alongside the
shared KV cache at the 21,682-block floor. This competes with S1's union for the
same headroom, and G98-A measured only ~2.3%.

### S3c — RL weight refresh (separate from the levers)

Not a selector choice: when the policy updates, the draft must be re-derived.
Phase 89's 113 ms @ 54 GB/s is the cost, Phase 92's staleness curve is the
trigger frequency. It must be sequenced against rollout batch boundaries, not
against the selector's switch policy.

Open: a target-matching draft **aliases** the target's weights rather than
owning a copy, so refreshing it may be free (the alias already points at the
updated tensors) while refreshing a quantized draft requires re-quantizing or
re-copying. That asymmetry is unchecked and could dominate the RL design.

### S4 — switch-frequency accounting in the policy

Per §2 the constraint is frequency, not legality. Only window carries a
per-switch cost a step-rate policy would notice (~2 ms, ~8% if switched every
step). The policy must refuse a switch whose amortised cost exceeds its
predicted gain over the expected dwell. This is where the cost model is consumed
at runtime, and it is why S0 blocks everything.

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

Corrected in §2: whole-chain does **not** need a captured graph per window.
The scratchpad's only window-sized object is the `_sp_col_arange` index vector,
and max-allocating it (S2b) removes the cap, exactly as the Phase-93 fix already
did for the data buffers it indexes.

What remains true is that whole-chain multiplies the **cut-point** graph budget,
since a captured chain is per (batch, K, cut point). That competes with S1's
layer union and S3b's dual residency for the same headroom above the
21,682-block floor.

Which side wins depends on how many distinct cut points the selector actually
uses, which D2 determines and D3 prices. **Do not fix the runtime for the
switching system until D2 reports how many settings survive.**

## 6. Gates

| gate | content |
| --- | --- |
| RS-0 | CPU: union build preserves every alias and KV spec; registry accepts the widened action fields; fail-closed on unknown cut/window/weight-set |
| RS-1 | one boot: capture all cut points, report graph HBM per set **against the 21,682-block shared-KV floor** |
| RS-2 | switch latency per lever, and a token-correctness proof across a target-step-boundary change |
| RS-3 | acceptance and draft cost per setting, matched against the boot-static equivalent |
| RS-4 | policy: switch-frequency accounting honoured, and D3 composite scored against **paid** switch cost |
| RS-5 | dual draft residency co-resides with shared KV at the 21,682-block floor, and a quant switch is proven to leave the KV binding untouched |

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
3. **Whether both drafts can co-reside** with the shared KV cache at the
   21,682-block floor (S3b). This competes with S1's layer union for the same
   ~2.3% headroom, and the two together may not fit.
4. **Whether refreshing a target-matching draft is free** (S3c). It aliases the
   target's weights, so an updated policy may already be visible through the
   alias, while a quantized draft needs re-quantizing or re-copying. If so, the
   RL refresh cost depends on which variant is active — which couples the
   selector's quant choice to the rollout cadence.
5. **Window cap at boot.** Window is switchable only *below* the init cap, so
   the boot cap must be the largest window any regime may select.
