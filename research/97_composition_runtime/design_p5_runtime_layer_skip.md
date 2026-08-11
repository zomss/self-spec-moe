# P5 design — runtime layer skip (non-destructive)

Date: 2026-08-11

Status: **design only. Nothing here is implemented, and no GPU command is
authorized.** Sequencing depends on Phase 98's D3 composite gap; see
"When to build this".

## Requirement

The selector must be able to turn draft layer skip **on and off**, and to
**change which layers are skipped**, at runtime — because the two-round search
selects a best layer set per regime, and regime is observed live.

## Why skip is not like the other two levers

Window and draft-weight swap are **data** mutations, which is why they are
cheap and already work:

| lever | mechanism | measured |
| --- | --- | --- |
| window (≤ init cap) | mutate `drafter._kv_window` | ~2 ms RPC, accept restores exact |
| draft weights | in-place `copy_` from pinned DRAM | 113 ms at 54 GB/s, bit-exact graph replay |

Skip is a **structural** mutation. Today `DraftModel._apply_skip_layers`
(`vllm/v1/spec_decode/draft_model.py:319`), called during model construction,
does two things that are baked in permanently:

1. replaces each skipped decoder layer with an index-preserving
   `_SkipDecoderLayer` passthrough; and
2. removes the skipped layers' attention registrations from
   `static_forward_context`, so they contribute **no KV spec** and are never
   discovered as draft attention layers.

Its own docstring states the consequence: *"Load-time static: the layer set
never changes after boot (CUDA-graph safe)."*

### The piecewise shortcut does not exist

Because the draft chain runs PIECEWISE (body on captured pieces, attention
eager), "just don't launch piece j" looks like a free runtime skip. It is not.
The compile splitting op is `vllm::unified_attention_with_output`, so each
piece spans a **layer boundary** — layer j's post-attention MLP together with
layer j+1's pre-attention block. Skipping one layer means skipping parts of two
pieces, which a captured piece cannot express.

This was checked before designing around it, and it rules out the cheapest
approach.

## The design

The lever is switchable **among a pre-validated pool**, not constructible from
nothing. This matches Phase 98, which specifies that a skip identity is
*"chosen offline (Round-2 knapsack), never constructed at runtime"* over a pool
of *"skip counts and 2–3 knapsack identities"*. Materialising an arbitrary new
set mid-run would require a recapture — seconds, not milliseconds — and is
explicitly out of scope.

### D1 — retain the full layer and shared-KV alias union

Stop removing skipped layers at construction. Build the draft with **every**
layer present and every attention registration intact, so the KV spec covers
the union of all layers in the pool. A skipped layer then becomes *unused*
rather than *absent*.

This is the change that makes the set switchable at all, and it is the
registered P5 line *"Retain the full layer/module and shared-KV alias union."*

Consequence to measure, not assume: KV capacity is sized for the union. If the
pool's sets differ widely, the union may cost more shared-KV blocks than any
single set does today. This interacts directly with the 21,682-block launch
floor and must be re-derived, not carried over.

### D2 — skip as a dispatch-time property

Replace the load-time structural edit with a per-forward decision: the draft
forward consults an active skip-set id and dispatches the graph captured for
that set. `_SkipDecoderLayer` is retained as the mechanism *within* a captured
set, not as a permanent model edit.

### D3 — one captured graph set per pool member

Capture the draft decode graphs once per skip identity in the pool, at
initialization. Switching is then a dispatch-key change with no recapture.

Cost driver is graph HBM. This boot captured **0.55 GiB** for the full
target+draft pool; a skip set changes only the draft, so the incremental cost
per additional set should be a fraction of that. That is an inference from one
log line — measuring graph HBM per set is a registered P5 deliverable and a
gate input, not something to estimate away.

### D4 — extend the action registry

`KOffAction` (`koff_runtime.py:681`) currently carries
`action_id, k, target_graph_id, target_query_width, draft_graph_id,
draft_query_width`, and `ACTIONS_BY_ID` holds exactly `OFF` (k=0) and `K4`
(k=4). Add a skip-set identifier to the action record, so an action names both
its K and its skip identity, and the existing closed-registry validation
extends unchanged.

Turning skip **off** is the empty set — the same mechanism, no special case.

### D5 — reuse the existing batch-keyed dispatch

K is already selected per step by batch via `dynamic_sd_lookup[batch]`, at zero
cost (Phase 82 E0: FREE, per-step). Skip-set selection should key off the same
path rather than introduce a second policy mechanism. The selector's per-regime
choice then lands through machinery that is already live and measured.

## Correctness requirements

These are the registered P5 gates, and none may be relaxed:

- **Shared-KV identity.** The union must preserve every alias and true-slot
  proof the phase already enforces. Skipped layers keep their original
  ModuleList indices so each remaining draft layer still name-matches its
  target twin — the property the current passthrough exists to protect.
- **Target-step-boundary switching.** A set change may only take effect at a
  target step boundary, never mid-chain.
- **Token correctness across a switch**, proven, not assumed.
- **Per-set measurement**: graph HBM, switch latency, draft cost, and
  acceptance for every pool member.

## Gates

| gate | content |
| --- | --- |
| P5-0 | CPU: union build preserves all aliases and KV specs; registry accepts skip-set ids; fail-closed on unknown set |
| P5-1 | one boot: capture N sets, report graph HBM per set against the shared-KV floor |
| P5-2 | switch latency and a token-correctness proof across a target-step-boundary change |
| P5-3 | acceptance and draft cost per set, matched against the boot-static equivalent |

P5-1 is the gate most likely to fail, because it is the one with a hard
resource ceiling rather than a software fix.

## Risks

1. **KV union cost.** The union may not fit alongside the shared target KV at
   the registered floor. This is the failure mode that would force the pool to
   shrink, or force skip back to boot-class.
2. **Graph HBM growth** is linear in pool size, so the pool must stay small —
   which is consistent with Phase 98's 2–3 identities but not with an
   open-ended search.
3. **Acceptance drift across a switch.** Window switching restored accept
   exactly; skip changes which layers compute, so there is no reason to expect
   the same, and P5-2 must measure rather than assume it.
4. The union build changes the draft's construction path, which every prior
   phase's boot-static skip measurement (e.g. Phase 89's `{2,4,7,16}`) was
   taken under. Those numbers may not transfer unchanged.

## When to build this

P5's value is sized by Phase 98's **D3 composite gap** — the measured distance
between the live arm and the omniscient per-segment composite. Phase 98 states
plainly that it produces the value case that sizes this work, and that Round 2
answers whether one robust skip identity serves all regimes or whether they
split.

If one identity serves every regime, the runtime lever is worth little and
boot-static skip suffices. If identities split by regime, this design is what
makes the split exploitable.

Building P5 before D3 exists risks engineering an axis the data may not
justify. The recommended order is: Phase 98 Round 2 → D3 → then P5-0.
