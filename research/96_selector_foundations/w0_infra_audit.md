# W0 — stack capability audit: what binds where, what can switch, what cannot

Source: user directive 2026-08-05 ("start with infrastructure — make sure our
stack supports all levers, switching, and others consistently and efficiently").
This is a CODE audit, not an experiment: every claim cites the binding site.
Its purpose is to make the runtime action space precise BEFORE the selector is
redesigned on top of it — phase 95's failures (`wnone` boot refusal, the
unpriced ~20 ms transition, K-only switching) were all infrastructure gaps that
leaked into science.

## 1. Lever binding matrix

| lever | binding site | binds at | runtime-switchable today | a switch would invalidate |
|---|---|---|---|---|
| **draft depth K** | boot `num_speculative_tokens` = KMAX; per-step K from policy (`scheduler.py:1260`); verify widths `{1} ∪ {1+K}` pre-captured per policy K (`gpu_model_runner.py:826-848`) | boot (max) + per step (≤ max) | **YES, down only.** Chain replays the per-step graph K times; verify dispatches a pre-captured width. "Full-CG keys are captured per width so K-switching carries no piecewise tax." | nothing (by design) |
| **spec OFF** | per-step K=0, width-1 verify graph (`scheduler.py:1238`) | per step | **YES** — but the transition costs ~20 ms/cycle, mechanism unidentified (I5) | nothing structural |
| **KV window** | proposer init `self._kv_window` (`llm_base_proposer.py:302`); scratchpad capacity `cap = ceil(sinks/B) + ceil((window+KMAX)/B)` blocks (`llm_base_proposer.py:2872-2883`) | boot | **NO.** But see §3: the captured graph reads live buffers through capture-time pointers and masks by live `seq_lens` (`llm_base_proposer.py:2885-2907`, `scratchpad_attn.py:36,148-154`) — DOWN-switch is mechanically reachable | scratchpad shape (gather width, mask arange, chain metadata) → graph re-capture for any window ABOVE boot's |
| **layer-skip set** | load-time model surgery: layers replaced by passthroughs, attn registrations deleted from `static_forward_context` (`draft_model.py:319-356`, "Load-time static: the layer set never changes after boot (CUDA-graph safe)") | boot | **NO** | the model itself — skipped layers' modules are gone; the graph's layer sequence; the draft-attn layer discovery |
| **draft quantization** | boot checkpoint (`--speculative-config` draft model); `SHARE_WEIGHTS` only when draft quant == target's (phase 94) | boot | **NO** | DRAM weight residency (a second quant = a second full weight copy) |
| **kernel realization** (FULLCG / PIECEWISE / EAGER / WHOLECHAIN) | capture at engine init; `DRAFT_FULLCG` hard-requires `window > 0` (`llm_base_proposer.py:357-365`) | boot | **NO** | the entire captured-graph set |
| **MoE resident experts** | lazy module-global from a `.pt` path (`local_route.py:41-74`); requires `DRAFT_FULL_REPLICA` (an EP shard cannot compute non-local experts) | boot (first touch) | **NO** (no reload mechanism) | the frequency maps per (layer, device) |

Two structural constraints that break lever-grid uniformity:

- **`{window=none} × {FULLCG} = ∅`** (`llm_base_proposer.py:360-365`). Any
  "windowless" arm runs a DIFFERENT kernel stack (paged FA), so no-window vs
  window comparisons are cross-stack. Phase 95 hit this as the `wnone` boot
  refusal and correctly dropped the arm.
- **Weight sharing is quant-conditional.** Only a quant-matching draft shares
  DRAM with the target; every other quant is a resident second copy. So quant
  is not merely boot-fixed — it is *memory-budget*-fixed.

## 2. The runtime action space today

The per-step decision (`scheduler.py:1238-1260`) is an argmax over
`{0} ∪ {options' K}` using `(K, R)` pairs from the policy cell — **one
scalar**. The policy schema (`cells[].options[]`) carries `K`, `R`, `S_ref`,
`f` and no composition field. Constraint **C-B** — lever =
`(composition, draft tokens)` as one action — is therefore not representable
in the deployed runtime, only in C2's offline search.

## 3. The reachable action space (the audit's main finding)

The stack already embodies one clean pattern — **boot at max, select down**:
K boots at KMAX and every step picks K ≤ KMAX from pre-captured widths, paying
no re-capture. The audit shows which levers can join that pattern and at what
price:

| lever | boot-at-max, select-down | mechanism | cost semantics |
|---|---|---|---|
| K | **have it** | pre-captured verify widths + chain step count | true cost scaling (fewer steps = less work) |
| OFF | **have it** | width-1 graph | transition cost unpriced |
| window | **reachable, two grades** | (a) *masked down-switch*: boot at pool-max window; per-step rewrite of `_win_seq_lens`/kept pages selects any smaller window inside the SAME graph (the mask already excludes unfilled slots) — (b) *per-window graphs*: one scratchpad capture per pool window | (a) acceptance lever ONLY — gather cost stays at max-window cap (the known window cost floor, 93/94). (b) true cost scaling; needs multi-capture residency |
| skip set | **reachable, expensive** | retain skipped layers' modules, gate passthrough at forward time, capture one graph per pool set | per-set graphs; DRAM already paid (layers retained); breaks the "CUDA-graph safe by surgery" simplification |
| quant | **no** | second DRAM copy per quant | boot-class, permanently |
| kernel realization | **no** | distinct capture stacks | boot-class, permanently |

This yields a precise two-tier statement that sharpens C-B and C-C:

> **Boot-class levers** (chosen once per deployment by Round 1/2): quant,
> kernel realization, max-window, max-K, skip-pool.
> **Runtime-class levers** (the online action): `(window ≤ W_boot,
> skip-set ∈ captured pool, K ≤ KMAX, OFF)`.

The selector's online action space is exactly the runtime class; the offline
search's job is to choose the boot class AND the runtime pool. S2's capture
budget is the size of that pool.

## 4. Consistency gaps (the instrument must not lie)

| # | gap | evidence | fix |
|---|---|---|---|
| G5 | flashinfer autotune selects kernels by boot-time timing → replicate boots land on different kernels; llama −25..−41.6% (I4) | phase 95 boot matrix; `R88_NO_AUTOTUNE` escape hatch exists | **W1** (blocker) |
| G6 | `load_regime` accepts `seed` and never uses it — "seeds" are replicate boots, prompt sets byte-identical | verified by hashing prompt sets (phase 95) | trivial harness fix; bundle with W1 |
| G7 | regime runner records ONE currency (end-to-end); the map is decode-only — 30–47% prefill share hides in the gap (I1) | phase 95 I1 table | **W2** (TTFT/TPOT split) |

## 5. Efficiency gaps (C-C is not met)

| # | gap | evidence | fix |
|---|---|---|---|
| G4 | arm/disarm transition ~20 ms/cycle vs ~8 ms decode step — 2.5 forwards, NOT hidden; mechanism unidentified (async-drain vs graph-class dispatch vs draft-state rebuild) | 4.6% duty cost 4.0% vs 0.4% predicted (I5) | **W4**: per-phase instrumentation; probe discriminator already written (95/`run_e1p_probe.sh`) |
| G9 | parked engine costs 2.5% at b16, scheduler hardcodes OFF=1.0 | phase 95 (I5) | **W4**: per-cell parked cost into the model |
| G1 | action space is K-only; C-B's `(composition, K)` unrepresentable | §2 | **W6/W7**: policy schema gains a composition id per option; scheduler argmax runs over (comp, K) pairs; proposer dispatches the comp's graph |
| G2 | window not runtime-switchable | §1, §3 | **W7**: grade (a) masked down-switch first (small, same graph, acceptance-only — enough wherever the map's window preference is acceptance-driven, e.g. R5cot); grade (b) per-window capture only if the pool needs cost-true switching |
| G3 | skip set not runtime-switchable | §1 | **W7, conditional**: build gated-passthrough + per-set capture ONLY if Round 1 shortlists >1 skip set per deployment; otherwise skip stays boot-class and the pool is (window, K) |

## 6. Order of work

1. **W1 + G6** — determinism and seed plumbing. Cheap, blocks all model
   validation.
2. **W2 (G7)** — dual-currency recording. Cheap, blocks all re-scoring.
3. **W4 (G4, G9)** — price the transition and the parked engine. The probe
   discriminator distinguishes per-flip from per-armed-step cost in one run
   pair; a phase-level profile (existing `SELF_SPEC_PROFILE` hooks) attributes
   the ~20 ms.
4. **G1 schema + masked window down-switch (G2a)** — the smallest change that
   makes the runtime action `(composition, K)`: composition = window choice
   under one graph. No new captures, no new memory. This alone lets Round 2's
   per-regime pick be EXECUTED by the runtime.
5. **G2b/G3 multi-capture residency** — only after W3's gate says per-regime
   switching clears the threshold, and sized by S2's capture budget.

Steps 1–3 are measurements the paper needs regardless of the design outcome;
steps 4–5 are the build and stay behind W3's pre-registered gate.

## 7. What this audit changes in the phase-96 README

- Work order gains **W0 (this audit, done)**; W7's scope is split into
  G2a (cheap, unconditional) and G2b/G3 (gated, budget-sized).
- The design's S2 (capture budget) now has its concrete currency: **number of
  resident captured graphs = |pool windows| × |verify widths| (+ |skip sets|
  if G3 is built)**, not "configurations".
- C-B's action is formally: boot-class chosen offline, runtime action =
  `(window ≤ W_boot, skip ∈ pool, K ≤ KMAX, OFF)`.
