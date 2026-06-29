# Phase 31: Comm-Free Pruning of the Exact-Verify Tree (does self-spec survive EAGLE?)

Source phases: `27_tree_drafting` (the verify-comm wall: verify is one forward over
`B*nodes` -> comm scales with tree size; chain beats tree at high batch),
`28_dynamic_cache` (pruner-quality caches: last-token / EMA / FP4 local routing),
`24_commbound_throughput` (forced-PCIe engine, measured `f`, `VLLM_SELF_SPEC_SKIP_A2A`),
`25_local_routing_strategy` (shared-expert anchor lifts local-routing fidelity),
`22_fp4_acceptance` / `23_activation_quant_acceptance` (low-precision local pruner cost).

> Phase 30 is being authored concurrently by another agent; Phase 31 does not depend
> on it.

---

## 1. Objective

Decide whether self-MoE-spec's contribution **survives in a world that already has a
trained draft head (EAGLE/MTP)**. The Phase-18+ thesis ("remove the all-to-all from
the draft") is *redundant at the draft layer* once EAGLE exists, because an EAGLE head
is already communication-free and ~10x cheaper on compute (see `00_proposal/note.md:46`).
The only place a comm-free local-routing MoE forward can still earn its keep is the
**exact full-EP verify of a wide draft tree**, whose all-to-all scales with tree size
(Phase 27) and which neither EAGLE nor the current chain design reduces.

Mechanism under test:

```text
draft a WIDE tree T            (cheap; source varies -- see 31a vs 31b below)
prune T -> T' comm-free        (local-routing MoE confidence; NO all-to-all)
exact-verify T' at full EP     (all-to-all over a SMALL node set; rejection sampling)
```

Pruning a tree is **lossless by construction**: proposing/verifying a *smaller* tree
can only lower accept length, never corrupt the output (rejection sampling on the
surviving sub-tree draws from the exact target). So the entire question is
**throughput**, not correctness.

Two variants, deliberately separated by risk:

- **31a (no EAGLE, no extra pass -- the low-risk GO path).** The tree source is the
  project's own comm-free local-routing draft. Its per-node local-routing
  probabilities are *already computed during drafting*, so pruning by them is **free**
  (zero extra forward). This directly revisits Phase 27's verdict "chain beats tree at
  high batch": pre-pruning lets the high-batch verify see only the pruned tree, so we
  may recover the tree's accept-length benefit *without* paying the full tree's verify
  comm. No EAGLE, no training, no extra pass.

- **31b (the EAGLE composition you asked about -- the high-risk gate).** The tree
  source is an EAGLE/MTP head (cheaper, comm-free draft). Pruning now requires a
  **separate comm-free local-routing MoE pass** over the tree (the head's own probs
  are lower fidelity). The make-or-break: does the local-MoE pruner (`P2`) rank
  branches enough better than EAGLE's free head probs (`P1`) to justify a near-full
  extra forward pass? This is a high bar by design.

**Central make-or-break question (kills the direction fast if it fails):**
> Does a comm-free local-routing forward prune the exact-verify tree *materially
> better* than the free signal already available (EAGLE head probs for 31b; or, for
> 31a, whether confidence-pruning recovers tree accept-length at high batch at all)?
> If not, recommend Option 1 (plain EAGLE-on-MoE) and close the composition.

---

## 2. Assumptions / what we rely on (each already measured in a prior phase)

1. EAGLE/MTP head is comm-free (dense, no expert all-to-all). `00_proposal/note.md:46,67`.
2. Tree-verify comm scales with node count `N`; verify is one forward over `B*N`. Phase 27.
3. The forced-PCIe engine reproduces a real comm-bound `f` (0.56 @ B128, 0.65 @ B512). Phase 24 3e.
4. Local-routing branch ranking quality is the `beta` characterized in Phases 9-28
   (shared anchor + FP4 cache); we are reusing that apparatus as a **pruner**, not a drafter.
5. Losslessness: rejection sampling on the pruned sub-tree; greedy bit-exactness modulo
   the batched-vs-sequential MoE float non-associativity already attributed in B1 / Phase 27
   (per-cycle fresh-reference control ~0.99, cascade not bug).

---

## 3. Metrics (defined precisely)

For a proposed tree `T` (`N_tree` nodes) at batch `B`:

- `L_full(T)` = accept length of the full-EP tree-attention verify on `T` (Phase 27
  `tree_verify.py` ground truth; full d2b2 -> 1.95).
- A **pruner** ranks branches by a signal `S` and keeps the top fraction `rho` of nodes
  -> sub-tree `T'(S, rho)` with `N_pruned = rho * N_tree`.
- **Prune recall (fidelity):** `R(S, rho) = L_full(T'(S, rho)) / L_full(T)`. Ideal ~1.
- **Compression:** `rho` (lower = more pruning = less verify comm).
- Ranking signals compared on the *same* tree:
  - `P1` = source/EAGLE-head draft probs (**free** -- no extra pass).
  - `P2` = comm-free local-routing MoE forward probs at the Phase 28 best cache
    (last-token / EMA / FP4). **The candidate.**
  - `P_exact` = full-EP probs (oracle pruner; upper bound on any pruner).

**Headline science number:** at fixed recall `R >= 0.95`, the compression `rho_P2` vs
`rho_P1` (and vs `rho_exact`). `Delta = rho_P1 - rho_P2` is the entire value of local
routing as a pruner. If `Delta ~ 0`, the local-MoE pass buys nothing the head didn't
already give -> NO-GO for 31b.

---

## 4. Economic model (the GO gate -- uses measured engine numbers, not assumed)

Normalize a full-EP single-token decode step = 1 = `(1-f)` compute + `f` comm at batch `B`.

Verify over `N` tree-nodes at the same `B`:
- comm volume linear in nodes: `comm_verify(N) = f * N`.
- compute sublinear in nodes (weights resident, reads amortized; arithmetic adds):
  `compute_verify(N) = (1-f) * (1 + alpha*(N-1))`, with `alpha < 1` **measured** on the
  engine (Phase 16/27 weight-bandwidth-bound MoE).

Per-cycle cost (step-units) and accept length `L`:

| design | cost per cycle | accept length |
| --- | --- | --- |
| EAGLE chain (Phase 27 high-batch optimum) | `eps + compute_verify(k+1) + f*(k+1)` | `L_chain` |
| EAGLE tree, no prune | `eps + compute_verify(N_tree) + f*N_tree` | `L_tree` |
| **31a** local-draft tree + free prune | `(1-f)*compute_local(N_tree)` *(the draft, already paid)* `+ compute_verify(N_pruned) + f*N_pruned` | `R * L_tree` |
| **31b** EAGLE tree + extra local pruner pass | `eps + (1-f)*compute_local(N_tree)` *(extra)* `+ compute_verify(N_pruned) + f*N_pruned` | `R * L_tree` |

Speedup vs autoregressive = `L / cost`. The **pruner's cost is a near-full comm-free
forward** over `N_tree` (`compute_local ~ compute_verify` minus comm) -- this is the main
threat. The prune wins iff comm+compute saved at verify exceeds the pruner pass:

```text
f*(N_tree - N_pruned)  +  (1-f)*alpha*(N_tree - N_pruned)   >   (1-f)*compute_local(N_tree)
\_______________ saved by shrinking the exact verify ______/      \___ extra pruner pass (31b only) ___/
```

Worked sanity (f=0.6, keep half rho=0.5):
- alpha=1 (pessimal, linear compute): saved `0.6*0.5*N = 0.30N` vs pruner `0.4*N` -> **NET NEGATIVE**.
- alpha=0.3 (weight-bandwidth-bound): pruner `~0.4*0.3N = 0.12N`, saved `~0.30N + 0.06N = 0.36N` -> **NET POSITIVE**.

=> the verdict for 31b **hinges on the measured `alpha`** (compute sublinearity) and the
prune ratio at high recall. For **31a the pruner term is zero** (probs already computed),
so 31a wins on a far weaker condition -- it only needs `R * L_tree / cost_31a > L_chain / cost_chain`.

---

## 5. Stages

### Stage A -- Pruner fidelity (the science; no EAGLE, no comm timing) -- CHEAP, run first

1. Build a draft tree from the source drafter. Primary: the project's comm-free
   **local-routing FP4 draft** (on-distribution). Also report **target-top-b** as an
   optimistic upper bound on tree quality.
2. Full-EP tree-attention verify (`27_tree_drafting/tree_verify.py`) -> `L_full(T)`,
   per-node accept probs, oracle ranking `P_exact`.
3. Compute `P1`, `P2` (Phase 28 best cache) over the *same* tree.
4. Sweep `rho` -> recall-vs-compression frontiers for `P1`, `P2`, `P_exact`.
5. Models: **Qwen3-30B-A3B** (no shared expert) and **DeepSeek-V2-Lite** + **Moonlight-16B**
   (shared expert). **Prediction:** `Delta = rho_P1 - rho_P2` is *larger* on shared-expert
   models (Phase 25: local routing is much stronger with the always-local anchor), so the
   pruner most clearly beats the head exactly where the project is strongest.

**GO gate A:** `P2` reaches `R >= 0.95` at `rho_P2 <= 0.6 * rho_P1` (or: at the `rho`
where `P1` keeps `<= 0.85`, `P2` keeps `>= 0.95`). If `P2 ~ P1`, **STOP** -> Option 1.

### Stage B -- Economics (cost side; reuse measured `f`, `alpha`, `eps`) -- only if A is GO

- Measure `alpha` (compute_verify(N) curve) and `eps` on the forced-PCIe engine
  (Phase 24 3e), at B = 128 / 512 / 1024.
- Plug Stage-A `(rho, R)` into Section 4. Report **31a** and **31b** speedups against
  *both* baselines: plain EAGLE chain and EAGLE tree-no-prune.

**GO gate B:** at serving batch (`B >= 128`, `f >= 0.56`),
- **31a** net speedup `>` best EAGLE/self-spec **chain** baseline by `>= 10%`, AND
- **31b** net speedup `>` plain EAGLE chain by `>= 10%` (the harder gate -- must pay for the extra pass).

### Stage C -- End-to-end measured (only if A+B GO)

- Real lockstep on the forced-PCIe engine: draft tree -> comm-free prune
  (`VLLM_SELF_SPEC_SKIP_A2A` for the local pass) -> full-EP verify of survivors.
- Measure wall-clock tokens/s vs plain EAGLE chain on the *same* engine.
- **Losslessness:** per-cycle fresh-reference greedy match (expect ~0.99; cascade caveat
  from B1 / Phase 27, not a bug).
- **EAGLE-head dependency (31b only):** if no public EAGLE/MTP head exists for the model,
  use the FP8/FP4 **quantized-full draft** as a stand-in for EAGLE's tree distribution and
  flag it -- it is conservative on `L_tree` (the quantized draft is a *better* tree source
  than a one-layer head, so it understates the slack EAGLE leaves to prune).

---

## 6. Commands / assets

```bash
# Stage A (fidelity) -- new
.venv/bin/python 31_eagle_tree_prune/prune_frontier.py \
    --model Qwen/Qwen3-30B-A3B --tree d2b3 --cache last_token --quant nvfp4
.venv/bin/python 31_eagle_tree_prune/prune_frontier.py \
    --model deepseek-ai/DeepSeek-V2-Lite --tree d2b3 --cache ema --quant nvfp4  # shared-anchor

# Stage B (economics) -- new; consumes Stage A frontier + measured alpha/eps/f
.venv/bin/python 31_eagle_tree_prune/prune_economics.py --f 0.56 0.65 --batch 128 512 1024

# Stage C (end-to-end) -- new; forced-PCIe engine (Phase 24 3e knobs)
NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1 VLLM_SELF_SPEC_SKIP_A2A=1 \
    .venv/bin/python 31_eagle_tree_prune/prune_e2e.py --model Qwen/Qwen3-30B-A3B --batch 512
```

Reused read-only: `27_tree_drafting/tree_verify.py`, `28_dynamic_cache/{dynamic_cache,skip_cold}.py`,
Phase 24 forced-PCIe engine + `VLLM_SELF_SPEC_SKIP_A2A`.

---

## 7. Decision criteria (summary)

| gate | condition | fail -> |
| --- | --- | --- |
| A (fidelity) | `P2` reaches R>=0.95 at `rho_P2 <= 0.6*rho_P1` | direction dead; recommend Option 1 (plain EAGLE) |
| B-31a (free prune) | 31a > best chain baseline by >=10% at B>=128 | tree-prune offers nothing over chain even when free; Phase 27 verdict stands |
| B-31b (EAGLE compose) | 31b > plain EAGLE chain by >=10% | the extra local pass doesn't pay; use EAGLE alone, self-spec doesn't compose |
| C (end-to-end) | measured tokens/s confirms B; greedy match ~0.99 | model/eng gap; report and scope |

---

## 8. Honest risks (this is the aggressive plan the user flagged)

1. **`P2 ~ P1` (most likely failure for 31b).** If EAGLE's own head ranks branches as
   well as a local-MoE pass, the pruner is dead weight. Stage A is built to find this
   cheaply (no timing, no engine). *Primary kill switch.*
2. **Pruner cost > comm saved** even at good `rho` -- a near-full extra forward is
   expensive; only pays at high `f`, wide trees, and sublinear `alpha`. Stage B catches it.
   31a sidesteps this (free pruner) and is the more likely positive.
3. **Non-multiplicativity.** At low batch / inter-node, EAGLE *already* amortizes
   collective *count* by accept length; the prune adds little there. The orthogonal
   headroom is high-batch / bandwidth-bound (prune cuts comm *volume*), which is exactly
   where spec decoding helps least -- a narrow corner.
4. **Training dependency (31b).** A real EAGLE head needs per-model training, abandoning
   the project's training-free differentiator. Losslessness is unaffected (verify guarantees it).

---

## 9. Expected next artifact

`31_eagle_tree_prune/results_prune.md`: the recall-vs-compression frontiers (`P1`/`P2`/
`P_exact`, per model), the `Delta = rho_P1 - rho_P2` headline, the economic comparison
of 31a/31b vs the EAGLE chain and tree-no-prune baselines, and a GO/NO-GO on **"does
self-MoE-spec survive EAGLE."** If NO-GO, the deliverable is the clean negative:
*EAGLE's comm-free draft subsumes self-spec at the draft layer, and a comm-free pruner
does not beat the head's free ranking on the verify -- recommend plain EAGLE-on-MoE.*
</content>
</invoke>
