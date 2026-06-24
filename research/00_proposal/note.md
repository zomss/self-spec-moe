# Idea Note with AI

## Chat History

### Me

I am considering self-speculative decoding method for MoE. As communication cost is bottleneck at MoE, how about doing draft steps without communication (expert re-routing at the same device) and doing verify step with communication? 

Is this feasible? Also, is it available at NVSwitch server?

### AI

Yes, it's feasible — it's essentially a self-speculative scheme where the draft model is "the same MoE with the gate masked to device-local experts," and standard rejection-sampling verification keeps it lossless regardless of how badly the masked routing perturbs the draft. But the economics are narrower than they first appear, and it's worth being precise about why.

**The cost model (and the catch)**

The catch: unlike layer-skip self-speculation or EAGLE/MTP, your draft is *not compute-cheap*. A local-routing draft step still does the full attention + full top-k expert FLOPs and weight reads — you only remove the all-to-all. So the method is purely a **communication amortization** scheme: you pay dispatch/combine once per verify cycle instead of once per token.

Let f = exposed all-to-all fraction of a decode step, c = draft compute relative to a full step (c ≈ 1 for local top-k), k = draft length, E = expected tokens per cycle (accepted + 1). Then with memory-bound verification (verifying k tokens ≈ one step):

- Speedup ≈ E / (1 + k·c·(1−f))
- Ceiling at perfect acceptance, k→∞: **1/(1−f)**

That ceiling is the whole story. If exposed all-to-all is 20% of your decode step, the method can never exceed 1.25× even with 100% acceptance. The verify-side assumption is reasonably safe for MoE: at BS=1, temporal correlation in expert routing keeps unique experts loaded during verification to only ~1.25–1.42× a single-token decode rather than the naïve (K+1)×, so verification stays near-free.

One lever to escape the ceiling: make the draft cheaper too — e.g., draft with shared expert + top-1 local expert (c ≈ 0.3–0.5). At f = 0.2, k = 3, c = 0.5, break-even drops to E > 2.2 (≈62% per-token acceptance), which is achievable territory. With c = 1 at the same f you'd need ~88% per-token acceptance just to break even, which is brutal for a perturbed router.

**The NVSwitch question**

Mechanically it works anywhere — nothing requires a slow interconnect, and with DeepSeek-style attention-DP + EP, draft steps are *fully* collective-free (if attention is TP'd across the EP group, you still pay allreduces in draft, weakening it further). The real question is what f is:

- Intra-node EP8 over NVSwitch: all-to-all at decode is latency-bound but fast; exposed comm is typically maybe 10–25% of the step → ceiling ~1.1–1.3×. Marginal as a headline result.
- Cross-node EP (DeepSeek-scale): all-to-all can exceed 70% of MoE-layer time and ~40% of end-to-end latency in cross-node settings → ceiling ~1.7×+. This is where the idea lives.
- The honest baseline you must beat there is comm-compute overlap (DeepEP low-latency + two-microbatch overlap), which shrinks *exposed* f. Your strongest regime is low-batch latency-critical decode, where there's no second microbatch to overlap with.

**Acceptance is the real research risk**

Masking the gate perturbs every MoE layer, and the divergence compounds across ~60 layers and across draft positions (the draft attends to its own approximate KV — you need a scratch KV window that verify overwrites, but that's cheap for small k). With random placement and 32/256 experts local, top-k overlap with true routing is poor. Mitigations, in order of leverage:

1. **Shared expert** (always local, replicated) carries a lot of the FFN mass in DeepSeek-style models — it's the anchor that makes this plausible at all.
2. **Locality-aware placement** — this is the direct synergy with your ELDR work: if co-activated experts are placed together, the gate mass covered by local experts (call it γ) rises, and γ is essentially a proxy for acceptance. The drafting scheme converts ELDR-style placement quality into tokens/sec, which is a nicer end-to-end metric than comm volume.
3. **Adaptive draft length for free**: the gate is computed locally, so during drafting you know the *true* top-k and can stop drafting when γ drops below threshold — a zero-cost confidence signal. (One constraint: the verify all-to-all is a collective, so k must be synchronized across the EP group per cycle, or padded.)

**Positioning against prior art**

Nothing I can find does exactly this, but you're adjacent to several lines and should frame against them: Sem-MoE/Speculative-MoE losslessly trims all-to-all volume via expert-placement/token-scheduling co-design and routing-path prediction (comm reduction without spec decoding); C2R co-locates frequently collaborating experts to cut all-to-all (placement-only); MoE-Spec does training-free verification-time expert budgeting, loading only top-scoring experts under a capacity limit (your dual — restricted experts at verify rather than draft, and it's lossy). The incumbent to beat on DeepSeek-class models is MTP-based speculation, which is already comm-free and compute-cheap in drafting; your differentiators are training-free, checkpoint-agnostic, and (plausibly) higher draft fidelity than a one-layer head. Also note layer-skip self-spec (SWIFT-style) on an EP'd MoE already skips those layers' all-to-alls proportionally — a natural hybrid is skip-layers × local-routing.

**Pivotal experiment** — cheap, no system build needed: on a single node, simulate placements by masking gate logits, and measure per-position acceptance vs. γ for (a) random placement, (b) ELDR placement, (c) hot-expert replication, (d) shared+top-1-local drafts. Combine the measured E with a profiled f on your target topology and the formula above tells you go/no-go before you touch vLLM's spec-decode path. My prior: marginal on a single NVSwitch node, genuinely interesting at multi-node EP — and the acceptance-vs-placement curve itself is a contribution either way, because it quantifies how much "expert locality" buys you in model-quality terms, not just bytes moved.

### Me

I think this approach's advantage can batch the draft steps and the verify steps at the same batch while draft step is much slower than batching only draft requests. 

At this case, does it still have advantage of speculative decoding?

### AI

This is exactly the right question to push on, and the answer is mostly negative: **token-level fusion of draft and verify in one step destroys this method's advantage entirely.** The reason falls out of where the gain comes from.

**Why fusion kills it**

The all-to-all is a collective with latency-bound cost at decode scale. If even a few requests in the step are verifying, the step pays the full comm latency M — draft tokens riding along don't add payload, but they don't avoid the latency either. Meanwhile the draft tokens still cost full model compute (c ≈ 1, the defining weakness of this scheme). So in a phase-mixed batch:

- Step time: T_step ≥ T_base (comm every step, plus the fused step processes ~2× the tokens — drafting requests contribute 1 token each, verifying requests k+1 each, and the union of unique experts touched is a superset of baseline).
- A request's cycle: k+1 steps for E expected tokens → TPOT_spec = (k+1)·T_step / E ≥ (k+1)·T_base / E ≥ T_base, since E ≤ k+1.

That last inequality is the punchline: **even at 100% acceptance, fused-mode speculation only breaks even; any rejection makes it strictly worse.** The draft's only saving was removing M from draft steps; fusion reinstates M in every step while the draft compute stays full-price. (This differs from EAGLE/MTP, where the draft is ~2–5% of a step — those survive fusion because they win on compute, not comm.)

So the method has a hard architectural requirement: **draft iterations must be globally collective-free across the entire EP/DP group.** That means lockstep cycles — all requests draft together for k steps, verify together in one step. Three corollaries:

1. **Uniform k per cycle** across the group (the adaptive-k signal from gate-mass coverage can only be applied at group granularity, or via padding).
2. **New arrivals join at cycle boundaries**, adding up to k steps of queueing delay.
3. **No chunked prefill in the decode batch** — a prefill chunk needs true routing, which forces the collective every step and reverts you to the fused case. So this scheme practically *requires* PD disaggregation, with cycle-aligned scheduling on the decode side. (This is the same request-alignment problem you solved in TTServing's scheduler, just with the alignment unit being the verify collective instead of search branches.)

**Your underlying concern is still valid, though**

The reason fusion is tempting is real: lockstep creates utilization ripple — k light steps (B tokens) then one heavy step (B(k+1) tokens), and the heavy verify step may go compute-bound at larger B. The fix that preserves the advantage isn't token-level fusion but **two phase-offset sub-batches**: split the batch into halves, offset their cycles so that sub-batch A's verify step (collective + compute) overlaps with sub-batch B's draft steps (pure local compute, no collective). This is structurally the DeepEP/DBO two-microbatch overlap, but speculation makes it *better* than the plain version: plain DBO at low batch has too little compute per step to hide M behind, whereas here each cycle supplies k steps of collective-free draft compute as overlap material. You get the utilization smoothing you wanted from mixing, without putting a collective in the draft steps.

**The batch-size boundary**

Separately from scheduling, batching erodes the advantage on two fronts even in perfect lockstep. As B grows: (i) the verify all-to-all payload is (k+1)·B tokens, so comm transitions from latency-bound to bandwidth-bound, at which point M(B(k+1)) ≈ (k+1)·M(B) and the amortization vanishes — you pay the same total bytes either way; (ii) compute saturates, and a c = 1 draft is a ~2× compute tax per output token, which is pure throughput loss. So the regime statement is: this method is a **TPOT optimization for latency-bound, low-to-moderate batch decode under tight SLOs** — precisely where comm latency is exposed and FLOPs are idle — and there's a crossover batch size beyond which you should disable speculation and run plain EP decode. That crossover (as a function of B, k, f, and interconnect) would itself be a nice analytical + empirical contribution, and a natural runtime knob: speculate when the decode batch is thin, fall back when it's thick.

### Me 

Okay. I understand. Then, just making k light steps + 1 heavy step approach is most reasonable rather than batching the all requests together.

### AI

Yes — for this method specifically, lockstep cycles are the only structure where the gain survives, so it's the right call. A few design consequences that follow from committing to it:

**Two step shapes, two CUDA graphs.** Draft steps (B tokens, no collective, local-expert grouped GEMM) and verify steps (B·(k+1) tokens, all-to-all) have completely different profiles. The draft step is actually the nicer one — no NCCL in the graph, fixed shape, fully local — so it captures cleanly and runs with minimal launch overhead. That's a small extra win: at decode scale, launch/sync overhead around collectives is part of what you're eliminating, not just wire latency.

**Acceptance heterogeneity is the new scheduling problem.** Uniform k means a request with γ ≈ 0.3 (poor local gate coverage) wastes most of its draft compute every cycle, while a γ ≈ 0.9 request could profitably draft deeper. Since k must be uniform *within* a cycle group, the natural answer is γ-aware request grouping: partition the decode batch into groups by predicted acceptance and run them as separate cycle streams with different k (including k = 0, i.e., plain decode, for hopeless requests). This composes with the phase-offset idea — the groups become the sub-batches whose verify collectives you stagger against each other's draft steps.

**Rank-level stragglers move to the verify boundary.** Draft steps are per-rank local, so imbalance between ranks (uneven request counts under attention-DP, or skewed local-expert load) doesn't cost anything *during* drafting — but it all materializes as wait time at the verify collective, k steps' worth at once. So request-to-rank placement on the decode side needs to balance draft-step time, not just KV occupancy. That's an ELDR-shaped problem: the same locality/affinity signals that drive expert placement now also inform which rank a request should decode on.

**One caveat to keep honest:** lockstep doesn't change the break-even math from before — it just stops fusion from making things worse. You still need E > 1 + k·c·(1−f) per cycle, and the whole-batch verify step pushes the comm toward bandwidth-bound faster than single-request analysis suggests (payload is B·(k+1) tokens). So the crossover batch size where you disable speculation arrives earlier than the per-request model predicts; measuring M(B·(k+1))/M(B) on your target fabric should be part of the first profiling pass, alongside the acceptance-vs-γ curve.

If the acceptance numbers from the masking simulation look viable, the system story writes itself fairly cleanly: cycle-aligned decode scheduler + γ-adaptive grouping + placement co-design, with the analytical crossover as the framing spine.