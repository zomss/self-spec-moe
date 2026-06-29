# Phase 26: Accept-length-per-GB frontier (single-node baseline)

**Source:** Phase 24 (single-node-PCIe win saturates ~1.47-1.7x, f-capped) + Phase 25
(local-routing / FP4 acceptance). With IB unavailable, we enhance the single-node
mechanism. Two objectives, in tension: **(1) increase accept length**, **(2) minimize
memory**. This phase maps the current Pareto frontier as the baseline to beat.

**Objective:** plot accept length (E[accepted]/cycle) vs per-device resident draft-expert
memory, from already-measured acceptance, and find the efficiency frontier + knee.

**Assumptions:** comm-free draft = local routing over a per-device FP4 expert cache (no
all-to-all). Extra memory = the quantized draft experts only (attention/shared/embeddings
shared with verify; bf16 verify shard is baseline). Accept length = E[accepted] at k=4
(geometric from measured one-step beta; validated vs Phase 24 B1 real k-sweep).

**Commands:** `python frontier.py` (reads Phase 25 data read-only; writes data/frontier.json).

**Decision criteria:** identify (a) the best acc/GB operating point, (b) the knee where
marginal acc/GB collapses, (c) per-lever targets to beat.

**Next artifact:** results_frontier.md, then a lever that pushes the curve up
(tree-drafting for accept length; verify-warmed dynamic cache for memory).
