# Phase 61 — the LAST World A cell: DeepSeek-V2 236B at REAL high f (1 rail)

**Question.** Phase 60 showed 1-NIC-per-node does NOT create high f at 30B
(payloads latency-bound; implied f ~0.1-0.26). At 236B (hidden 5120, ~59 MoE
layers) the all-to-all payloads are ~10 MB/collective — the bandwidth zone —
and 8-rail f was already ~0.5 (Phase 53), so ONE rail should finally produce
REAL f ~0.7+. Does World A self-spec (comm-free node-local draft, accept ~1.9)
WIN vs no-spec when the no-spec step is genuinely comm-dominated?

**Arithmetic prediction.** The no-spec step inflates ~2x at 1 rail while the
comm-free draft's cost stays fixed -> World A K=1 ~1.15-1.25x if f ~0.75.
This is World A's first possible real-hardware win, and the final cell of the
regime map either way.

## Setup

- DeepSeek-V2 236B, 2-node TP2 x DP4-per-node = EP16 (h107+h106), the
  Phase 53/54 known-good engine recipe (`run_ladder.sh` step3 at
  gpu_mem 0.95 per `run_step3_095.sh`): max_model_len 1024,
  CG sizes 8,16,32,64,128, max_num_batched 2048. Default synthetic prompts
  (no W7_PROMPT_FILE/W7_CHAT — matches the Phase 53/54 8-rail references).
  b8/32/64, greedy, two-length decode slope (OUTLEN 160 vs SHORTLEN 32),
  2 iters + 1 warmup.
- 1-RAIL env: `scripts/env_1rail_2node.sh` — clone of Phase-52
  `env_2node.sh` (full Phase-47 spec stack) with the single fabric change
  `NCCL_IB_HCA='^mlx5_8'` -> `'=mlx5_0'` (exact-match one HCA;
  NCCL_IB_GID_INDEX=3 and enmlx0 socket/gloo unchanged).
- Arms (`scripts/run_1rail_236b.sh {nospec|worldA}`, Phase-60 runner
  pattern: hard per-try timeout, GPU-pid force-kill both nodes, retry):
  1. **nospec** — the 1-rail baseline. 8-rail refs: 264/756/1130 tok/s.
  2. **worldA** — node-local self-spec draft, K=1
     (`W7_DRAFT_LOCAL_ROUTE=0 W7_DRAFT_NODE_LOCAL=1 W7_DRAFT_FULL_REPLICA=0`;
     a full replica does not fit at 236B; the SP/TP2 node-gather branch
     engages, commit 2d4a930f2). 8-rail refs: 81.0/273.4/378.6 tok/s,
     accept 1.87-1.92.
- Rail-engagement check: first run with NCCLDBG=1 — the `NET/IB : Using`
  lines must list ONLY mlx5_0 on both nodes
  (`data/rail_engagement_evidence_236b.txt`).
- Analysis: `scripts/analyze_1rail_236b.py` (implied f from the no-spec
  step ratio, two-way table).

## Measurements

- Implied f at 1 rail per batch: f ~= 1 - t_8rail/t_1rail from the no-spec
  step ratios (8-rail steps 30.3/42.3/56.7 ms at b8/32/64).
- Two-way table {no-spec, WorldA-K1} x {b8,32,64}: tok/s (accept),
  speedup vs 1-rail no-spec.
- Headline: does World A cross 1.0x at real high f, and how does it compare
  to the ~1.15-1.25x arithmetic prediction?

Results: [results_236b_one_rail.md](results_236b_one_rail.md).
