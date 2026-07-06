# Phase 60 — the HIGH-f regime, made real by rail restriction

**Question.** Our 2-node fabric is rail-optimized best-case: 8x400G NICs per
node. Restricting NCCL to ONE rail per node (`NCCL_IB_HCA='=mlx5_0'`) makes it
a REAL 1-NIC-per-node cluster — the common production topology — dropping
per-GPU inter-node bandwidth ~8x and pushing the comm fraction f from
~0.25-0.35 (Qwen3-30B, 8 rails) to an expected ~0.7+. At real high f, which of
{no-spec, EAGLE3, World A self-spec} wins at serving batch?

**Prediction under test** (Phase 50's EMULATED result, never measured on real
fabric): at high f, accept-per-verified-token IS comm efficiency — World A
self-spec (accept ~2.9, 3 verify tokens/cycle -> 3/2.9 ~ 1.03 comm-neutral)
should beat EAGLE (accept ~1.7 at K=1: 2/1.72 = 1.16x comm inflation), and
possibly beat no-spec. At 8 rails EAGLE b64 was 0.95x and World A ~0.3x of
no-spec (draft-compute-bound because comm was cheap); at 1 rail the balance
should flip toward World A.

## Setup

- Qwen3-30B-A3B, 2-node DP16/EP16 (h107+h106), short context (max_model_len
  2048), chat-templated on-distribution prompts
  (`research/57_large_ep_spec_strategy/data/prompts_ondist.txt`, `W7_CHAT=1`),
  b8/32/64, greedy, two-length decode slope (OUTLEN 160 vs SHORTLEN 32).
- 1-RAIL envs: `scripts/env_1rail_eagle.sh` (no-spec + EAGLE arms; clone of
  Phase-57 `env_eagle_2node.sh`) and `scripts/env_1rail_worldA.sh` (World A
  arm; clone of Phase-52 `env_2node.sh` full spec stack). Single change to the
  fabric: `NCCL_IB_HCA='^mlx5_8'` -> `'=mlx5_0'` (exact-match one HCA).
- Arms: no-spec baseline; EAGLE3 K=1/K=2 (`Tengyunw/qwen3_30b_moe_eagle3`,
  async sched off, retry-on-wedge runner); World A self-spec K=2
  (`W7_SPEC_METHOD=draft_model`, FP8 full-replica comm-free draft:
  `W7_DRAFT_LOCAL_ROUTE=1 W7_DRAFT_FULL_REPLICA=1`, Phase-47 stack:
  DRAFT_FULL_CG/COMPILE_CONSISTENT/DRAFT_CHAIN_PIECEWISE, AMORTIZE_DP_COORD=0).
- Runner: `scripts/run_1rail.sh {nospec|eagleK1|eagleK2|worldA}` — Phase-57
  ksweep pattern (hard per-try timeout, GPU-pid force-kill both nodes, retry).
- Rail-engagement check: `scripts/run_smoke_nccl.sh` (NCCL_DEBUG=INFO nospec
  b8) — `NET/IB : Using` must list ONLY mlx5_0 on both nodes.
- Analysis: `scripts/analyze_1rail.py` (three-way table, implied f, ranking).

## Measurements

- Three-way table: tok/s (accept) per {config x batch}, speedups vs the
  1-rail no-spec.
- Implied f at 1 rail: 1-rail vs 8-rail no-spec step time at each batch (comm
  grew ~8x, compute didn't): f_1rail ~= 1 - t_8rail/t_1rail.
- Headline: at 1-rail serving batch (b64), the ranking of
  {no-spec, EAGLE-K1, EAGLE-K2, WorldA-K2}.

Results: [results_one_rail.md](results_one_rail.md).
