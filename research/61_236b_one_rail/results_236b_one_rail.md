# Phase 61 results — 236B + 1 rail: the final World A cell (negative, conclusive)

**Setup.** DeepSeek-V2 236B, 2-node TP2xDP4/node = EP16, NCCL restricted to
one rail per node (`NCCL_IB_HCA==mlx5_0`, engagement verified), gpu_mem 0.93
(0.95 OOMs by ~100 MiB on the rank sharing GPU0 with h106's stale 522 MiB
context), Phase 53/54 engine recipe otherwise. Same-HEAD 8-rail no-spec rerun
included (the Phase 53 references are stale — the branch got faster).

## The full table (same HEAD, same recipe; tok/s)

| batch | no-spec 8-rail | no-spec 1-rail | 1-rail cost | World A K1 @1-rail (accept) | WorldA/nospec |
|---:|---:|---:|---:|---:|---:|
| 8  | 283.0  | 225.5  | 25%  | 79.7 (1.82)  | 0.35x |
| 32 | 866.3  | 823.4  | 5%   | 236.9 (1.87) | 0.29x |
| 64 | 1347.7 | 1230.0 | 10%  | 354.5 (1.89) | **0.29x** |

## Takeaway 1 — one 400G NIC per node ABSORBS 236B EP16 decode traffic

An 8x NIC-bandwidth cut costs only 5-25% of the step. Together with Phase 60
(same result at 30B), this falsifies "thin fabric => high f" for modern
400G-class NICs at <=236B / 2-node scale: the decode all-to-all is
latency/overhead-bound, not wire-bandwidth-bound. The earlier f~0.5
attribution at 236B (Phase 53 roofline) was dominated by latency/launch/
software overheads that do NOT scale with NIC count. Genuinely
bandwidth-starved (high-f) decode requires much larger aggregate traffic
(huge batch x wide EP x many nodes) or legacy interconnects (<=100G) —
neither producible on this testbed.

## Takeaway 2 — World A is closed on every measurable cell

World A node-local at 236B/1-rail: **0.29-0.35x**, identical to 8-rail
(0.29-0.36x). The ratio is itself the f-probe: had f reached ~0.75, the
arithmetic guaranteed ~1.1-1.2x; 0.29x means the comm-free draft still costs
~2x the whole no-spec step. The comm-free-draft premise (large exposed comm
to save) does not exist at any (model, fabric, rail, context, batch, K)
combination this hardware can produce:

| cell | result |
|---|---|
| 30B, forced-PCIe single node (Ph 42-47) | 1.03-1.30x only vs emulated-inflated baselines |
| 30B, 8-rail 2-node, short ctx (Ph 52) | 0.35-0.49x |
| 236B, 8-rail 2-node (Ph 53-55) | 0.29-0.36x |
| 30B, 1-rail 2-node (Ph 60) | 0.45x |
| 236B, 1-rail 2-node (this) | 0.29-0.35x |
| long context (Ph 58 analysis) | worse (draft pays full KV) |

Verdict: self-speculative decoding via communication removal is CLOSED as a
method on real hardware of this class. Its validity regime (bandwidth-starved
decode) requires production many-node scale (tens of nodes, DeepSeek-EP320-
class) or legacy fabrics, and even there a dense trained head is comm-free by
construction at a fraction of the draft cost.

## Bookkeeping

Same-HEAD 8-rail 236B no-spec (283/866/1348) is 7-19% above the stale Phase 53
references (264/756/1130) — branch fixes sped up the baseline; use these
numbers for future comparisons. gpu_mem for 236B on HEAD: 0.93 (0.95 OOMs on
the zombie-sharing rank).
