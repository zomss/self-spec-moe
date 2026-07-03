# Phase 53 results — DeepSeek-V2 (236B) on 2 real nodes: f clears the gate

**Setup.** DeepSeek-V2 (236B-A21B: 60 layers, hidden 5120, 160 routed + 2
shared experts, MLA), TP2 x DP8 = EP16 across h107+h106 (real fabric), bf16
target + FP8 EP-shard draft, K=1, harness `w7_2node.py` (two-length slope),
b8/32/64 per DP rank. Memory recipe that fits (three OOMs to find it):
`gpu_mem 0.90, max_model_len 1024, max_num_batched_tokens 2048,
cudagraph_capture_sizes {8,16,32,64,128}` -> weights 63 GB/GPU (target
42 + FP8 draft 21) + capped ~4 GB MoE workspace + ~4.5 GB KV + small pools.

## 1. Baseline (no-spec): the comm-bound regime is REAL at scale

| b/rank | tok/s/rank | step ms | roofline read |
|---:|---:|---:|---|
| 8  | 264  | 30.3 | comm ~15-18 ms vs compute ~11 -> f ~0.5-0.6 |
| 32 | 756  | 42.3 | f ~0.5 |
| 64 | 1130 | 56.7 | f ~0.45-0.55 |

The Phase 19 gate quantity, finally measured on real hardware at scale:
per-collective payloads (0.6-4.6 MB/rank) sit at/past the fabric's bandwidth
knee, so comm grows WITH batch and **f holds ~0.5 across the range** --
unlike Qwen3-30B on the same fabric (f 0.35->0.1, latency-floor regime).
Model scale, not fabric degradation, produces the comm-bound regime.

## 2. Spec (device-local EP-shard draft): MEASURED -- the draft is the binding constraint

First attempt OOM'd on a 15 GiB fused-MoE workspace (profile forward at the
8192-token default over the DP8 gather); fixed by max_num_batched_tokens=2048.

| b/rank | nospec tok/s | spec tok/s | speedup | accept @K=1 |
|---:|---:|---:|---:|---:|
| 8  | 264 | 92.9  | 0.35x | **1.078** |
| 32 | 756 | 173.3 | 0.23x | **1.087** |
| 64 | 1130 | (engine hang, dequeue timeout) | -- | -- |

Accept collapsed to ~1.08 (beta ~0.08) -- BELOW the ~1.3 predicted from the
V2-Lite slope: at 236B the shared-expert anchor does not rescue 6.25%
device-local coverage (V2-236B's shared mass fraction is smaller than
V2-Lite's -- the Phase 25 small-shared caveat, now measured). Flat in batch
(coverage-limited) as theory says. **Contrast with the node-local draft at
the same scale/fabric: accept 1.90 (Phase 54)** -- node-locality is worth
+0.8 tokens/cycle and is the difference between a useless and a viable
training-free draft at scale.

## 3. Known issue found at this scale (affects any gathering draft)

At b>=32, DeepSeek-V2's prefill is admitted in ~203-token chunks; the draft
proposes for early-finished requests DURING mixed prefill/decode steps, and
the draft forward's dp_metadata (DP-padded to the prefill chunk) disagrees
with its actual input (`AssertionError: 1 != 203` in AgRs all_gatherv via
`naive_dp_ep.prepare`). The device-local draft skips the gather and is
immune; any draft that actually dispatches (node-local, EP-full) crashes.
Fix = make the drafter's forwards carry their own coordinated sizes in mixed
steps (proposer surgery); until then gathering drafts are b8-clean only on
this model.
