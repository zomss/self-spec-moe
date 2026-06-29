# W7-FP8 results: does a quantized (FP8) draft flip the W7 verdict?

**Stage-1 (FP8) re-test of `results_W7.md`.** W7 found World A's bf16 full-replica
comm-free draft LOSES to no-spec at every (batch,K) -- the loss is **draft compute**
(fabric-independent), not comm. This stage asks: does an on-the-fly **FP8** draft
(experts FP8 -> ~2x lighter weight-load + FLOPs) cut that draft cost enough to win?

**Answer: No.** FP8 does not flip the verdict. On DeepSeek-V2-Lite a FP8 draft is
actually **slower** than the bf16 draft at every point (on-the-fly quant overhead
on a small MoE outweighs the weight-load it saves). On Qwen3-30B-A3B the FP8
full-replica draft fits and runs but still loses to no-spec (0.15-0.26x). The draft
stays the bottleneck.

## 1. Wiring change (FP8 draft, target stays bf16)

Single additive, flag-gated change in `vllm/v1/spec_decode/draft_model.py`
(`_create_draft_vllm_config`): instead of unconditionally forcing
`quant_config=None`, honor `speculative_config.quantization`. The draft
`ModelConfig` already receives `quantization=self.quantization`
(`speculative.py:729`); we now derive the draft quant config from it via
`VllmConfig.get_quantization_config(...)` when set, else keep `None` (= W0 bf16,
default unchanged). One gotcha handled: the draft `ModelConfig.hf_overrides` is the
`SpeculativeConfig.hf_config_override` callable, but `get_quant_config` indexes it
as a dict on the on-the-fly path (bf16 ckpt has no embedded quant config), so we
normalize it to `{}` on a copy before deriving.

Run config: `speculative_config={..., "quantization": "fp8"}` (the harness sets it
from `W7_DRAFT_QUANT=fp8`). On-the-fly FP8 is native on H100; for a bf16 checkpoint
`get_quant_config` returns `Fp8Config(is_checkpoint_fp8_serialized=False)`, which
selects `Fp8OnlineMoEMethod` -> the draft's MoE **experts** are quantized to FP8.

**Confirmed at load (both DeepSeek-V2-Lite and Qwen3-30B):**
- `draft_model.py: Draft model quantization: fp8 (target stays None)` -- draft FP8,
  target bf16.
- draft MoE runs the `dtype=fp8_w8a8` fused-MoE kernel (V2-Lite log:
  `E=64,N=704,...,dtype=fp8_w8a8.json`).
- full replica preserved: `Draft MoE EP status ... use_ep=False expert_map=None`.

## 2. V2-Lite -- controlled bf16-vs-FP8 draft (Regime A: forced-PCIe + 100us A2A)

Same harness/method as W7 (two-length slope, CUDA graphs ON, WARMUP=2, ITERS=3,
OUTLEN=160/SHORTLEN=32, DP=2+EP, greedy). Same self-consistent no-spec baseline for
both bf16 and FP8 (reproduces W7 Regime-A no-spec within ~1%: e.g. batch=256 11241
vs W7 11209). bf16 spec re-measured here for an apples-to-apples comparison.

| batch | K | nospec tok/s | bf16 spec tok/s | **bf16 sp** | fp8 spec tok/s | **fp8 sp** | acc(bf16) | acc(fp8) |
|------:|--:|------------:|---------------:|--------:|--------------:|-------:|---------:|---------:|
| 8   | 2 | 631.3   | 267.4  | **0.424** | 172.1  | **0.273**¹ | 2.727 | 2.723 |
| 32  | 2 | 2207.6  | 567.3  | **0.257** | 405.0  | **0.183**¹ | 2.760 | 2.731 |
| 64  | 2 | 3681.0  | 1106.9 | **0.301** | 872.0  | **0.237**  | 2.713 | 2.706 |
| 128 | 2 | 6675.2  | 2107.7 | **0.316** | 1657.3 | **0.248**  | 2.683 | 2.671 |
| 256 | 2 | 11240.9 | 3590.8 | **0.319** | 2902.1 | **0.258**  | 2.448 | 2.439 |
| 8   | 3 | 631.3   | 197.7  | **0.313** | 154.4  | **0.245**  | 3.374 | 3.415 |
| 32  | 3 | 2207.6  | 769.0  | **0.348** | 620.3  | **0.281**  | 3.404 | 3.375 |
| 64  | 3 | 3681.0  | 1505.2 | **0.409** | 1155.6 | **0.314**  | 3.327 | 3.319 |
| 128 | 3 | 6675.2  | 1554.1 | **0.233** | 1233.7 | **0.185**  | 3.324 | 3.318 |
| 256 | 3 | 11240.9 | 2848.9 | **0.253** | 2280.8 | **0.203**  | 3.011 | 2.993 |
| 8   | 4 | 631.3   | 159.4  | **0.252** | 122.3  | **0.194**  | 4.274 | 4.280 |
| 32  | 4 | 2207.6  | 618.9  | **0.280** | 481.0  | **0.218**  | 4.163 | 4.172 |
| 64  | 4 | 3681.0  | 1161.7 | **0.316** | 845.8  | **0.230**  | 3.866 | 3.868 |
| 128 | 4 | 6675.2  | 1171.2 | **0.175** | 900.8  | **0.135**  | 3.866 | 3.853 |
| 256 | 4 | 11240.9 | 2250.9 | **0.200** | 1757.6 | **0.156**  | 3.290 | 3.274 |

¹ K=2 batch=8/32 FP8 flagged `suspect` (first FP8 engine warmup variance); the clean
K=2 batch>=64 and all K=3/K=4 rows confirm the trend.

**FP8 best speedup = 0.314 (K=3,batch=64); FP8 crosses 1.0x nowhere.**
**FP8 is slower than bf16 at EVERY (batch,K).** On-the-fly FP8 adds a per-step
quantize/dequantize cost; on a small MoE (V2-Lite experts are tiny) that overhead
exceeds the weight-load/FLOP it saves, and the spec step is CPU-orchestration-bound
anyway, so cutting GPU work on the draft doesn't move the bottleneck. **Acceptance is
unchanged** by FP8 (e.g. K=3 batch=64: 3.319 vs bf16 3.327) -- the FP8 draft is as
good a proposer as bf16; it's just not cheaper end-to-end here.

## 3. Qwen3-30B-A3B -- FP8 full-replica draft vs no-spec (Regime A)

`Qwen/Qwen3-30B-A3B` (128 experts, top-8, 48 layers, ~30B total / 3B active),
DP=2+EP forced-PCIe, greedy, CUDA graphs ON, same method. **The FP8 full-replica
draft FIT at DP=2** (no OOM, no fallback): target bf16 EP-shard + full-replica FP8
draft both resident on 80GB H100s. (FP8 halving the draft to ~30GB/rank is what makes
the full replica fit alongside the ~32GB/rank bf16 target shard.)

| batch | K | nospec tok/s | fp8 spec tok/s | **fp8 sp** | acc(fp8) |
|------:|--:|------------:|--------------:|-------:|---------:|
| 8  | 2 | 396.6  | 101.4 | **0.256** | 2.197 |
| 32 | 2 | 1285.4 | 262.9 | **0.205** | 2.028 |
| 64 | 2 | 2278.0 | 509.3 | **0.224** | 1.911 |
| 8  | 3 | 396.6  | 75.3  | **0.190** | 2.474 |
| 32 | 3 | 1285.4 | 191.6 | **0.149** | 2.200 |
| 64 | 3 | 2278.0 | 378.1 | **0.166** | 2.068 |

**The bigger model did NOT win: FP8 spec is 0.15-0.26x no-spec.** Two compounding
reasons: (a) acceptance is LOW (accept-len 1.9-2.5; the full-replica draft proposes
poorly on Qwen3 under greedy), so few draft tokens are accepted per step; (b) the
full-replica draft forward runs ALL 128 experts K times per step -- the draft forward
is heavy (a 30B model run K extra times), and even at FP8 it dwarfs the comm it
removes. The comm/compute ratio did not tip in spec's favor; the draft forward
dominates exactly as the analysis worried, but via a different term (forward cost,
not draft-loop CPU overhead).

## 4. Losslessness spot-check (W0/W2 standard)

DP=2+EP forced-PCIe, greedy, 16 prompts, K=4, vs no-spec greedy reference:

| spec draft | exact seqs | token agreement |
|-----------|:----------:|:---------------:|
| bf16 full replica | 12/16 | 82.4% |
| **fp8 full replica** | **12/16** | **82.4%** |
| fp8 vs bf16 (precision delta) | **16/16** | **100.0%** |

**FP8 is exactly as lossless as bf16** (both match W7/W2's 12/16, 82.4% -- the 4
mismatches are the documented pre-existing batched-verify near-tie FP flips, not
corruption). And **fp8 == bf16 bit-for-bit (16/16, 100%)**: quantizing the draft to
FP8 changes NO output token, because rejection sampling makes the (bf16) verify the
sole source of truth -- the draft precision only affects acceptance rate, which here
is unchanged.

## 5. Memory / fit

- V2-Lite: target shard + full-replica FP8 draft fit comfortably at gpu_mem=0.85.
- Qwen3-30B: full-replica FP8 draft **fit at DP=2** (gpu_mem=0.90), no OOM. This is a
  genuine FP8 win on the memory axis (a bf16 full replica of a 30B model would be
  ~60GB/rank and would not fit alongside the target shard).

## 6. Anomalies / notes

- A transient `EADDRINUSE` DP-master-port race hit the V2-Lite FP8 K=3/K=4 and one
  losslessness run when a new per-K engine started before the previous engine's
  TCPStore released its port (FP8 engine teardown is slower than bf16). Re-run as
  isolated one-K-per-process invocations -> clean, low-variance numbers (the table
  above). Unrelated to the FP8 wiring.
- Strictly serial throughout (one engine on the box at a time), CUDA graphs ON,
  forced-PCIe (`NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1`),
  `VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0`.

## Verdict

**Quantizing the draft to FP8 does not change the W7 verdict.** World A's comm-free
draft still loses on wall-clock decode tok/s: on V2-Lite FP8 is *slower* than bf16
(on-the-fly quant overhead > the weight-load it saves on a small MoE), and on
Qwen3-30B the heavier FP8 full-replica draft forward (all 128 experts x K) plus low
acceptance keeps spec at 0.15-0.26x no-spec. FP8's only clear win here is **memory**
(the 30B full replica fits at DP=2). For a wall-clock win the draft must be both
*cheap* (fewer experts -- W2b globally-hot top-C + skip-cold, not just lower
precision) AND a high enough acceptance to amortize K forwards; FP8 alone delivers
neither the FLOP reduction that matters (the draft-loop / full-replica-forward cost
dominates) nor any acceptance gain. Losslessness is fully preserved (FP8 == bf16
bit-for-bit). Recommendation for the FP4 stage: FP4 will help memory further but, on
this evidence, will not by itself flip wall-clock unless paired with a *sparser*
(top-C, not full-replica) draft.

## Reproduce

```bash
cd /data/smcho/ssm-w7q
# V2-Lite controlled bf16-vs-FP8 (Regime A):
bash research/34_worldA_system/scripts/w7_fp8_serial_v2lite.sh
# Qwen3-30B FP8 vs no-spec:
bash research/34_worldA_system/scripts/w7_fp8_serial_qwen30b.sh
# Losslessness (nospec/bf16/fp8 over 16 prompts):
PYTHONPATH=/data/smcho/ssm-w7q ... python research/34_worldA_system/scripts/w7_fp8_lossless.py {nospec|bf16|fp8}
# Tables:
python research/34_worldA_system/scripts/w7_fp8_analyze.py
```
Raw JSON: `research/34_worldA_system/data/w7fp8_*.json`.
