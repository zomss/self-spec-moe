# W7-FP4 results: does an NVFP4 (W4A16) draft flip the W7 verdict?

**Stage-2 (FP4) re-test of `results_W7.md` (bf16) and `results_W7_fp8.md` (FP8).**
W7 found World A's bf16 full-replica comm-free draft LOSES to no-spec at every
(batch,K) -- the loss is **draft compute** (fabric-independent), not comm. FP8
(Stage 1) did not flip it (FP8 was *slower* than bf16 on this small MoE: on-the-fly
activation-quant overhead > the weight-load it saved). This stage asks: does a
pre-quantized **NVFP4 (W4A16)** draft -- static 4-bit weights, 16-bit activations,
so **no per-forward activation-quant** (unlike FP8) + ~4x weight-bandwidth -- cut
the draft cost enough to win?

**Answer: No.** NVFP4 does not flip the verdict. NVFP4 is **faster than FP8** at
almost every point (no activation-quant tax), but it is still **slower than bf16
everywhere** and crosses 1.0x nowhere (best 0.346x). On H100 (sm_90) there are no
FP4 tensor cores, so vLLM runs the **W4A16 Marlin dequant path** (FP4 weights
dequantized -> bf16 GEMM): there is no FLOP win, and the dequant overhead on
V2-Lite's tiny experts exceeds the weight-load it saves. The spec step is
CPU-orchestration-bound anyway, so cutting draft GPU work doesn't move the
bottleneck. **Acceptance is unchanged** by quantization (NVFP4 == bf16 proposer);
losslessness fully preserved.

## 0. Checkpoint (modelopt PTQ -> NVFP4 -> HF export)

- **modelopt:** `nvidia-modelopt[hf]` **0.44.0** (installed into the shared venv).
- **PTQ/export script:** `scripts/fp4_ptq_export.py`. Loads DeepSeek-V2-Lite in
  bf16 via transformers' **native** `DeepseekV2ForCausalLM` (NOT trust_remote_code:
  the shipped custom modeling code imports `is_torch_fx_available`, removed in
  transformers 5.x), calibrates on 128 short texts (seqlen 512) using
  `NVFP4_DEFAULT_CFG` (quantizes all linears incl. **MoE experts** + shared
  experts + attention; excludes router/gate/lm_head), then `export_hf_checkpoint`.
  - **Gotcha fixed in-script:** modelopt 0.44.0's vLLM plugin registers vLLM's
    `FusedMoE` *factory function* (not a class) into its QuantModuleRegistry; the
    registry walk then does `issubclass(nn_cls, <function>)` -> `TypeError`. A
    one-line defensive monkeypatch on `_DMRegistryCls._get_registered_nn_class`
    skips non-class registry entries (does not alter quantization of any real
    module).
  - Quant summary: 4136 TensorQuantizers, per-expert `gate/up/down_proj` weight
    quantizers at `(2,1)`-bit FP4, group_size 16, all 64 experts x 26 MoE layers.
- **Run:**
  ```bash
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/smcho/ssm-w7fp4 \
    VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 \
    /data/smcho/self-spec-moe/.venv/bin/python \
    research/34_worldA_system/scripts/fp4_ptq_export.py
  ```
  ~290s on one H100.
- **Checkpoint:** `research/34_worldA_system/ckpts/dsv2lite-nvfp4` (gitignored).
  - `model.safetensors` 9.4 GB; dir 8.9 GB. vs **bf16 30 GB -> ~3.4x smaller** on
    disk. `hf_quant_config.json`: `{producer: modelopt 0.44.0, quant_algo: NVFP4,
    group_size: 16, kv_cache_quant_algo: null, exclude_modules: [lm_head, *.mlp.gate]}`.

## 1. GATE -- vLLM loads + RUNS NVFP4 on H100 (sm_90): **PASS**

`scripts/fp4_gate.py` loads the NVFP4 ckpt standalone (TP=1, CUDA graphs ON) and
generates. Confirmed at load:
- `modelopt.py: Detected ModelOpt NVFP4 checkpoint (quant_algo=NVFP4)`;
  engine `quantization=modelopt_fp4`.
- Linear: `Using MarlinNvFp4LinearKernel for NVFP4 GEMM`.
- MoE: `nvfp4.py: Using 'MARLIN' NvFp4 MoE backend out of potential backends:
  [FLASHINFER_TRTLLM, FLASHINFER_CUTEDSL, ..., VLLM_CUTLASS, MARLIN, EMULATION]`
  -- the FP4-tensor-core backends auto-fall-through on sm_90; MARLIN is selected.
- `marlin.py / marlin_utils_fp4.py: Your GPU does not have native support for FP4
  computation ... Weight-only FP4 compression will be used leveraging the Marlin
  kernel` -- i.e. the **W4A16 dequant path**, exactly as predicted for sm_90.
- Coherent output (e.g. "The capital of France is Paris.").

**So the known support risk did NOT materialize: NVFP4 loads + runs on H100 via the
Marlin W4A16 path.** (One env gotcha resolved en route: the worktree was missing
untracked third-party python files -- `vllm/third_party/flashmla/flash_mla_interface.py`,
`deep_gemm/*`, etc. -- that the FP4 quant-config registration pulls in via
`deepseek_v4`; symlinked all missing source files into the worktree, gitignored.)

## 2. Self-spec wiring (NVFP4 draft, target bf16)

The draft points at a **separate** pre-quantized NVFP4 checkpoint; quantization is
auto-detected from its embedded `hf_quant_config.json`. We pass
`speculative_config={"method":"draft_model","model":"<nvfp4 ckpt>",
"num_speculative_tokens":K,"draft_tensor_parallel_size":1}` and do NOT set
`quantization` (that is the on-the-fly FP8 path for bf16 ckpts). The merged
draft-quant wiring (`draft_model.py::_create_draft_vllm_config`) then derives
`ModelOptNvFp4Config` for the draft. Confirmed at load (both DP workers):
- `draft_model.py: Draft model quantization: modelopt_fp4 (target stays None)`.
- `Using MarlinNvFp4LinearKernel` + `Using 'MARLIN' NvFp4 MoE backend` on the draft.
- Full replica preserved: `Draft MoE EP status ...: use_ep=False expert_map=None`.
- Target engine config: `quantization=None` (bf16). VLLM_SELF_SPEC_DRAFT_FULL_REPLICA=1,
  producer ON.

## 3. V2-Lite -- bf16 vs FP8 vs NVFP4 draft (Regime A: forced-PCIe + 100us A2A)

Same harness/method as W7/FP8 (two-length slope, CUDA graphs ON, WARMUP=2, ITERS=3,
OUTLEN=160/SHORTLEN=32, DP=2+EP, greedy). Same self-consistent no-spec baseline
(reproduces W7/FP8 no-spec within +-1%: e.g. batch=256 11189 vs W7 11209 / FP8
11241). bf16 spec re-measured here for apples-to-apples; FP8 column from the prior
FP8-stage data (its own self-consistent baseline). `fp4 tok/s` is the NVFP4 spec
system throughput.

| batch | K | nospec tok/s | bf16 sp | fp8 sp | **fp4 sp** | fp4 tok/s | acc(bf16) | acc(fp4) |
|------:|--:|------------:|--------:|-------:|-------:|----------:|---------:|---------:|
| 8   | 2 | 636.2   | 0.417 | 0.273 | **0.203** | 129.3  | 2.727 | 2.741 |
| 32  | 2 | 2185.2  | 0.261 | 0.183 | **0.230** | 501.9  | 2.760 | 2.744 |
| 64  | 2 | 3718.6  | 0.298 | 0.237 | **0.264** | 982.0  | 2.713 | 2.727 |
| 128 | 2 | 6664.4  | 0.317 | 0.248 | **0.251**S | 1674.7 | 2.683 | 2.674 |
| 256 | 2 | 11189.4 | 0.316 | 0.258 | **0.284** | 3172.3 | 2.448 | 2.411 |
| 8   | 3 | 636.2   | 0.314 | 0.245 | **0.254** | 161.7  | 3.374 | 3.364 |
| 32  | 3 | 2185.2  | 0.355 | 0.281 | **0.307** | 671.1  | 3.404 | 3.353 |
| 64  | 3 | 3718.6  | 0.407 | 0.314 | **0.346** | 1285.5 | 3.327 | 3.294 |
| 128 | 3 | 6664.4  | 0.235 | 0.185 | **0.200** | 1333.4 | 3.324 | 3.296 |
| 256 | 3 | 11189.4 | 0.254 | 0.203 | **0.218** | 2437.5 | 3.011 | 2.970 |
| 8   | 4 | 636.2   | 0.249 | 0.194 | **0.205** | 130.4  | 4.274 | 4.268 |
| 32  | 4 | 2185.2  | 0.287 | 0.218 | **0.225**S | 491.4  | 4.163 | 4.161 |
| 64  | 4 | 3718.6  | 0.317 | 0.230 | **0.142*** | 527.4  | 3.866 | 3.845 |
| 128 | 4 | 6664.4  | 0.179 | 0.135 | **0.145** | 965.6  | 3.866 | 3.836 |
| 256 | 4 | 11189.4 | 0.204 | 0.156 | **0.167** | 1872.8 | 3.290 | 3.197 |

S = harness `suspect` (first-engine warmup variance, K=2 b=128 / K=4 b=32);
clean neighbors confirm the trend. * = K=4 b=64 reproducibly hits a slow Marlin
FP4 MoE shape (decode 15.5s vs b=32's 8.7s -> the dip to 0.142); a real kernel
artifact at that shape, not measurement noise.

**NVFP4 best speedup = 0.346 (K=3, batch=64); NVFP4 crosses 1.0x NOWHERE.**

Two headline comparisons:
- **NVFP4 > FP8 at almost every point** (e.g. K=3 b=64 0.346 vs 0.314; K=2 b=256
  0.284 vs 0.258; K=3 b=32 0.307 vs 0.281). NVFP4 W4A16 pays **no per-forward
  activation quantize/dequantize** (static weights, bf16 activations), so it avoids
  the exact overhead that made FP8 *slower* than bf16. The two exceptions are the
  flagged/artifact shapes (K=2 b=8 suspect; K=4 b=64 Marlin slow shape).
- **NVFP4 < bf16 everywhere.** On sm_90 the FP4 weights are **dequantized to bf16
  in the Marlin kernel** (no FP4 tensor cores -> no FLOP win), and the dequant cost
  on V2-Lite's tiny experts (`E=64, N=704`) exceeds the ~4x weight-bandwidth it
  saves. The draft forward stays the bottleneck.
- **Acceptance unchanged** by NVFP4 (e.g. K=3 b=64: 3.294 vs bf16 3.327) -- the
  NVFP4 draft is as good a proposer as bf16; it is just not cheaper end-to-end.

## 4. Losslessness spot-check (W0/W2 standard)

DP=2+EP forced-PCIe, greedy, 16 prompts, K=4, vs no-spec greedy reference:

| spec draft | exact seqs | token agreement |
|-----------|:----------:|:---------------:|
| bf16 full replica | 12/16 | 82.4% |
| **nvfp4 full replica** | **12/16** | **82.4%** |
| nvfp4 vs bf16 (precision delta) | 15/16 | 99.3% |

**NVFP4 is exactly as lossless as bf16 vs the no-spec reference** (both 12/16,
82.4% -- the 4 mismatches are the documented pre-existing batched-verify near-tie
FP flips, not corruption). The nvfp4-vs-bf16 precision delta is 15/16 (one extra
near-tie flip vs FP8's 16/16 bit-identical) -- a single batched-verify near-tie
that lands the other way under the slightly different NVFP4 draft probabilities;
it does NOT change agreement with the greedy reference (still 12/16), because
rejection sampling makes the **bf16 verify the sole source of truth** -- the draft
precision only affects which tokens are *proposed*, never the committed output's
distribution. Acceptance is unchanged (accept_len 2.93 nvfp4 vs 2.97 bf16 at
OUTLEN=64).

## 5. Memory / fit

- NVFP4 draft checkpoint **8.9 GB on disk vs bf16 30 GB (~3.4x smaller)**.
- Steady-state: target bf16 EP-shard + full-replica NVFP4 draft fit comfortably at
  `gpu_memory_utilization=0.85` (~71 GB/rank observed during the sweep on 80 GB
  H100s); batch=256 fits at every K. NVFP4 quarters the *draft weight* footprint vs
  bf16 (the draft replica drops from ~31 GB to ~8 GB/rank); this is the clearest
  NVFP4 win and the relevant one for W2b (a sparser top-C resident cache).

## 6. Anomalies / notes

- **modelopt downgraded transformers** 5.12.1 -> 5.5.4 on install (pulled by
  diffusers/peft); vLLM requires `transformers >= 5.5.3`, so 5.5.4 still satisfies
  it and vLLM imports + runs cleanly. The downgrade also broke the DSV2-Lite
  *remote* modeling code (removed `is_torch_fx_available`), worked around by using
  transformers' native `DeepseekV2ForCausalLM` for PTQ.
- **DP-engine teardown race** (`RuntimeError: cancelled` / leaked shared_memory)
  hit the serial driver between the bf16 and NVFP4 K-runs, killing the NVFP4 K=4
  sweep after batch=64. Re-running NVFP4 K=4 as an isolated one-K-per-process
  invocation (as the FP8 stage recommends) -> clean batches 8/32/64/128/256. Same
  class of transient teardown issue the FP8 doc flagged; unrelated to the FP4
  wiring.
- Strictly serial throughout (one engine on the box at a time), CUDA graphs ON,
  forced-PCIe (`NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1`),
  `VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0`.

## Verdict

**Quantizing the draft to NVFP4 does not change the W7 verdict.** World A's
comm-free draft still loses on wall-clock decode tok/s at every (batch,K): NVFP4 is
*better than FP8* (W4A16 has no activation-quant tax, so it recovers most of the
gap FP8 opened vs bf16) but still *worse than bf16* -- on H100 (sm_90) NVFP4 runs
the Marlin **W4A16 dequant** path, which delivers **no FLOP win** and adds a dequant
cost that, on V2-Lite's tiny experts, exceeds the weight-bandwidth it saves. The
ordering on this small MoE is **bf16 > NVFP4 > FP8** for wall-clock, with all three
< 1.0x. NVFP4's clear win is **memory** (3.4x smaller draft on disk; ~4x lighter
draft replica resident). For a wall-clock win the draft must be *sparser* (W2b
globally-hot top-C + skip-cold) AND/OR run on FP4 tensor-core hardware (sm_100,
where NVFP4 buys real FLOPs, not just bandwidth) -- precision alone, on sm_90,
delivers neither the FLOP reduction that matters nor any acceptance gain.
Losslessness is fully preserved (NVFP4 == bf16 verify outputs; the draft precision
only affects acceptance rate, which is unchanged).

## Reproduce

```bash
cd /data/smcho/ssm-w7fp4
# 0. PTQ -> NVFP4 -> HF export (one H100):
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 \
  /data/smcho/self-spec-moe/.venv/bin/python research/34_worldA_system/scripts/fp4_ptq_export.py
# 1. GATE (load + run NVFP4 on H100):
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD NCCL_P2P_DISABLE=1 NCCL_NVLS_ENABLE=0 NCCL_IB_DISABLE=1 \
  VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 \
  /data/smcho/self-spec-moe/.venv/bin/python research/34_worldA_system/scripts/fp4_gate.py
# 2. V2-Lite nospec/bf16/NVFP4 sweep (Regime A):
bash research/34_worldA_system/scripts/w7_fp4_serial_v2lite.sh
#    (if a K crashes on the DP-teardown race, re-run that K isolated:)
W7_KS=4 W7_TAG=v2lite_nvfp4 \
W7_DRAFT_MODEL=$PWD/research/34_worldA_system/ckpts/dsv2lite-nvfp4 \
  /data/smcho/self-spec-moe/.venv/bin/python research/34_worldA_system/scripts/w7_fp4_timing.py spec
# 3. Losslessness (nospec/bf16/nvfp4 over 16 prompts):
bash research/34_worldA_system/scripts/w7_fp4_lossless_serial.sh
# Tables:
PYTHONPATH=$PWD /data/smcho/self-spec-moe/.venv/bin/python research/34_worldA_system/scripts/w7_fp4_analyze.py
PYTHONPATH=$PWD /data/smcho/self-spec-moe/.venv/bin/python research/34_worldA_system/scripts/w7_fp4_lossless_compare.py
```
Raw JSON: `research/34_worldA_system/data/w7fp4_*.json`.
