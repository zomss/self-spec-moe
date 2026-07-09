# Quant-effectiveness study — infrastructure & kernel-readiness audit

Goal: characterize whether each quant type speeds the **model forward** (not self-spec)
vs full precision, on **dense (Qwen3-8B)** and **MoE (Qwen3-30B-A3B)**, per regime.
Maps directly to the read-vs-compute / binding-term model: weight-only = read cut only
(wins weight-I/O-bound); W8A8 = read + compute cut (wins compute-bound); KV-quant = KV
read cut (wins KV-bound). All kernel routing agent-verified on H100/SM90, flashinfer
0.6.12 installed, DeepGEMM absent. NO GPU runs yet (GPUs reserved).

## Models (✓ cached, same family — controlled dense-vs-MoE pair)
- **Dense**: `Qwen/Qwen3-8B` — hidden 4096, 36L, GQA 32/8 heads, head_dim 128, ~8B.
  Weight-I/O-heavy (full 8B read/token) → weight-quant should win at small-batch decode.
- **MoE**: `Qwen/Qwen3-30B-A3B` — 128 experts, ~3B active, EP4. KV-bound at long ctx.
- Spare MoEs if needed: Mixtral-8x7B, DeepSeek-V2-Lite, gpt-oss-20b.

## Harness (✓): `vllm bench latency`
`vllm/benchmarks/latency.py` — `--input-len/--output-len/--batch-size/--num-iters` +
`EngineArgs.add_cli_args` (so `--quantization`, `--kv-cache-dtype`,
`--enable-expert-parallel`, `-tp/-dp`, `--gpu-memory-utilization`, `--max-model-len`,
`--enforce-eager`). Plain model forward, no self-spec stack. **Regime control via sweep:**
- compute-bound: large `--batch-size` × long `--input-len` (prefill).
- weight/KV-I/O-bound: small batch, long `--input-len` (context), decode via `--output-len`.

## KERNEL-READINESS MATRIX (H100/SM90, agent-verified)

| # | cell | knob | kernel | native fp8 MACs? | checkpoint? | efficient? |
|---|---|---|---|---|---|---|
| 1 | DENSE weight-only fp8 (W8A16) | CT W8A16 ckpt + `VLLM_TEST_FORCE_FP8_MARLIN=1` | FP8-Marlin | **No (dequant)** | **Yes (ckpt)** | dequant only* |
| 2 | MoE weight-only fp8 (W8A16) | `--moe-backend marlin` | MarlinExperts | **No (dequant)** | No (on-the-fly) | dequant only* |
| 3 | DENSE W8A8 fp8 | `--quantization fp8` | **CUTLASS** | **Yes** | No | ✅ |
| 4 | MoE W8A8 fp8 | `--quantization fp8_per_block` (EP4) | **FlashInfer-CUTLASS** | **Yes** | No | ✅ |
| 5 | DENSE KV fp8 | `--kv-cache-dtype fp8_e4m3` | **FA3** | **Yes** | No | ✅ |
| 6 | MoE KV fp8 | `--kv-cache-dtype fp8_e4m3` | **FA3** | **Yes** | No | ✅ |

\* "dequant only" is **not a bug** — weight-only quant *inherently* can't use fp8 MACs
(operands must both be fp8; weight-only keeps fp16 activations → dequant→fp16 MACs). So
Marlin-dequant IS the weight-only fp8 kernel. Its only overhead beyond the read-cut is
the dequant tax (η_d). There is **no native weight-only fp8 kernel on SM90** — for dense
(only FP8-Marlin, `__init__.py:337-340`; Machete is int-only) or MoE (only MARLIN maps to
the W8A16 config, `oracle/fp8.py:511-522`; all other backends are W8A8). Inherent, expected.

> **UPDATE (measured, h106).** Cell 4 (MoE W8A8 block-fp8 → FlashInfer-CUTLASS) is now
> **verified engaged at runtime** on the self-spec DRAFT: `Using FLASHINFER_CUTLASS Fp8
> MoE backend`, on-the-fly from bf16, no checkpoint. Two caveats for anyone using this
> matrix: (1) reaching it from a *speculative* draft required a fix — online-quant
> shorthands were never desugared for the draft's ModelConfig (`draft_model.py`); (2)
> **cell 2 (MoE FP8-Marlin) is UNRUNNABLE on h106**: `cudaErrorUnsupportedPtxVersion`
> raised at load in `marlin_utils_fp8.py:270 repack_weight` →
> `prepare_fp8_moe_layer_for_marlin` (driver 580.65.06 vs CUDA-13.0 torch build). bf16
> and FI-CUTLASS ship SM90 cubins and run. Result: native fp8 is **bf16 parity**, not a
> win — see `results_draft_cost.md`.
>
> **Correction to an earlier draft of this note**, which said "cells 1-2 are unrunnable":
> the PTX failure is confined to the **fp8 MoE** Marlin repack. **Dense Marlin works on
> h106** — a dense int4 W4A16 `marlin_gemm` executes cleanly (Phase-75 `preflight.sh`),
> as does Machete. Cell 1 (dense weight-only fp8 Marlin) was never actually run here, so
> no claim should have been made about it.

## Readiness summary

- **Items 2 & 3 (cells 3,4,5,6) = READY NOW.** Native fp8, on-the-fly from bf16, **no
  checkpoint**. Runnable the moment GPUs free.
- **Item 1 (cell 1, dense weight-only fp8) = needs a decision** (see below).
- **MoE weight-only (cell 2)** = on-the-fly via `--moe-backend marlin`, dequant. Already
  measured in the self-spec draft (parity); trivially re-runnable on the plain forward.

## Item 1 checkpoint: BUILT ✓ (decision = fp8 W8A16 checkpoint)

`~/ckpts/Qwen3-8B-W8A16-FP8` — produced offline via isolated llmcompressor venv,
**CPU / data-free RTN** (no reserved-GPU use). Verified weight-only: `type=float`
(fp8), `strategy=channel`, `dynamic=false`, **`input_activations=null`** (A16),
`ignore=[lm_head]`, `quant_method=compressed-tensors`, `format=naive-quantized`,
`model.safetensors` 9.4 GB (~½ of bf16). Serve with `VLLM_TEST_FORCE_FP8_MARLIN=1`
(Marlin gated off ≥SM89). Wired into `run_quant_sweep.sh` (arms `d_wo_*`). Scripts:
`make_ckpt.sh` + `make_w8a16_fp8_ckpt.py`.

## (resolved) The decision: Item 1 (dense weight-only fp8)

Cell 1 is the only cell needing offline work: true weight-only fp8 (A16) on a dense model
is reachable ONLY via a **compressed-tensors W8A16-fp8 checkpoint** (fp8 weights, NO
activation quant) + `VLLM_TEST_FORCE_FP8_MARLIN=1` (Marlin is gated off ≥SM89 otherwise,
`scaled_mm/marlin.py:46-56`). There is no on-the-fly W8A16-fp8 shorthand. Options:

- **(A) Produce the CT W8A16-fp8 checkpoint** of Qwen3-8B offline (llm-compressor,
  `FP8` weight scheme, no input-quant), run with force-marlin. Clean weight-only-fp8
  measurement; dequant path (read-cut only, as intended).
- **(B) Substitute W4 weight-only** (int4, GPTQ/RTN → Marlin int4). This is the *better*
  read-cut lever (4× vs fp8's 2×) and matches EfficientRollout. Reachable on-the-fly
  (RTN) or via an int4 checkpoint. Recommended if the goal is "biggest read cut," not
  "fp8 specifically."
- **(C) Approximate via cell 3 (dense W8A8) in the decode regime.** At small-batch
  decode the compute cut is hidden (memory-bound), so W8A8 ≈ weight-only read-cut there;
  the prefill regime then isolates the added compute cut. Difference from true W8A16:
  W8A8 pays a small activation-quant cost even in decode. No checkpoint needed.

Recommendation: **(C) for a quick read** (no checkpoint, sweep cell 3 across regime),
then **(A) or (B)** if we want the clean weight-only isolation — with (B)/W4 the more
informative lever.

## Predicted results (from the binding-term model — the experiments test these)

- Item 1 (weight-only): **wins on dense** (weight-I/O-bound at small batch), **weak/parity
  on MoE** (sparse → weight not binding). Dequant tax caps it; W4 > fp8.
- Item 2 (W8A8): **wins compute-bound** (prefill / large batch) on both; in memory-bound
  decode reduces to the read cut only.
    - **SCORED on the MoE self-spec draft (h106): prediction FAILED.** Native block-fp8
      W8A8 (FI-CUTLASS) is bf16 **parity** at the compute-bound cell (2k, b64: 0.99x of
      the bf16 draft), not a win. Why the prediction missed: the *machine* is
      compute-bound at b64, but the **draft's MoE is not** — 64 tokens × top-8 over 128
      experts ≈ 4 tokens/expert, i.e. many tiny latency-bound GEMMs, so 2× MAC peak buys
      nothing. Regime labels must be applied to the *kernel that runs*, not the engine.
      Untested for the plain (non-spec) forward and for dense Qwen3-8B, where the GEMMs
      are large and the prediction may still hold.
- Item 3 (KV fp8): **wins KV-bound** (small-batch long-ctx decode), more on MoE (bigger KV
  share) than dense; helps the *system* globally (not a self-spec ratio lever).

## Next (GPU-gated)
`scripts/run_quant_sweep.sh` — cells 3,4,5,6 (ready) across {dense, MoE} × regimes vs bf16.
Each arm logs the engaged backend (must match the matrix) before the number is trusted.

## Reproduce on another server (handoff)

Checkpoints live in `~/ckpts/` and are **NOT in git** — rebuild them locally (CPU,
data-free RTN, no GPU):

```bash
git fetch && git checkout research/self-spec-moe && git pull
cd research/74_draft_step_cost
bash scripts/make_ckpt.sh        # -> ~/ckpts/Qwen3-8B-W8A16-FP8  (also makes ~/.cache/eff_lc_venv)
bash scripts/make_int4_ckpt.sh   # -> ~/ckpts/Qwen3-8B-W4A16-INT4 (reuses the venv)
# then, when GPUs are free (edit CUDA_VISIBLE_DEVICES in the script if not 4-7):
bash scripts/run_quant_sweep.sh  # Items 1-3 x {dense,MoE} x regimes vs bf16
```

Notes: needs `uv` + PyPI access for the isolated venv; `VLLM_TEST_FORCE_FP8_MARLIN=1`
is set inside the script for the fp8 weight-only arms; each arm logs the engaged
kernel — verify it matches the matrix before trusting a number.
