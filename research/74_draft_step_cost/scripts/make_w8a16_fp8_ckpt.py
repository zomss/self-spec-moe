"""Produce a compressed-tensors WEIGHT-ONLY fp8 (W8A16) checkpoint of Qwen3-8B for
Item-1 of the quant-effectiveness study. Weight-only = fp8 weights, NO activation
quant (input_activations=None) -> vLLM routes it to the FP8-Marlin weight-only
kernel (needs VLLM_TEST_FORCE_FP8_MARLIN=1 at serve time on SM90).

Data-free RTN (no calibration dataset, no forward passes) -> runs on CPU, does NOT
touch the reserved GPUs. Run inside the ISOLATED llmcompressor venv (see make_ckpt.sh),
not the main vLLM venv.
"""
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.quantization import (
    QuantizationArgs, QuantizationScheme, QuantizationStrategy, QuantizationType,
)

SRC = os.environ.get("SRC_MODEL", "Qwen/Qwen3-8B")
OUT = os.environ.get("OUT_DIR", os.path.expanduser("~/ckpts/Qwen3-8B-W8A16-FP8"))

# fp8 weights, per-channel static, symmetric; NO input_activations -> weight-only A16.
scheme = QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(
        num_bits=8,
        type=QuantizationType.FLOAT,          # fp8 (e4m3)
        strategy=QuantizationStrategy.CHANNEL, # per-output-channel scale
        symmetric=True,
        dynamic=False,
    ),
    input_activations=None,                    # <-- the key: A16 (no activation quant)
)
recipe = QuantizationModifier(
    config_groups={"group_0": scheme},
    ignore=["lm_head"],
)

print(f"[ckpt] loading {SRC} on CPU (bf16)...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    SRC, torch_dtype="bfloat16", device_map="cpu",
)
tok = AutoTokenizer.from_pretrained(SRC)
print("[ckpt] applying data-free RTN W8A16-fp8 (weight-only)...", flush=True)
oneshot(model=model, recipe=recipe, output_dir=OUT)
tok.save_pretrained(OUT)
print(f"[ckpt] DONE -> {OUT}", flush=True)
