"""Data-free RTN weight-only INT4 (W4A16) checkpoint of Qwen3-8B — the stronger
read-cut lever (4x) than fp8 (2x), routed to Machete (SM90-native mixed-precision
dequant) / Marlin. int4 = symmetric, group_size 128 (GPTQ/Machete-friendly), NO
activation quant (input_activations=None -> A16). Data-free RTN, CPU, no GPU.
Run in the ISOLATED lc_venv (already has llmcompressor)."""
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.quantization import (
    QuantizationArgs, QuantizationScheme, QuantizationStrategy, QuantizationType,
)

SRC = os.environ.get("SRC_MODEL", "Qwen/Qwen3-8B")
OUT = os.environ.get("OUT_DIR", os.path.expanduser("/data/smcho/ckpts/Qwen3-8B-W4A16-INT4"))

scheme = QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,             # int4 (symmetric -> uint4b8 -> Machete/Marlin)
        strategy=QuantizationStrategy.GROUP,
        group_size=128,
        symmetric=True,
        dynamic=False,
    ),
    input_activations=None,                    # weight-only A16
)
recipe = QuantizationModifier(
    config_groups={"group_0": scheme},
    ignore=["lm_head"],
)

print(f"[ckpt] loading {SRC} on CPU (bf16)...", flush=True)
model = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype="bfloat16", device_map="cpu")
tok = AutoTokenizer.from_pretrained(SRC)
print("[ckpt] applying data-free RTN W4A16-int4 (weight-only, group128 sym)...", flush=True)
oneshot(model=model, recipe=recipe, output_dir=OUT)
tok.save_pretrained(OUT)
print(f"[ckpt] DONE -> {OUT}", flush=True)
