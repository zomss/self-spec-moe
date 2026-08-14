"""Data-free RTN weight-only INT{4,8} (WxA16) checkpoint, CPU, no GPU.

EfficientRollout 4.1: "apply lightweight RTN quantization to the FFN and QKVO
projection layers" -> targets=["Linear"], ignore=["lm_head"] is exactly that set for a
dense model. Weight-only (input_activations=None -> A16), so the verify path is
untouched and self-SD stays lossless via rejection sampling.

int4 sym -> uint4b8 and int8 sym -> uint8b128: BOTH ride vLLM's mixed-precision kernel
family (Machete auto / Marlin when forced), so W4-vs-W8 is a clean bit-width comparison
on one kernel. Asymmetric (SYM=0) uses zero-points -> uint4/uint8; the paper says
"simplest ASYMMETRIC RTN", so build it too and report tau for both.

Env: SRC_MODEL, OUT_DIR, BITS (4|8), SYM (1|0), GROUP (default 128).
Run in the ISOLATED llmcompressor venv (see make_ckpts.sh).
"""

import os

from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

SRC = os.environ.get("SRC_MODEL", "Qwen/Qwen2.5-7B-Instruct")
BITS = int(os.environ.get("BITS", "4"))
SYM = os.environ.get("SYM", "1") == "1"
GROUP = int(os.environ.get("GROUP", "128"))
OUT = os.environ.get(
    "OUT_DIR",
    os.path.expanduser(
        f"/data/smcho/ckpts/{SRC.split('/')[-1]}-W{BITS}A16-INT{BITS}-{'sym' if SYM else 'asym'}"
    ),
)

scheme = QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(
        num_bits=BITS,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        group_size=GROUP,
        symmetric=SYM,
        dynamic=False,
    ),
    input_activations=None,  # weight-only: A16
)
recipe = QuantizationModifier(config_groups={"group_0": scheme}, ignore=["lm_head"])

print(f"[ckpt] {SRC} -> W{BITS}A16 int{BITS} "
      f"{'symmetric' if SYM else 'asymmetric'} group={GROUP}", flush=True)
model = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype="bfloat16", device_map="cpu")
tok = AutoTokenizer.from_pretrained(SRC)
oneshot(model=model, recipe=recipe, output_dir=OUT)
tok.save_pretrained(OUT)
print(f"[ckpt] DONE -> {OUT}", flush=True)
