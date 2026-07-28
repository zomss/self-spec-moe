"""Data-free RTN weight-only INT{4,8} (WxA16) checkpoint, CPU.

Phase-93 variant of research/75_efficient_rollout_repro/scripts/
make_wxa16_ckpt.py: adds IGNORE env (router gates on MoE-FFN models
stay unquantized, like lm_head) and trust_remote_code (DeepSeek-V2-
Lite).

Env: SRC_MODEL, OUT_DIR, BITS (4|8), SYM (1|0), GROUP (128), IGNORE.
Run in the isolated llmcompressor venv.
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

SRC = os.environ["SRC_MODEL"]
OUT = os.environ["OUT_DIR"]
BITS = int(os.environ.get("BITS", "4"))
SYM = os.environ.get("SYM", "1") == "1"
GROUP = int(os.environ.get("GROUP", "128"))
IGNORE = os.environ.get(
    "IGNORE", "lm_head,re:.*mlp.gate$,re:.*shared_expert_gate$").split(",")

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

model = AutoModelForCausalLM.from_pretrained(
    SRC, dtype="bfloat16", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
recipe = QuantizationModifier(
    config_groups={"group_0": scheme}, ignore=IGNORE)
oneshot(model=model, recipe=recipe)
model.save_pretrained(OUT, save_compressed=True)
tok.save_pretrained(OUT)
print("saved:", OUT)
