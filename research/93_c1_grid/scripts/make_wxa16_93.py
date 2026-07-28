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
# default NATIVE classes: V2-Lite remote code breaks on new
# transformers (is_torch_fx_available removed); native deepseek_v2 exists
TRUST = os.environ.get("TRUST_REMOTE", "0") == "1"
OUT = os.environ["OUT_DIR"]
BITS = int(os.environ.get("BITS", "4"))
SYM = os.environ.get("SYM", "1") == "1"
GROUP = int(os.environ.get("GROUP", "128"))
IGNORE = os.environ.get(
    "IGNORE", "lm_head,re:.*mlp.gate$,re:.*shared_expert_gate$").split(",")

# GROUP=-1 -> channel-wise (vLLM's Marlin-MoE int8 path asserts
# group_size == -1; measured phase 93 step 1)
if GROUP == -1:
    weight_args = QuantizationArgs(
        num_bits=BITS, type=QuantizationType.INT,
        strategy=QuantizationStrategy.CHANNEL, symmetric=SYM,
        dynamic=False)
else:
    weight_args = QuantizationArgs(
        num_bits=BITS, type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP, group_size=GROUP,
        symmetric=SYM, dynamic=False)
scheme = QuantizationScheme(
    targets=["Linear"],
    weights=weight_args,
    input_activations=None,  # weight-only: A16
)

model = AutoModelForCausalLM.from_pretrained(
    SRC, dtype="bfloat16", trust_remote_code=TRUST)
tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=TRUST)
recipe = QuantizationModifier(
    config_groups={"group_0": scheme}, ignore=IGNORE)
oneshot(model=model, recipe=recipe)
model.save_pretrained(OUT, save_compressed=True)
tok.save_pretrained(OUT)
print("saved:", OUT)
