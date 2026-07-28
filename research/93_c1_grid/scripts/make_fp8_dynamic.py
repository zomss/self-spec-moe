"""Data-free FP8-dynamic (W8A8) checkpoint: fp8 weights + dynamic
per-token fp8 activations. CPU, no calibration.

The activation-quant arm of the phase-93 ladder (llmcompressor
FP8_DYNAMIC). MoE-FFN models (Qwen3-30B-A3B, DeepSeek-V2-Lite): the
router gates stay unquantized (IGNORE env), like lm_head.

Env: SRC_MODEL, OUT_DIR, IGNORE (comma list, default router-safe set).
Run in the isolated llmcompressor venv (phase-75 make_ckpts.sh).
"""
import os

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

SRC = os.environ["SRC_MODEL"]
# default NATIVE classes: V2-Lite remote code breaks on new
# transformers (is_torch_fx_available removed); native deepseek_v2 exists
TRUST = os.environ.get("TRUST_REMOTE", "0") == "1"
OUT = os.environ["OUT_DIR"]
IGNORE = os.environ.get(
    "IGNORE", "lm_head,re:.*mlp.gate$,re:.*shared_expert_gate$").split(",")

model = AutoModelForCausalLM.from_pretrained(
    SRC, dtype="bfloat16", trust_remote_code=TRUST)
tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=TRUST)
recipe = QuantizationModifier(
    targets="Linear", scheme="FP8_DYNAMIC", ignore=IGNORE)
oneshot(model=model, recipe=recipe)
model.save_pretrained(OUT, save_compressed=True)
tok.save_pretrained(OUT)
print("saved:", OUT)
