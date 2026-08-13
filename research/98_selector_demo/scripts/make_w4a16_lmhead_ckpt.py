# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a W4A16 draft checkpoint with lm_head ALSO quantized.

The existing draft (`Qwen3-8B-W4A16-INT4`) follows the standard recipe:
`targets=["Linear"], ignore=["lm_head"]`. That leaves `lm_head.weight` and
`model.embed_tokens.weight` in bf16 -- verified by reading the safetensors --
at 1.245 GB each, 2.49 GB total, **41% of the 6.07 GB checkpoint**.

For the draft that is expensive in a specific way: `lm_head` is a full-vocab
GEMM executed once per chain step, measured at 412.8 us/call and 90% of
bandwidth-bound, i.e. 4.98 GB/step of read traffic -- 31% of the draft's total
weight traffic. It is also the `F` term that breaks the cost model, because the
model multiplies ALL weight bytes by keep_frac while lm_head does not scale with
skipped layers at all.

Keeping lm_head in bf16 is the right default for a STANDALONE model, where its
output is the product. It is not obviously right for a speculative DRAFT: every
draft token is verified by the target, so a degraded draft logit is rejected
rather than emitted. Quantizing it cannot change correctness -- only the
acceptance rate, which is measurable.

This builds the same recipe with `ignore=[]` so lm_head is quantized too, into a
SEPARATE directory. The original checkpoint is not touched, so the two are an
A/B pair.

`model.embed_tokens` stays bf16 either way: it is an nn.Embedding, not a Linear,
so `targets=["Linear"]` never reaches it. Embeddings cost memory (1.245 GB of
resident weights against a KV cache G98-A measured at ~2.3% headroom) but almost
no decode time -- a row lookup is ~8 KB/step. Different problem, left alone.

Env: SRC_MODEL, OUT_DIR, BITS (default 4), SYM (default 1), GROUP (default 128).
Run in the isolated llmcompressor venv (see phase 75's make_ckpts.sh). CPU only.
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
BITS = int(os.environ.get("BITS", "4"))
SYM = os.environ.get("SYM", "1") == "1"
GROUP = int(os.environ.get("GROUP", "128"))
OUT = os.environ["OUT_DIR"]

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
    input_activations=None,  # weight-only: A16, so the verify path is untouched
)
# The ONLY difference from the standard recipe: lm_head is not excluded.
recipe = QuantizationModifier(config_groups={"group_0": scheme}, ignore=[])

print(
    f"[ckpt] {SRC} -> W{BITS}A16 int{BITS} group={GROUP} "
    f"{'sym' if SYM else 'asym'}, lm_head QUANTIZED",
    flush=True,
)
model = AutoModelForCausalLM.from_pretrained(
    SRC, torch_dtype="bfloat16", device_map="cpu"
)
tok = AutoTokenizer.from_pretrained(SRC)
oneshot(model=model, recipe=recipe, output_dir=OUT)
tok.save_pretrained(OUT)
print(f"[ckpt] DONE -> {OUT}", flush=True)
