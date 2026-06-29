"""PTQ DeepSeek-V2-Lite -> NVFP4 and export an HF checkpoint (W2c, Stage-2).

Uses NVIDIA TensorRT Model Optimizer (modelopt) to post-training-quantize the
MoE expert weights (and other linears) of DeepSeek-V2-Lite to NVFP4 with a small
calibration set, then exports a standard HF checkpoint (with hf_quant_config.json,
NVFP4 algo) that vLLM can deserialize via ModelOptNvFp4Config.

The point of W2c is quantizing the routed MoE experts to 4-bit; NVFP4_DEFAULT_CFG
quantizes all linears (experts included) and excludes router/gate/lm_head, exactly
the faithful-strategy precision target.

Run (single GPU):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/smcho/ssm-w7fp4 \
    VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 \
    /data/smcho/self-spec-moe/.venv/bin/python \
    research/34_worldA_system/scripts/fp4_ptq_export.py
"""
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint

# modelopt 0.44.0's vLLM plugin registers vLLM's `FusedMoE` factory (a FUNCTION,
# not a class) into the QuantModuleRegistry; because vLLM is importable from the
# shared venv, the plugin loads and poisons the registry. The registry walk then
# calls issubclass(nn_cls, <function FusedMoE>) and raises
# "TypeError: issubclass() arg 2 must be a class". Guard the lookup to skip any
# non-class registry entry (purely defensive; does not alter quantization of any
# real nn.Module).
import modelopt.torch.opt.dynamic as _mo_dynamic  # noqa: E402

_orig_get_registered = _mo_dynamic._DMRegistryCls._get_registered_nn_class


def _safe_get_registered_nn_class(self, nn_cls):
    for nn_cls_ in self._registry:
        if not isinstance(nn_cls_, type):
            continue
        if issubclass(nn_cls, nn_cls_) and nn_cls.forward is nn_cls_.forward:
            return nn_cls_
    return None


_mo_dynamic._DMRegistryCls._get_registered_nn_class = _safe_get_registered_nn_class

MODEL = os.environ.get(
    "FP4_MODEL",
    "/home/smcho/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite"
    "/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0",
)
EXPORT_DIR = os.environ.get(
    "FP4_EXPORT_DIR",
    "/data/smcho/ssm-w7fp4/research/34_worldA_system/ckpts/dsv2lite-nvfp4",
)
N_CALIB = int(os.environ.get("FP4_NCALIB", "128"))
CALIB_SEQLEN = int(os.environ.get("FP4_SEQLEN", "512"))


def get_calib_texts(tokenizer, n_samples):
    """Small calibration set from C4 (fall back to a fixed text list)."""
    try:
        ds = load_dataset(
            "allenai/c4", "en", split="train", streaming=True,
        )
        texts = []
        for ex in ds:
            t = ex.get("text", "")
            if len(t) > 200:
                texts.append(t)
            if len(texts) >= n_samples:
                break
        if texts:
            return texts
    except Exception as e:
        print(f"[FP4] C4 unavailable ({e!r}); using fallback corpus.")
    base = [
        "The history of artificial intelligence began in antiquity.",
        "In mathematics, a prime number is a natural number greater than one.",
        "The mitochondrion is the powerhouse of the cell, producing ATP.",
        "Quantum mechanics describes nature at the smallest scales of energy.",
        "The French Revolution was a period of radical political change.",
        "Photosynthesis converts light energy into chemical energy in plants.",
        "A compiler translates source code into machine code for execution.",
        "The theory of relativity transformed our understanding of space and time.",
    ]
    return [(base[i % len(base)] + " ") * 20 for i in range(n_samples)]


def main():
    t_start = time.time()
    print(f"[FP4] modelopt PTQ -> NVFP4 for {MODEL}")
    print(f"[FP4] export_dir={EXPORT_DIR} n_calib={N_CALIB} seqlen={CALIB_SEQLEN}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use transformers' NATIVE deepseek_v2 implementation (NOT trust_remote_code):
    # the checkpoint ships custom modeling code that imports
    # is_torch_fx_available, which transformers 5.x removed. The native class
    # avoids the broken remote code entirely.
    from transformers import DeepseekV2ForCausalLM

    model = DeepseekV2ForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    calib_texts = get_calib_texts(tokenizer, N_CALIB)
    print(f"[FP4] got {len(calib_texts)} calibration texts")

    def forward_loop(m):
        for i, text in enumerate(calib_texts):
            enc = tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=CALIB_SEQLEN,
            )
            input_ids = enc["input_ids"].to("cuda")
            with torch.no_grad():
                m(input_ids)
            if (i + 1) % 16 == 0:
                print(f"[FP4]   calib {i + 1}/{len(calib_texts)}", flush=True)

    cfg = mtq.NVFP4_DEFAULT_CFG
    print("[FP4] quantizing (NVFP4_DEFAULT_CFG: experts + linears, "
          "router/gate/lm_head excluded) ...")
    t0 = time.time()
    model = mtq.quantize(model, cfg, forward_loop)
    print(f"[FP4] quantize done in {time.time() - t0:.1f}s")
    mtq.print_quant_summary(model)

    os.makedirs(EXPORT_DIR, exist_ok=True)
    print(f"[FP4] exporting HF checkpoint -> {EXPORT_DIR}")
    t0 = time.time()
    export_hf_checkpoint(model, export_dir=EXPORT_DIR)
    print(f"[FP4] export done in {time.time() - t0:.1f}s")

    # The HF tokenizer/config from the source are needed for vLLM to load it.
    tokenizer.save_pretrained(EXPORT_DIR)
    print(f"[FP4] ALL DONE in {time.time() - t_start:.1f}s. Checkpoint at "
          f"{EXPORT_DIR}")


if __name__ == "__main__":
    main()
