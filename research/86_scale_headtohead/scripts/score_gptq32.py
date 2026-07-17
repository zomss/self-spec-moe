#!/usr/bin/env python3
"""E3 scoring: beta of the GPTQ-calibrated W4A16 ckpt on 77's dense refs
(vs RTN fake-quant 0.924 and the deployed data-free RTN artifact).

Loads the compressed-tensors ckpt DECOMPRESSED (run_compressed=False) so
the forward is the bf16-equivalent of the calibrated int4 weights --
matching the fake-quant semantics of the 77 harness. Writes
data/beta_gptq.csv (arm q_int4_gptq); map_v5.py picks it up.
"""

import csv
import sys
from pathlib import Path

import torch

PHASE = Path(__file__).resolve().parents[1]
P77 = PHASE.parent / "77_acceptance_map"
sys.path.insert(0, str(P77 / "scripts"))
import score_accept as SA  # noqa: E402

CKPT = Path.home() / "ckpts/Qwen3-32B-W4A16-INT4-gptq"
REFDIR = Path(__file__).resolve().parents[1] / "data/refs/q3_32b_c16384"
OUT = PHASE / "data/beta_gptq32.csv"


class Args:
    model, ctx, prompts, gen, chunk = "q3_32b", 16384, 12, 96, 4096


def main() -> int:
    from transformers import AutoModelForCausalLM
    from transformers.utils.quantization_config import CompressedTensorsConfig

    model = AutoModelForCausalLM.from_pretrained(
        CKPT, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa",
        quantization_config=CompressedTensorsConfig(run_compressed=False),
    )
    model.eval()
    SA.MODELS["q3_32b"] = dict(hf="Qwen/Qwen3-32B", trust=False, moe=False, layers=64)
    cfg = SA.MODELS["q3_32b"]
    row = SA.score_arm(model, cfg, "q_int4_gptq", REFDIR, Args())
    with OUT.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    print("beta_gptq:", row["beta_greedy"], "(RTN fake-quant: 0.924)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
