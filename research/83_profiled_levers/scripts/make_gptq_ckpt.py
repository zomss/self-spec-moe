"""E3: GPTQ-calibrated W4A16-INT4 of Qwen2.5-7B-Instruct (vs the deployed
DATA-FREE RTN ckpt). Same scheme as 74's make_w4a16_int4_ckpt.py (int4 sym
g128, weight-only) but with GPTQ + 512 calibration samples. Run in the
isolated eff_lc_venv on ONE GPU."""

import os

from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

SRC = "Qwen/Qwen2.5-7B-Instruct"
OUT = os.path.expanduser("/data/smcho/ckpts/Qwen2.5-7B-Instruct-W4A16-INT4-gptq")

model = AutoModelForCausalLM.from_pretrained(SRC, dtype="bfloat16",
                                             device_map="cuda")
tok = AutoTokenizer.from_pretrained(SRC)

ds = load_dataset("allenai/c4", split="train",
                  data_files={"train": "en/c4-train.00000-of-01024.json.gz"})
NUM, MAXLEN = 512, 2048


def preprocess(ex):
    return tok(ex["text"], truncation=True, max_length=MAXLEN)


ds = ds.select(range(NUM)).map(preprocess, remove_columns=ds.column_names)

recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
oneshot(model=model, dataset=ds, recipe=recipe,
        max_seq_length=MAXLEN, num_calibration_samples=NUM,
        output_dir=OUT)
tok.save_pretrained(OUT)
print("saved:", OUT)
