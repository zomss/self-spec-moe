"""R1: SparseGPT one-shot 2:4 of Qwen3-8B (the FAIR pruning form).
512 C4 samples; sparsity-only (no quant) so the beta gate isolates the
pruning factor; the int4 combo is scored by stacking RTN fake-quant on
the decompressed sparse model."""
import os, glob, gzip, json
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.pruning import SparseGPTModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

SRC = "Qwen/Qwen3-8B"
OUT = os.path.expanduser("~/ckpts/Qwen3-8B-sparse24-sgpt")
model = AutoModelForCausalLM.from_pretrained(SRC, dtype="bfloat16", device_map="cuda")
tok = AutoTokenizer.from_pretrained(SRC)
files = glob.glob(os.path.expanduser(
    "/data/smcho/huggingface/hub/datasets--allenai--c4/**/*.json.gz"), recursive=True) or \
    glob.glob("/data/smcho/huggingface/datasets/downloads/*.gz")
rows = []
with gzip.open(files[0], "rt") as f:
    for i, line in enumerate(f):
        if i >= 512: break
        rows.append({"text": json.loads(line)["text"]})
ds = Dataset.from_list(rows).map(
    lambda ex: tok(ex["text"], truncation=True, max_length=2048),
    remove_columns=["text"])
recipe = SparseGPTModifier(sparsity=0.5, mask_structure="2:4",
                           targets=["Linear"], ignore=["lm_head"])
oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=2048,
        num_calibration_samples=512, output_dir=OUT)
tok.save_pretrained(OUT)
print("saved:", OUT)
