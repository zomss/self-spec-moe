"""Queue item (map v6.3 standing): W4A8 ckpt of Qwen3-32B (GPTQ int4 weights + dynamic 8-bit acts).
Deploys via vLLM compressed_tensors_w4a8 schemes (CutlassW4A8 kernel)."""
import os, glob, gzip, json
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

SRC = "Qwen/Qwen3-32B"
OUT = os.path.expanduser("/data/smcho/ckpts/Qwen3-32B-W4A8-gptq")
model = AutoModelForCausalLM.from_pretrained(SRC, dtype="bfloat16", device_map="cuda")
tok = AutoTokenizer.from_pretrained(SRC)
files = glob.glob("/data/smcho/huggingface/hub/datasets--allenai--c4/**/*.json.gz",
                  recursive=True) or glob.glob(
    "/data/smcho/huggingface/datasets/downloads/*.gz")
rows = []
with gzip.open(files[0], "rt") as f:
    for i, line in enumerate(f):
        if i >= 512: break
        rows.append({"text": json.loads(line)["text"]})
ds = Dataset.from_list(rows).map(
    lambda ex: tok(ex["text"], truncation=True, max_length=2048),
    remove_columns=["text"])
recipe = GPTQModifier(targets="Linear", scheme="W4A8", ignore=["lm_head"])
oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=2048,
        num_calibration_samples=512)
# vLLM's CompressedTensorsW4A8Int expects PACK-QUANTIZED (weight_packed);
# llm-compressor infers int-quantized for this scheme -- force the format.
model.save_pretrained(OUT, save_compressed=True,
                      quantization_format="pack-quantized")
tok.save_pretrained(OUT)
print("saved:", OUT)
