#!/usr/bin/env python3
"""Gap probe: decode-only speedup at (b8, 14k ctx) measured in the
DRIVER (LLM API), to compare against the harness's decode-only 1.81x at
the same nominal cell. T_decode = T(max_tokens=1+N) - T(max_tokens=1)
on identical prompts (prefill cancels; N=160 matches W7_OUTLEN).
"""
import argparse
import glob
import gzip
import json
import time
from pathlib import Path

CKPT = str(Path.home() / "ckpts/Qwen3-8B-W4A16-INT4")


def load_docs(tok, n=8, target_tok=14000):
    files = glob.glob(
        "/data/smcho/huggingface/hub/datasets--allenai--c4/**/*.json.gz",
        recursive=True)
    docs, buf = [], ""
    with gzip.open(files[0], "rt") as f:
        for line in f:
            buf += json.loads(line)["text"] + "\n\n"
            ids = tok(buf).input_ids
            if len(ids) >= target_tok:
                docs.append(tok.decode(ids[:target_tok]))
                buf = ""
                if len(docs) >= n:
                    return docs
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["off", "k4"])
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    spec = None
    if a.arm == "k4":
        spec = {"method": "draft_model", "model": CKPT,
                "num_speculative_tokens": 4,
                "draft_tensor_parallel_size": 1}
    import os
    kw = {}
    if os.environ.get("GAP_PARITY"):
        kw = dict(max_num_batched_tokens=8192, max_num_seqs=8)
    if os.environ.get("GAP_MNS"):
        kw["max_num_seqs"] = int(os.environ["GAP_MNS"])
    if os.environ.get("GAP_MNB"):
        kw["max_num_batched_tokens"] = int(os.environ["GAP_MNB"])
    if os.environ.get("GAP_MAXCG"):
        kw["compilation_config"] = {
            "max_cudagraph_capture_size": int(os.environ["GAP_MAXCG"])}
    llm = LLM(model="Qwen/Qwen3-8B", speculative_config=spec,
              max_model_len=20480, gpu_memory_utilization=0.90,
              max_num_seqs=kw.pop("max_num_seqs", 32),
              enable_prefix_caching=False,
              disable_log_stats=False, async_scheduling=True, **kw)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    docs = load_docs(tok)
    sp1 = SamplingParams(max_tokens=1, ignore_eos=True, temperature=0)
    spN = SamplingParams(max_tokens=161, ignore_eos=True, temperature=0)
    # warm both shapes
    llm.generate(docs, sp1, use_tqdm=False)
    llm.generate(docs, spN, use_tqdm=False)
    t1s, tNs = [], []
    for _ in range(3):
        t0 = time.perf_counter()
        llm.generate(docs, sp1, use_tqdm=False)
        t1s.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        llm.generate(docs, spN, use_tqdm=False)
        tNs.append(time.perf_counter() - t0)
    t1, tN = min(t1s), min(tNs)
    dec = tN - t1
    ntok = 8 * 160
    acc = None
    for m in llm.get_metrics():
        if m.name == "vllm:spec_decode_num_accepted_tokens":
            acc = m.value
    print(f"[gap] arm={a.arm} prefill+1={t1:.2f}s decode160={dec:.2f}s "
          f"decode_toks={ntok/dec:.1f}", flush=True)


if __name__ == "__main__":
    main()
