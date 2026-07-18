#!/usr/bin/env python3
"""E2: switching demo on a regime-shifting REAL-data trace.

Trace (identical for every arm):
  P1 interactive: b1  x AIME problems (real math dataset), 256 new tok,
                  8 rounds
  P2 serving:     b8  x C4 web documents (~6k tok prompts), 256 new tok,
                  2 rounds
  P3 burst:       b32 x AIME problems, 256 new tok, 2 rounds

Arms (one engine each, same trace):
  off     no speculation (the AR baseline)
  k4      W4 draft + win512, static K4
  k6      W4 draft + win512, static K6
  policy  W4 draft + win512, the strategy map compiled into runtime
          primitives: per-batch-size K schedule [(1,12,4),(13,24,6),
          (25,32,0)] (b>=25 -> OFF: the verify-width regime) + the
          accept-feedback OFF gate (VLLM_SELF_SPEC_ACCEPT_OFF_THRESHOLD
          =0.8, content axis: off-distribution text -> OFF with probe
          recovery). Per-step scheduler switching, zero toggle cost (E0).

Gate (82 README): policy >= every static on aggregate tok/s AND within
5% of the per-phase static winner in every phase. Fail-honest.
"""
import argparse
import glob
import gzip
import json
import time
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
CKPT = str(Path.home() / "ckpts/Qwen3-8B-W4A16-INT4")
SCHEDULE = [(1, 12, 4), (13, 24, 6), (25, 32, 0)]


def load_aime(n=40):
    from datasets import load_dataset
    ds = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")
    probs = [r["Question"] for r in ds.select(range(n))]
    return ["Solve this competition math problem step by step.\n\n" + p
            for p in probs]


def load_docs(tok, n=8, target_tok=6000):
    files = glob.glob(
        "/data/smcho/huggingface/hub/datasets--allenai--c4/**/*.json.gz",
        recursive=True)
    docs, buf = [], ""
    with gzip.open(files[0], "rt") as f:
        for line in f:
            buf += json.loads(line)["text"] + "\n\n"
            ids = tok(buf).input_ids
            if len(ids) >= target_tok:
                docs.append(tok.decode(ids[:target_tok])
                            + "\n\nSummarize the text above in detail.")
                buf = ""
                if len(docs) >= n:
                    return docs
    return docs


def spec_counters(llm):
    acc = drafts = 0
    for m in llm.get_metrics():
        if m.name == "vllm:spec_decode_num_accepted_tokens":
            acc = m.value
        elif m.name == "vllm:spec_decode_num_drafts":
            drafts = m.value
    return acc, drafts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["off", "k4", "k6", "policy"])
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    spec = None
    if a.arm != "off":
        spec = {"method": "draft_model", "model": CKPT,
                "num_speculative_tokens": {"k4": 4}.get(a.arm, 6),
                "draft_tensor_parallel_size": 1}
        if a.arm == "policy":
            spec["num_speculative_tokens_per_batch_size"] = SCHEDULE
    llm = LLM(model="Qwen/Qwen3-8B", speculative_config=spec,
              max_model_len=12288, gpu_memory_utilization=0.90,
              max_num_seqs=32, enable_prefix_caching=False,
              disable_log_stats=False,
              async_scheduling=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    aime = load_aime()
    docs = load_docs(tok)
    sp = SamplingParams(max_tokens=256, ignore_eos=True, temperature=0)

    # warm both shapes
    llm.generate(aime[:1], sp, use_tqdm=False)
    llm.generate(docs[:8], sp, use_tqdm=False)

    # One generate() per serving/burst phase: round-based replay drains
    # the engine between rounds, retriggering regime detection at every
    # round start -- an artifact a continuous workload doesn't have.
    sp512 = SamplingParams(max_tokens=512, ignore_eos=True, temperature=0)
    trace = ([("P1_b1_aime", [aime[i]], sp) for i in range(8)]
             + [("P2_b8_docs", docs[:8], sp512)]
             + [("P3_b32_aime", aime[8:40], sp512)])
    import os
    if os.environ.get("E2_PHASES"):
        keep = os.environ["E2_PHASES"].split(",")
        trace = [t for t in trace if any(t[0].startswith(k) for k in keep)]

    per_phase = {}
    for name, prompts, sparams in trace:
        a0, d0 = spec_counters(llm)
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sparams, use_tqdm=False)
        dt = time.perf_counter() - t0
        a1, d1 = spec_counters(llm)
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        ph = per_phase.setdefault(name, dict(tok=0, s=0.0, acc=0, dr=0))
        ph["tok"] += ntok
        ph["s"] += dt
        ph["acc"] += a1 - a0
        ph["dr"] += d1 - d0
    out = dict(arm=a.arm)
    tot_tok = tot_s = 0
    for name, ph in per_phase.items():
        tot_tok += ph["tok"]
        tot_s += ph["s"]
        out[name] = dict(toks=round(ph["tok"] / ph["s"], 1),
                         accept=round(1 + ph["acc"] / max(1, ph["dr"]), 2)
                         if ph["dr"] else None)
    out["aggregate_toks"] = round(tot_tok / tot_s, 1)
    print("[E2]", json.dumps(out), flush=True)
    p = PHASE / f"data/e2_{a.arm}.json"
    p.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
