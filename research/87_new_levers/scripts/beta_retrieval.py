#!/usr/bin/env python3
"""R3: window-stress beta on a true needle bank (fact planted ~2k tokens
into a 16k prompt, question at the end -- the fact is OUTSIDE any 512
window). Arms: win512, win128, kvq_fp8 (prediction: window collapses on
answer positions, kvq holds). Refs built here (bespoke construction),
scored with 77's score_arm."""
import csv
import random
import sys
from pathlib import Path

import torch

PHASE = Path(__file__).resolve().parents[1]
P77 = PHASE.parent / "77_acceptance_map"
sys.path.insert(0, str(P77 / "scripts"))
import score_accept as SA  # noqa: E402
from anchor_gate import FILLER  # noqa: E402

SA.MODELS["q3_8b"] = dict(hf="Qwen/Qwen3-8B", trust=False, moe=False, layers=36)
CFG = SA.MODELS["q3_8b"]
REFDIR = PHASE / "data/refs/q3_8b_retrieval_c16384"
OUT = PHASE / "data/beta_87.csv"
CTX = 16384
random.seed(86)

FACTS = [
    ("the harbor master's ledger", "the ship 'Meridian Star' carried {} crates of saffron"),
    ("the observatory's log", "the comet was recorded at {} degrees above the horizon"),
    ("the merchant's contract", "the agreed price was {} silver coins per bale"),
    ("the census scroll", "the village of Thornfield counted {} households"),
]


def build_needle_prompts(tok, n=12):
    corpus = tok(FILLER, add_special_tokens=False).input_ids
    while len(corpus) < 2 * CTX + 64:
        corpus = corpus + corpus
    out = []
    for i in range(n):
        key, tmpl = FACTS[i % len(FACTS)]
        val = random.randint(137, 977)
        fact = f" According to {key}, {tmpl.format(val)}. "
        q = (f"\n\nNow answer this question. Based on the document above, "
             f"state exactly what {key} recorded, including the precise "
             f"number.")
        fact_ids = tok(fact, add_special_tokens=False).input_ids
        q_ids = tok(q, add_special_tokens=False).input_ids
        # needle at ~depth 2k tokens from the START; tail filler >> window
        head = 2048
        tail = CTX - head - len(fact_ids) - len(q_ids)
        off = (i * 997) % (len(corpus) - (head + tail))
        text = (tok.decode(corpus[off:off + head]) + fact
                + tok.decode(corpus[off + head:off + head + tail]) + q)
        out.append(tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, tokenize=False))
    return out


class Args:
    model, ctx, prompts, gen, chunk = "q3_8b", CTX, 12, 96, 4096


def append_row(row, tag):
    row = dict(row)
    row["arm"] = f"ret_{row['arm']}"
    exists = OUT.exists()
    with OUT.open("a") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(f"[87-ret] {row['arm']}: beta={row['beta_greedy']}", flush=True)


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CFG["hf"])
    args = Args()
    model = SA.load_model(CFG)
    # bespoke stage A
    import anchor_gate
    orig = anchor_gate.build_prompts
    try:
        SA.build_prompts = lambda t, c, n: build_needle_prompts(t, n)
        SA.stage_a(model, tok, CFG, args, REFDIR)
    finally:
        SA.build_prompts = orig
    done = set()
    if OUT.exists():
        done = {r["arm"] for r in csv.DictReader(OUT.open())}
    for arm in ("win512", "win128", "kvq_fp8"):
        if f"ret_{arm}" in done:
            continue
        append_row(SA.score_arm(model, CFG, arm, REFDIR, args), arm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
