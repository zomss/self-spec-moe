#!/usr/bin/env python3
"""Final-eval cell loaders (Phase 100, `final_eval_design.md`).

One entry point: load_cell(cell, tok, n, seed=0) -> (prompts, spec) with
spec = {max_tokens, temperature, batches, thinking}. Cells:

* LI  -- GovReport test, token band [12K, 20K], summarize        (thinking off)
* LO  -- AIME 2024 + 2025, step-by-step, boxed answer            (thinking ON)
* LIO -- BookSum test AGGREGATE (multi-chapter) rows, band
         [8K, 20K], detailed summary                             (thinking off)
* SS  -- GSM8K test, "answer concisely"                          (thinking off)

Every source is pinned to an explicit revision commit. GovReport and
BookSum stream (big corpora; order under a fixed revision is stable);
AIME and GSM8K load normally (tiny). ``seed`` offsets by seed*n MATCHING
examples so different seeds draw disjoint content, as in the phase-98
loader. max_tokens values are the safety caps of `final_eval_design.md`,
not budgets -- scored runs respect EOS.

LO uses Qwen3 thinking mode: that is where natural long CoT comes from
and it matches KnapSpec's Qwen3 reasoning setup (32K generation budget).
The scored campaign must replace streaming with snapshot download plus a
byte-hash manifest (registration checklist); this loader is for the
pilot and for loader-level pinning.
"""
from __future__ import annotations

PINS = {
    "govreport": ("ccdv/govreport-summarization",
                  "4e21184e01ae8017e2c036e180fe5e541fef60a0"),
    "booksum": ("kmfoda/booksum",
                "c62321036e5647db5767ecaff139912b554dc938"),
    "aime24": ("HuggingFaceH4/aime_2024",
               "2fe88a2f1091d5048c0f36abc874fb997b3dd99a"),
    "aime25": ("math-ai/aime25",
               "563bb8404243c5f09de6ec262f2db674fe5bce9b"),
    "gsm8k": ("openai/gsm8k",
              "740312add88f781978c0658806c59bc2815b9866"),
}

# Token bands (Qwen3 tokenizer) for the long-input cells.
LI_BAND = (12_000, 20_000)
LIO_BAND = (8_000, 20_000)
# Chars-per-token prefilter bounds so streaming does not tokenize every
# row: Qwen3 on English prose runs ~3.5-4.5 chars/token.
_PREFILTER = (3.0, 6.0)

CELLS = ("LI", "LO", "LIO", "SS")


def _chat(tok, user_msg: str, thinking: bool) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False, add_generation_prompt=True,
        enable_thinking=thinking)


def _stream_banded(stream, text_of, band, tok, n, skip):
    """First n rows (after skipping `skip` matches) whose text token
    count lies in `band`, in stream order."""
    lo, hi = band
    out, matched = [], 0
    for row in stream:
        text = text_of(row)
        if not text:
            continue
        c = len(text)
        if c < lo * _PREFILTER[0] or c > hi * _PREFILTER[1]:
            continue
        t = len(tok.encode(text))
        if not lo <= t <= hi:
            continue
        matched += 1
        if matched <= skip:
            continue
        out.append(row)
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(
            f"band {band} exhausted: wanted {n}, found {len(out)} "
            f"(after skipping {skip})")
    return out


def load_cell(cell, tok, n=16, seed=0):
    from datasets import load_dataset

    skip = seed * n

    if cell == "LI":
        rid, rev = PINS["govreport"]
        stream = load_dataset(rid, revision=rev, streaming=True)["test"]
        rows = _stream_banded(stream, lambda r: r["report"],
                              LI_BAND, tok, n, skip)
        prompts = [_chat(tok, "Summarize the following government "
                              "report.\n\n" + r["report"], thinking=False)
                   for r in rows]
        return prompts, dict(max_tokens=2048, temperature=0.0,
                             batches=[1, 8, 16], thinking=False)

    if cell == "LO":
        rid24, rev24 = PINS["aime24"]
        rid25, rev25 = PINS["aime25"]
        a24 = load_dataset(rid24, revision=rev24, split="train")
        a25 = load_dataset(rid25, revision=rev25, split="test")
        probs = [a24[i % len(a24)]["problem"] for i in range(len(a24))] + \
                [a25[i % len(a25)]["problem"] for i in range(len(a25))]
        qs = [probs[(skip + i) % len(probs)] for i in range(n)]
        prompts = [_chat(tok, q + "\nPlease reason step by step, and put "
                              "your final answer within \\boxed{}.",
                         thinking=True)
                   for q in qs]
        return prompts, dict(max_tokens=32_768, temperature=0.0,
                             batches=[1, 8, 16], thinking=True)

    if cell == "LIO":
        rid, rev = PINS["booksum"]
        stream = load_dataset(rid, revision=rev, streaming=True)["test"]
        rows = _stream_banded(
            stream,
            lambda r: r["chapter"] if r["is_aggregate"] else None,
            LIO_BAND, tok, n, skip)
        prompts = [_chat(tok, "Summarize the following book excerpt in "
                              "detail, covering all major plot points "
                              "and characters.\n\n" + r["chapter"],
                         thinking=False)
                   for r in rows]
        return prompts, dict(max_tokens=4096, temperature=0.0,
                             batches=[1, 8, 16], thinking=False)

    if cell == "SS":
        rid, rev = PINS["gsm8k"]
        ds = load_dataset(rid, "main", revision=rev, split="test")
        prompts = [_chat(tok, ds[(skip + i) % len(ds)]["question"] +
                         "\nAnswer concisely.", thinking=False)
                   for i in range(n)]
        return prompts, dict(max_tokens=1024, temperature=0.0,
                             batches=[1, 8, 32, 64], thinking=False)

    raise ValueError(f"unknown cell {cell!r}")


if __name__ == "__main__":
    import os

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        os.environ.get("MODEL", "Qwen/Qwen3-8B"))
    n = int(os.environ.get("N", "2"))
    for cell in (os.environ.get("CELLS") or ",".join(CELLS)).split(","):
        prompts, spec = load_cell(cell, tok, n=n)
        lens = [len(tok.encode(p)) for p in prompts]
        print(f"[{cell}] n={len(prompts)} prompt_tok={lens} spec={spec}")
