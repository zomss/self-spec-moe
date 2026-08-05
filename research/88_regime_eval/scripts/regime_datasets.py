#!/usr/bin/env python3
"""Canonical regime -> dataset loaders (Phase 88 E0).

One entry point: load_regime(rid, tok, n, ...) -> list[str] prompts +
a gen spec dict. Datasets follow prior work (Spec-Bench, KnapSpec,
EfficientRollout); see README for the matrix. All HF downloads land in
the /data cache (HF_HOME already points there).
"""
import glob
import os

MAXTOK_DEFAULT = 512

CHAT_MODELS = ("Qwen",)  # prompts get chat template when tok has one


def _chat(tok, user_msg, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) \
        + [{"role": "user", "content": user_msg}]
    try:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)


def _c4_docs(tok, n, target_tok, skip=0):
    """Real web documents packed to target_tok (kept for RAG context).

    ``skip`` drops the first ``skip`` PACKED documents so different seeds
    draw disjoint content; skip=0 is byte-identical to the historical
    loader.
    """
    docs, buf = [], ""
    import gzip
    import json
    for path in sorted(glob.glob(
            "/data/smcho/huggingface/hub/datasets--allenai--c4/**/*.json.gz",
            recursive=True)):
        with gzip.open(path, "rt") as f:
            for line in f:
                buf += json.loads(line)["text"] + "\n\n"
                ids = tok.encode(buf)
                if len(ids) >= target_tok:
                    docs.append(tok.decode(ids[:target_tok]))
                    buf = ""
                    if len(docs) >= skip + n:
                        return docs[skip:]
    return docs[skip:]


def load_regime(rid, tok, n=8, ctx_target=None, seed=0):
    """Returns (prompts, spec) where spec = {max_tokens, temperature,
    batch} recommended for the regime.

    ``seed`` selects the CONTENT DRAW: examples are taken at offset
    ``seed * n`` into each source dataset (mod its length), so different
    seeds carry disjoint prompt sets. seed=0 is byte-identical to the
    historical loader (phases 88-95, which ignored ``seed`` entirely --
    their "seeds" were replicate boots; 96/G6).
    """
    from datasets import load_dataset

    off = seed * n

    if rid == "R1":  # interactive math CoT -- GSM8K + AIME (KnapSpec class)
        gsm = load_dataset("openai/gsm8k", "main", split="test")
        aime = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")
        qs = [gsm[(off + i) % len(gsm)]["question"]
              for i in range(n // 2)] + \
             [aime[(off + i) % len(aime)]["Question"]
              for i in range(n - n // 2)]
        prompts = [_chat(tok, q + "\nPlease reason step by step.")
                   for q in qs]
        return prompts, dict(max_tokens=1024, temperature=0.0, batch=1)

    if rid == "R2":  # conversation -- MT-Bench first turns
        ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        prompts = [_chat(tok, ds[(off + i) % len(ds)]["prompt"][0])
                   for i in range(n)]
        return prompts, dict(max_tokens=MAXTOK_DEFAULT, temperature=0.0,
                             batch=1)

    if rid == "R3":  # code -- HumanEval
        ds = load_dataset("openai/openai_humaneval", split="test")
        prompts = [_chat(tok, "Complete this Python function. Return "
                              "the full function.\n\n"
                              + ds[(off + i) % len(ds)]["prompt"])
                   for i in range(n)]
        return prompts, dict(max_tokens=MAXTOK_DEFAULT, temperature=0.0,
                             batch=8)

    if rid == "R4":  # summarization of LONG docs -- CNN/DM, packed long
        ds = load_dataset("abisee/cnn_dailymail", "3.0.0",
                          split="test[:2000]")
        target = ctx_target or 8000
        prompts, buf, used = [], [], 0
        for ex in ds:
            buf.append(ex["article"])
            used = len(tok.encode("\n\n---\n\n".join(buf)))
            if used >= target - 200:
                body = "\n\n---\n\n".join(buf)
                prompts.append(_chat(
                    tok, "Summarize each of the following news "
                         "articles in 2-3 sentences each.\n\n" + body))
                buf, used = [], 0
                if len(prompts) >= off + n:   # skip first off PACKED prompts
                    break
        return prompts[off:], dict(max_tokens=MAXTOK_DEFAULT,
                                   temperature=0.0, batch=8)

    if rid == "R5":  # RAG QA -- NQ-open questions over real doc context
        nq = load_dataset("google-research-datasets/nq_open",
                          split="validation")
        target = ctx_target or 14000
        docs = _c4_docs(tok, n, target, skip=off)
        prompts = []
        for i in range(n):
            prompts.append(_chat(
                tok, docs[i] + "\n\nUsing the context above where "
                "relevant plus your knowledge, answer and explain: "
                + nq[(off + i) % len(nq)]["question"]))
        return prompts, dict(max_tokens=MAXTOK_DEFAULT, temperature=0.0,
                             batch=8)

    if rid == "R5cot":  # long-CoT RAG variant (82's S-trace, kept)
        aime = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")
        target = ctx_target or 14000
        docs = _c4_docs(tok, n, target, skip=off)
        prompts = [_chat(tok, docs[i] + "\n\nNow solve this problem "
                         "step by step.\n\n"
                         + aime[(off + (i % 8)) % len(aime)]["Question"])
                   for i in range(n)]
        return prompts, dict(max_tokens=3072, temperature=0.0, batch=8)

    if rid == "R6":  # high-batch burst -- GSM8K short answers, b32/2k
        gsm = load_dataset("openai/gsm8k", "main", split="test")
        prompts = [_chat(tok, gsm[(off + i) % len(gsm)]["question"] +
                         "\nAnswer concisely.") for i in range(n)]
        return prompts, dict(max_tokens=256, temperature=0.0, batch=32)

    if rid == "R7":  # translation -- WMT14 de-en
        ds = load_dataset("wmt/wmt14", "de-en", split="test[:200]")
        prompts = [_chat(tok, "Translate this German text to English:"
                              "\n\n"
                              + ds[(off + i) % len(ds)]["translation"]["de"])
                   for i in range(n)]
        return prompts, dict(max_tokens=MAXTOK_DEFAULT, temperature=0.0,
                             batch=8)

    if rid == "R8":  # RL rollout -- MATH prompts, T=1.0 (EfficientRollout)
        ds = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")
        prompts = [_chat(tok, ds[(off + i) % len(ds)]["Question"] +
                         "\nPlease reason step by step.")
                   for i in range(n)]
        return prompts, dict(max_tokens=2048, temperature=1.0, batch=16)

    if rid == "RKS":  # KnapSpec parity cell: b1 x 16k-ctx prose, 160 out
        lines = [ln.strip() for ln in open(
            "research/57_large_ep_spec_strategy/data/prompts_ondist.txt")
            if ln.strip()]
        target = ctx_target or 16000
        docs = _c4_docs(tok, n, target, skip=off)
        prompts = [_chat(tok, docs[i] + "\n\nNow, a separate task:\n"
                         + lines[(off + i) % len(lines)])
                   for i in range(n)]
        return prompts, dict(max_tokens=160, temperature=0.0, batch=1)

    raise ValueError(f"unknown regime {rid}")


REGIMES = ["R1", "R2", "R3", "R4", "R5", "R5cot", "R6", "R7", "R8"]


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        os.environ.get("MODEL", "Qwen/Qwen3-8B"))
    only = os.environ.get("REGIMES")
    for rid in (only.split(",") if only else REGIMES):
        try:
            prompts, spec = load_regime(rid, tok, n=4)
            lens = [len(tok.encode(p)) for p in prompts]
            print(f"[{rid}] n={len(prompts)} prompt_tok={lens} "
                  f"spec={spec}")
        except Exception as e:
            print(f"[{rid}] FAIL: {type(e).__name__}: {e}")
