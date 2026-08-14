#!/usr/bin/env python3
"""Correctness gate for E2: greedy self-spec output must be TOKEN-IDENTICAL to no-spec.

Speculative decoding is lossless by rejection sampling, so at temperature 0 the spec and
no-spec token streams must match exactly. A mismatch means a plumbing bug, not a method
property -- most likely the draft corrupting the target's KV pool under
VLLM_SELF_SPEC_SHARED_KV=1 (the draft aliases the target's KV tensors, and a
weight-quantized draft writes DIFFERENT K/V for speculative positions).

Run once per mode (separate processes; a 7B target + 7B draft in one process is wasteful
and the second engine can inherit the first's allocator state):

  python check_lossless.py --mode nospec --out data/lossless_nospec.json
  python check_lossless.py --mode spec --draft /data/smcho/ckpts/... --out data/lossless_spec.json
  python check_lossless.py --compare data/lossless_nospec.json data/lossless_spec.json
"""

import argparse
import json
import os
import sys


def compare(a_path, b_path) -> int:
    a, b = json.load(open(a_path)), json.load(open(b_path))
    if a["prompts"] != b["prompts"]:
        print("FAIL: different prompt sets; not comparable")
        return 2
    bad = [i for i, (x, y) in enumerate(zip(a["tokens"], b["tokens"])) if x != y]
    if not bad:
        n = sum(len(t) for t in a["tokens"])
        print(f"PASS: {len(a['tokens'])} sequences, {n} tokens, token-identical")
        return 0
    print(f"FAIL: {len(bad)}/{len(a['tokens'])} sequences differ (indices {bad[:5]})")
    i = bad[0]
    x, y = a["tokens"][i], b["tokens"][i]
    j = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), min(len(x), len(y)))
    print(f"  seq {i}: first divergence at token {j}")
    print(f"    nospec {x[max(0,j-3):j+3]}")
    print(f"    spec   {y[max(0,j-3):j+3]}")
    print("  => losslessness violated. Suspect VLLM_SELF_SPEC_SHARED_KV=1 with a")
    print("     weight-quantized draft. Re-run with SHARED_KV=0 before trusting any tau.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs=2, metavar=("NOSPEC", "SPEC"))
    ap.add_argument("--mode", choices=("nospec", "spec"))
    ap.add_argument("--model", default=os.environ.get("E75_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    ap.add_argument("--draft")
    ap.add_argument("--gamma", type=int, default=5)
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.compare:
        return compare(*a.compare)
    if not a.mode or not a.out:
        ap.error("--mode and --out required unless --compare")
    if a.mode == "spec" and not a.draft:
        ap.error("--draft required for --mode spec")

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pf = a.prompts or os.path.join(here, "prompts", "math_prompts.txt")
    prompts = [ln.strip() for ln in open(pf) if ln.strip()][: a.n_prompts]

    from vllm import LLM, SamplingParams

    kw = dict(model=a.model, tensor_parallel_size=1, max_model_len=4096,
              gpu_memory_utilization=0.85, enforce_eager=False)
    if a.mode == "spec":
        kw["speculative_config"] = {
            "method": "draft_model",
            "model": a.draft,
            "num_speculative_tokens": a.gamma,
            "draft_tensor_parallel_size": 1,
        }
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    chat = [tok.apply_chat_template([{"role": "user", "content": p}],
                                    add_generation_prompt=True, tokenize=False)
            for p in prompts]
    # temperature 0 -> the two streams must agree exactly.
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, seed=0)
    outs = llm.generate(chat, sp, use_tqdm=False)
    payload = {
        "mode": a.mode, "model": a.model, "draft": a.draft, "gamma": a.gamma,
        "prompts": prompts,
        "tokens": [list(o.outputs[0].token_ids) for o in outs],
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(payload, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}  ({len(outs)} sequences)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
