#!/usr/bin/env python3
"""Policy compiler: measure the map ON the deployment path, solve, emit.

The search-vs-trace gap (E2b/E2c) came from importing cell verdicts
measured on a different stack/accounting. This compiler closes it:

  --measure ARM   one engine per arm (off/k4/k6), decode-only time per
                  (batch, ctx) cell via T(1+N) - T(1) on identical
                  prompts (prefill cancels). Appends data/policy_cells.csv.
  --solve         per cell: R_K = (tau_ref/S_dec - 1)/K from the paired
                  measurements; K* = argmax_K tau(f_ref,K)/(K R_K + 1);
                  OFF if best S_dec < 1 + margin; gate threshold =
                  break-even acceptance frac* solving tau(f*,K*) =
                  K* R_K* + 1. Emits data/policy_table.json.

No hand constants: K set from the measured arms, margin = measured
arming rent (1.5%), thresholds from the cost model per cell.
"""
import argparse
import csv
import glob
import gzip
import json
import time
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
DATA = PHASE / "data"
CKPT = str(Path.home() / "ckpts/Qwen3-8B-W4A16-INT4")
BATCHES = [1, 8, 16, 32]
CTXS = [2000, 8000, 14000]
NDEC = 160
ARM_K = {"off": 0, "k4": 4, "k6": 6}
ARMING_RENT = 0.015


def load_ctx_prompts(tok, n, target_tok):
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


def spec_counters(llm):
    acc = dr = 0
    for m in llm.get_metrics():
        if m.name == "vllm:spec_decode_num_accepted_tokens":
            acc = m.value
        elif m.name == "vllm:spec_decode_num_drafts":
            dr = m.value
    return acc, dr


def measure(arm):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    k = ARM_K[arm]
    spec = None
    if k:
        spec = {"method": "draft_model", "model": CKPT,
                "num_speculative_tokens": k,
                "draft_tensor_parallel_size": 1}
    llm = LLM(model="Qwen/Qwen3-8B", speculative_config=spec,
              max_model_len=20480, gpu_memory_utilization=0.90,
              max_num_seqs=max(BATCHES), enable_prefix_caching=False,
              disable_log_stats=False, async_scheduling=True,
              max_num_batched_tokens=8192)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    prompts_by_ctx = {c: load_ctx_prompts(tok, max(BATCHES), c)
                      for c in CTXS}
    sp1 = SamplingParams(max_tokens=1, ignore_eos=True, temperature=0)
    spN = SamplingParams(max_tokens=1 + NDEC, ignore_eos=True,
                        temperature=0)
    rows = []
    KV_LIMIT_TOKENS = 330_000  # ~50 GB pool / 147 KB per token
    for ctx in CTXS:
        for b in BATCHES:
            if b * ctx > KV_LIMIT_TOKENS:
                print(f"[cell] SKIP infeasible b{b}/ctx{ctx} (residency)")
                continue
            ps = prompts_by_ctx[ctx][:b]
            llm.generate(ps, spN, use_tqdm=False)  # warm shapes
            t1s, tns, accs = [], [], []
            for _ in range(3):
                t0 = time.perf_counter()
                llm.generate(ps, sp1, use_tqdm=False)
                t1s.append(time.perf_counter() - t0)
                a0, d0 = spec_counters(llm)
                t0 = time.perf_counter()
                llm.generate(ps, spN, use_tqdm=False)
                tns.append(time.perf_counter() - t0)
                a1, d1 = spec_counters(llm)
                if d1 > d0:
                    accs.append(1 + (a1 - a0) / (d1 - d0))
            dec = min(tns) - min(t1s)
            toks = b * NDEC / dec
            acc = round(sum(accs) / len(accs), 3) if accs else 0.0
            row = dict(arm=arm, K=k, batch=b, ctx=ctx,
                       decode_toks=round(toks, 1), accept=acc)
            rows.append(row)
            print("[cell]", json.dumps(row), flush=True)
    DATA.mkdir(exist_ok=True)
    out = DATA / "policy_cells.csv"
    exists = out.exists()
    with out.open("a") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if not exists:
            w.writeheader()
        w.writerows(rows)
    print("appended ->", out)


def solve():
    """v2: emit per-cell (K, R) OPTIONS. The scheduler picks K online by
    argmax S_K(f) = (1 + f*K)/(K*R_K + 1) with f = live accept-fraction
    EMA (identity: tau = 1 + f*K, so this is exact, unit-consistent, and
    content enters ONLY through the live signal -- R is a pure time
    ratio measured here). K=0 (S=1) is always an option."""
    cells = {}
    for row in csv.DictReader((DATA / "policy_cells.csv").open()):
        key = (int(row["batch"]), int(row["ctx"]))
        cells.setdefault(key, {})[row["arm"]] = row
    table = []
    for (b, ctx), arms in sorted(cells.items()):
        if "off" not in arms:
            continue
        base = float(arms["off"]["decode_toks"])
        options = []
        for arm, row in arms.items():
            k = int(row["K"])
            if k == 0:
                continue
            s_dec = float(row["decode_toks"]) / base
            t_meas = float(row["accept"])
            if s_dec <= 0 or t_meas <= 1:
                continue
            r = (t_meas / s_dec - 1) / k
            options.append(dict(K=k, R=round(r, 3),
                                S_ref=round(s_dec, 3),
                                f_ref=round((t_meas - 1) / k, 3)))
        entry = dict(batch=b, ctx=ctx, options=options)
        table.append(entry)
        print("[solve]", json.dumps(entry), flush=True)
    out = DATA / "policy_table.json"
    out.write_text(json.dumps(
        dict(model="Qwen3-8B", draft=CKPT, ndec=NDEC,
             arming_rent=ARMING_RENT, cells=table), indent=1))
    print("emitted ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", choices=list(ARM_K))
    ap.add_argument("--solve", action="store_true")
    a = ap.parse_args()
    if a.measure:
        measure(a.measure)
    if a.solve:
        solve()
