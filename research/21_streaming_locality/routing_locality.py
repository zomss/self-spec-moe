#!/usr/bin/env python3
"""Union-over-k reuse and cache-hit curves for PCIe-streamed expert offload.

Phase 21. The PCIe-viable expert-offload draft (see ../20_injection_emulation and
the design discussion) hides DRAM->GPU expert streaming behind compute only if the
*bytes streamed per draft cycle* fall below the hideable budget. Two empirical
unknowns decide that:

  1. union-over-k reuse: across a k-token draft cycle, how many *distinct* experts
     per layer must be resident (streamed once, reused for all k tokens)? High
     temporal locality => the union grows sublinearly in k => few streams.
  2. cache-hit curve: a verify-warmed (or hot) resident cache of size C serves some
     fraction of routing for free; only the misses are streamed.

This script captures per-layer per-position true top-k experts from a real-weight
decode trace (reusing the Phase 10 capture), saves the raw routing, then computes
both curves and the PCIe-budget synthesis: streamed bytes/cycle (after cache+reuse)
vs the hideable budget k*T_step*BW. Real weights are mandatory -- dummy weights
route uniformly and erase exactly the skew/locality this measures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PROMPT_BANK: dict[str, list[str]] = {
    "general": [
        "The future of artificial intelligence is",
        "Summarize why distributed inference needs communication.",
        "List three benefits of expert parallelism.",
        "Describe how a mixture-of-experts layer routes tokens.",
    ],
    "chat": [
        "Hi! Can you recommend a good book for a long flight?",
        "I'm feeling stressed about work. Any advice?",
        "What's a fun fact about the ocean?",
        "Explain mixture of experts models in simple terms.",
    ],
    "code": [
        "Write a Python function to compute the nth Fibonacci number.",
        "Implement binary search over a sorted list in Python.",
        "Explain what a race condition is and how to avoid it.",
        "Write a SQL query to find the second highest salary.",
    ],
    "math": [
        "Solve step by step: if x + 3 = 7, what is x?",
        "What is the derivative of x^2 + 3x with respect to x?",
        "Compute the greatest common divisor of 48 and 36, showing work.",
        "A train travels 60 km in 45 minutes. What is its speed in km/h?",
    ],
}

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def parse_int_list(raw: str) -> list[int]:
    return [int(v) for v in raw.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True, help="short name for output files")
    p.add_argument("--outdir", type=Path, default=Path(__file__).parent)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--prompts-per-bucket", type=int, default=4)
    p.add_argument("--gen-tokens", type=int, default=160)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reuse", action="store_true", help="load saved trace, skip GPU")
    p.add_argument("--k-list", type=parse_int_list, default=[1, 2, 4, 8, 16])
    p.add_argument("--batch-list", type=parse_int_list, default=[1, 2, 4, 8])
    p.add_argument(
        "--cache-list",
        type=parse_int_list,
        default=[0, 8, 16, 24, 32, 48, 64],
        help="resident cache size C (experts/layer)",
    )
    # PCIe-budget parameters
    p.add_argument("--pcie-gbps", type=float, default=50.0, help="effective GB/s")
    p.add_argument("--tstep-ms", type=float, default=9.0, help="draft step compute")
    p.add_argument("--draft-bits", type=float, default=4.0, help="streamed quant bits")
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Capture (GPU): decode real continuations, read per-layer per-position top-k
# --------------------------------------------------------------------------- #
def capture(args, npz_path: Path) -> dict:
    cfg = AutoConfig.from_pretrained(args.model, local_files_only=args.local_files_only)
    model_type = getattr(cfg, "model_type", "")
    num_experts = getattr(cfg, "num_experts", None) or getattr(
        cfg, "num_local_experts", None
    )
    top_k = getattr(cfg, "num_experts_per_tok")
    hidden = getattr(cfg, "hidden_size")
    moe_inter = getattr(cfg, "moe_intermediate_size", None) or getattr(
        cfg, "intermediate_size"
    )
    print(f"[cfg] {model_type} E={num_experts} top_k={top_k} "
          f"hidden={hidden} moe_inter={moe_inter}")

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    load_kwargs = dict(
        dtype=DTYPES[args.dtype], low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    )
    if model_type == "gpt_oss":
        from transformers import Mxfp4Config

        load_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
        print("[load] Mxfp4Config(dequantize=True)")
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.to(args.device).eval()

    prompts = [
        (b, pr)
        for b, items in PROMPT_BANK.items()
        for pr in items[: args.prompts_per_bucket]
    ]
    store: dict = {}
    num_layers = None
    for i, (bucket, prompt) in enumerate(prompts):
        enc = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=args.max_length).to(args.device)
        plen = int(enc["input_ids"].shape[1])
        with torch.inference_mode():
            gen = model.generate(
                **enc, max_new_tokens=args.gen_tokens, do_sample=True,
                temperature=args.temperature, top_p=args.top_p,
                pad_token_id=tok.pad_token_id,
            )
        full = gen[0]
        with torch.inference_mode():
            out = model(input_ids=full.unsqueeze(0), use_cache=False,
                        output_router_logits=True)
        # [num_layers, T, top_k] int16
        topk = np.stack([
            rl.float().topk(top_k, dim=-1).indices.cpu().numpy().astype(np.int16)
            for rl in out.router_logits
        ])
        num_layers = topk.shape[0]
        store[f"seq{i}_topk"] = topk
        store[f"seq{i}_plen"] = np.int32(plen)
        store[f"seq{i}_bucket"] = bucket
        print(f"  [{i}] {bucket}: plen={plen} T={full.shape[0]} layers={num_layers}")

    meta = dict(
        model=args.model, model_type=model_type, num_experts=int(num_experts),
        top_k=int(top_k), num_layers=int(num_layers), hidden=int(hidden),
        moe_inter=int(moe_inter), n_seq=len(prompts), gen_tokens=args.gen_tokens,
    )
    store["meta"] = json.dumps(meta)
    np.savez(npz_path, **store)
    print(f"[saved] {npz_path}")
    del model
    torch.cuda.empty_cache()
    return load_trace(npz_path)


def load_trace(npz_path: Path) -> dict:
    z = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    seqs = []
    for i in range(meta["n_seq"]):
        seqs.append(dict(
            topk=z[f"seq{i}_topk"].astype(np.int64),  # [L, T, k]
            plen=int(z[f"seq{i}_plen"]),
            bucket=str(z[f"seq{i}_bucket"]),
        ))
    return dict(meta=meta, seqs=seqs)


# --------------------------------------------------------------------------- #
# Analysis (CPU)
# --------------------------------------------------------------------------- #
def decode_slices(seqs):
    """Yield (layer-major) decode-region top-k arrays [D, k] per seq per layer."""
    for s in seqs:
        L, T, k = s["topk"].shape
        ds = s["plen"]
        if T - ds < 2:
            continue
        yield s, ds


def union_over_k(trace, k_list, batch_list, rng):
    """Mean distinct experts/layer in a cycle of k tokens x B sequences."""
    meta = trace["meta"]
    L, top_k = meta["num_layers"], meta["top_k"]
    seqs = trace["seqs"]
    # decode-region top-k per seq: list of [L, D, k]
    dec = []
    for s in seqs:
        T = s["topk"].shape[1]
        ds = s["plen"]
        if T - ds >= 2:
            dec.append(s["topk"][:, ds:, :])  # [L, D, k]
    rows = []
    for B in batch_list:
        if B > len(dec):
            continue
        for k in k_list:
            uvals = []  # distinct experts per layer per cycle
            # sample many cycles: random group of B seqs, random start window
            n_samples = 400
            for _ in range(n_samples):
                grp = rng.choice(len(dec), size=B, replace=False)
                # common window length over the group
                dmin = min(dec[g].shape[1] for g in grp)
                if dmin < k:
                    continue
                start = int(rng.integers(0, dmin - k + 1))
                for layer in range(L):
                    s = set()
                    for g in grp:
                        blk = dec[g][layer, start:start + k, :]  # [k, top_k]
                        s.update(blk.reshape(-1).tolist())
                    uvals.append(len(s))
            u = float(np.mean(uvals))
            naive = B * k * top_k
            rows.append(dict(
                B=B, k=k, distinct_per_cycle=round(u, 2),
                amortized_per_step=round(u / k, 2),
                naive=naive, reuse_factor=round(naive / u, 2),
                frac_of_pool=round(u / meta["num_experts"], 3),
            ))
    return rows


def cache_hit_curves(trace, cache_list):
    """Coverage (fraction of a position's top-k already resident) vs cache C,
    for: global-static (top-C by global decode freq), per-request static (top-C
    by request gate-count), verify-warmed LRU, and random C/E baseline."""
    meta = trace["meta"]
    L, E = meta["num_layers"], meta["num_experts"]
    seqs = trace["seqs"]

    # global per-layer frequency over all decode positions
    gfreq = np.zeros((L, E), dtype=np.int64)
    for s in seqs:
        T = s["topk"].shape[1]
        ds = s["plen"]
        if T - ds < 2:
            continue
        for layer in range(L):
            ids, cnt = np.unique(s["topk"][layer, ds:, :], return_counts=True)
            gfreq[layer, ids] += cnt
    grank = np.argsort(-gfreq, axis=1)  # [L, E] hottest-first per layer

    rows = []
    for C in cache_list:
        gcov, rcov, lcov = [], [], []
        gkeep = [set(grank[layer, :C].tolist()) for layer in range(L)]
        for s in seqs:
            T = s["topk"].shape[1]
            ds = s["plen"]
            if T - ds < 2:
                continue
            # per-request static keep set per layer (top-C by request count)
            rkeep = []
            for layer in range(L):
                ids, cnt = np.unique(s["topk"][layer, ds:, :], return_counts=True)
                order = ids[np.argsort(-cnt)]
                rkeep.append(set(order[:C].tolist()))
            for layer in range(L):
                tk = s["topk"][layer]
                # LRU warmed by verified routing (positions < t)
                cache, cset = [], set()
                for t in range(T):
                    experts = tk[t].tolist()
                    if t >= ds:
                        gcov.append(sum(e in gkeep[layer] for e in experts) / len(experts))
                        rcov.append(sum(e in rkeep[layer] for e in experts) / len(experts))
                        lcov.append(sum(e in cset for e in experts) / len(experts))
                    for e in experts:
                        if e in cset:
                            cache.remove(e)
                        cache.insert(0, e)
                        cset.add(e)
                    while len(cache) > C:
                        cset.discard(cache.pop())
        rows.append(dict(
            C=C, C_over_E=round(C / E, 3),
            global_static=round(float(np.mean(gcov)), 4),
            per_request_static=round(float(np.mean(rcov)), 4),
            verify_warmed_lru=round(float(np.mean(lcov)), 4),
            random_baseline=round(C / E, 4),
        ))
    return rows, gfreq


def skew_summary(gfreq):
    """Fraction of routing traffic captured by the hottest top-q experts."""
    L, E = gfreq.shape
    out = {}
    for q in (0.1, 0.25, 0.5):
        c = max(1, int(round(q * E)))
        fr = []
        for layer in range(L):
            f = np.sort(gfreq[layer])[::-1]
            tot = f.sum()
            if tot > 0:
                fr.append(f[:c].sum() / tot)
        out[f"top{int(q*100)}pct_traffic"] = round(float(np.mean(fr)), 4)
    return out


def pcie_synthesis(trace, union_rows, cache_rows, args):
    """Streamed bytes/cycle (after cache+reuse) vs hideable budget k*T_step*BW."""
    meta = trace["meta"]
    L, E = meta["num_layers"], meta["num_experts"]
    expert_params = 3 * meta["hidden"] * meta["moe_inter"]
    expert_bytes = expert_params * args.draft_bits / 8.0
    bw = args.pcie_gbps * 1e9
    # resident cache = bf16 verify shard (E/ep) + we report at the cache sizes given;
    # use the verify-warmed LRU hit rate as the resident-served fraction.
    lru = {r["C"]: r["verify_warmed_lru"] for r in cache_rows}
    rows = []
    # pick a representative cache size = 32 (Qwen3 25%) if present else max
    for C in sorted(lru):
        hit = lru[C]
        for ur in union_rows:
            B, k, u = ur["B"], ur["k"], ur["distinct_per_cycle"]
            streamed_experts = u * (1.0 - hit) * L  # over all layers, per cycle
            streamed_bytes = streamed_experts * expert_bytes
            hideable = k * (args.tstep_ms / 1e3) * bw
            fit = streamed_bytes / hideable if hideable > 0 else float("inf")
            rows.append(dict(
                C=C, B=B, k=k, cache_hit=round(hit, 3),
                streamed_experts=round(streamed_experts, 1),
                streamed_GB=round(streamed_bytes / 1e9, 3),
                hideable_GB=round(hideable / 1e9, 3),
                fit_ratio=round(fit, 3),
                hidden=bool(fit <= 1.0),
            ))
    return rows, dict(
        expert_bytes_MB=round(expert_bytes / 1e6, 3),
        pcie_gbps=args.pcie_gbps, tstep_ms=args.tstep_ms, draft_bits=args.draft_bits,
    )


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    npz_path = args.outdir / "data" / f"{args.tag}_routing.npz"
    if args.reuse and npz_path.exists():
        print(f"[reuse] {npz_path}")
        trace = load_trace(npz_path)
    else:
        trace = capture(args, npz_path)

    meta = trace["meta"]
    print(f"\n=== {args.tag}: E={meta['num_experts']} top_k={meta['top_k']} "
          f"layers={meta['num_layers']} seqs={meta['n_seq']} ===")

    union_rows = union_over_k(trace, args.k_list, args.batch_list, rng)
    cache_rows, gfreq = cache_hit_curves(trace, args.cache_list)
    skew = skew_summary(gfreq)
    pcie_rows, pcie_meta = pcie_synthesis(trace, union_rows, cache_rows, args)

    print("\n--- union-over-k (distinct experts/layer per cycle) ---")
    for r in union_rows:
        print(f"B={r['B']:>2} k={r['k']:>2}: distinct={r['distinct_per_cycle']:>6} "
              f"amort/step={r['amortized_per_step']:>5} reuse={r['reuse_factor']:>5} "
              f"frac_pool={r['frac_of_pool']}")
    print("\n--- cache-hit curve (coverage vs C) ---")
    for r in cache_rows:
        print(f"C={r['C']:>3} (C/E={r['C_over_E']:<5}): "
              f"global={r['global_static']} per_req={r['per_request_static']} "
              f"lru={r['verify_warmed_lru']} rand={r['random_baseline']}")
    print(f"\n--- routing skew --- {skew}")
    print(f"\n--- PCIe synthesis (expert={pcie_meta['expert_bytes_MB']}MB "
          f"@ {pcie_meta['draft_bits']}bit, {pcie_meta['pcie_gbps']}GB/s, "
          f"T_step={pcie_meta['tstep_ms']}ms) ---")
    for r in pcie_rows:
        if r["k"] in (1, 4) and r["B"] in (1, 8):
            print(f"C={r['C']:>3} B={r['B']} k={r['k']}: hit={r['cache_hit']} "
                  f"streamed={r['streamed_GB']}GB hideable={r['hideable_GB']}GB "
                  f"fit={r['fit_ratio']} hidden={r['hidden']}")

    out = dict(
        meta=meta, pcie_meta=pcie_meta, skew=skew,
        union_over_k=union_rows, cache_hit=cache_rows, pcie_synthesis=pcie_rows,
    )
    json_path = args.outdir / "data" / f"{args.tag}_locality.json"
    with json_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
