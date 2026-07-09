#!/usr/bin/env python3
"""E1 analyzer: measured Tq/Tp, plus a fit for F (fixed overhead) and BW_eff.

Per-decode-step ms is the SLOPE across two output lengths, so prefill cancels:
    t_step = (T(o2) - T(o1)) / (o2 - o1)

Then, for the three arms (bf16 / W8 / W4) whose weight BYTES are known exactly:
    t_step(arm) = F + bytes(arm) / BW_eff
is 3 equations in 2 unknowns -> least squares. The fit answers the question Phase 74
raised: is the quantized kernel actually CONVERTING bytes into time?
  - good fit + BW_eff near HBM peak  -> the lever works; any shortfall vs the paper is F
  - poor fit, or W4 far above its predicted t_step -> the kernel is NOT bandwidth-bound
    (tile/latency bound), i.e. halving bytes buys nothing. That was the fp8 result.

Only FFN+QKVO are quantized (lm_head stays fp16), matching EfficientRollout 4.1.
"""

import argparse
import glob
import json
import os
import sys


def model_bytes(cfg, ctx, batch, w_bits):
    L = cfg["num_hidden_layers"]
    H = cfg["hidden_size"]
    I = cfg["intermediate_size"]
    KVH = cfg["num_key_value_heads"]
    HD = cfg.get("head_dim", H // cfg["num_attention_heads"])
    VOC = cfg["vocab_size"]
    attn = H * H + 2 * H * (KVH * HD) + H * H          # q,k,v,o
    mlp = 3 * H * I                                     # gate,up,down
    body = (attn + mlp) * L                             # the RTN-quantized set
    lmh = VOC * H                                       # NOT quantized
    kv = batch * ctx * (2 * KVH * HD * 2 * L)           # bf16 KV
    return body * (w_bits / 8) + lmh * 2 + kv, body, lmh, kv


def lstsq_F_BW(rows):
    """t = F + bytes/BW  ->  solve [1, bytes] @ [F, 1/BW] = t  (normal equations)."""
    n = len(rows)
    sx = sum(b for b, _ in rows)
    sy = sum(t for _, t in rows)
    sxx = sum(b * b for b, _ in rows)
    sxy = sum(b * t for b, t in rows)
    den = n * sxx - sx * sx
    if den == 0:
        return None, None
    inv_bw = (n * sxy - sx * sy) / den
    F = (sy - inv_bw * sx) / n
    return F, (1.0 / inv_bw if inv_bw > 0 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--o1", type=int, default=1)
    ap.add_argument("--o2", type=int, default=33)
    ap.add_argument("--data", default=None)
    a = ap.parse_args()

    data = a.data or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(a.model).to_dict()

    BITS = {"bf16": 16, "w8": 8, "w4": 4}
    arms = {}
    for f in sorted(glob.glob(os.path.join(data, f"e1_*_o{a.o1}.json"))):
        name = os.path.basename(f)[len("e1_"):-len(f"_o{a.o1}.json")]
        f2 = os.path.join(data, f"e1_{name}_o{a.o2}.json")
        if not os.path.exists(f2):
            continue
        t1 = json.load(open(f))["avg_latency"]
        t2 = json.load(open(f2))["avg_latency"]
        step_ms = (t2 - t1) / (a.o2 - a.o1) * 1e3
        bits = BITS[name.split("_")[0]]
        nbytes, body, lmh, kv = model_bytes(cfg, a.ctx, 1, bits)
        arms[name] = dict(step_ms=step_ms, bytes=nbytes, bits=bits)

    if not arms:
        print(f"no E1 data in {data}; run run_e1_tqtp.sh first")
        return 1

    base = arms.get("bf16")
    if not base:
        print("missing bf16 baseline arm -- cannot form Tq/Tp")
        return 1
    Tp = base["step_ms"]

    print(f"model {a.model}   ctx={a.ctx}  batch=1   step = slope over {a.o2-a.o1} decodes\n")
    print(f"{'arm':>12} {'bytes(GB)':>10} {'t_step(ms)':>11} {'Tq/Tp':>8}   paper")
    print("-" * 60)
    PAPER = {"w4": 0.360, "w8": 0.573}
    for name in sorted(arms):
        r = arms[name]
        ratio = r["step_ms"] / Tp
        key = name.split("_")[0]
        ref = f"{PAPER[key]:.3f}" if key in PAPER else "1.000 (ref)"
        print(f"{name:>12} {r['bytes']/1e9:10.2f} {r['step_ms']:11.3f} {ratio:8.3f}   {ref}")

    # Fit F and BW_eff on the AUTO-kernel arms (one kernel family, comparable).
    fit_rows = [(arms[n]["bytes"], arms[n]["step_ms"] / 1e3)
                for n in ("bf16", "w8_machete", "w4_machete") if n in arms]
    if len(fit_rows) < 3:
        fit_rows = [(arms[n]["bytes"], arms[n]["step_ms"] / 1e3)
                    for n in ("bf16", "w8_marlin", "w4_marlin") if n in arms]
    if len(fit_rows) >= 3:
        F, BW = lstsq_F_BW(fit_rows)
        print(f"\nfit  t = F + bytes/BW_eff   ->  F = {F*1e3:.3f} ms   "
              f"BW_eff = {BW/1e12:.2f} TB/s")
        print("residuals (measured - fit), ms:")
        for (b, t) in fit_rows:
            print(f"   bytes={b/1e9:6.2f} GB  resid={(t - (F + b/BW)) * 1e3:+.3f}")
        print("\nread this as:")
        print("  * BW_eff near HBM peak + small residuals -> quant IS converting bytes to time")
        print("    (lever works; any shortfall vs the paper's 0.360 is F, i.e. your GPU).")
        print("  * W4 residual strongly POSITIVE -> the W4 kernel is latency/tile-bound, not")
        print("    bandwidth-bound: halving bytes buys nothing. That is the Phase-74 fp8 story.")
    else:
        print("\n(need bf16 + w8 + w4 on the same kernel to fit F and BW_eff)")

    print("\nPASS: Tq/Tp(W4) <= 0.50 on H100 (<= 0.40 on A100). FAIL: >= 0.70.")
    print("H100 is the HARDER test: faster BW shrinks the byte term, so F is a bigger share.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
