"""Phase 72: decompose ONE FULL-CG draft-chain forward from a kineto trace.

Segments the whole GPU timeline into 48-layer forwards using the per-layer
reshape_and_cache landmark (1/layer, used by both draft and target), cutting
forward boundaries at the largest inter-kernel gap between layer-47 of one
forward and layer-0 of the next (this isolates the pure model forward and
drops the inter-step sample/window-build). Classifies each forward draft (D)
vs target (T) by the presence of the target's FlashAttnFwd kernel (the draft
uses math-backend scratchpad SDPA instead). Reports, for a steady-state DRAFT
forward: GPU-active(union) vs wall vs gap fraction; #kernels; median kernel +
gap; and the per-category ms/% table.

Usage: decompose.py <trace.json[.gz]>
"""
import gzip
import json
import sys
import statistics
import collections

path = sys.argv[1]
op = gzip.open if path.endswith(".gz") else open
d = json.load(op(path))
ev = d["traceEvents"]

GPU_CATS = {"kernel", "gpu_memset", "gpu_memcpy"}
gk = [e for e in ev if e.get("cat") in GPU_CATS and e.get("pid") == 0]
gk.sort(key=lambda e: e["ts"])


def categorize(name):
    n = name.lower()
    # MoE: router gate, fused-moe grouped GEMM, silu act, moe-fused rms.
    if ("fused_moe" in n or "topkgating" in n or "act_and_mul" in n
            or "moe_align" in n or "count_and_sort_expert" in n
            or "moe_forward" in n or "moe_sum" in n):
        return "MoE"
    # Attention (scratchpad): math-backend SDPA (bmm QK^T / attn.V), the
    # index/gather that builds the dense scratchpad, softmax (exp + row
    # reduce), the additive mask (bool reduce + masked-fill), FA (target).
    if ("bmm_kernel" in n or "gather_kernel" in n or "index_select" in n
            or "gpu_index_kernel" in n or "index_elementwise" in n
            or "scaled_dot_product" in n or "fmha" in n or "cutlassf" in n
            or "flashattn" in n or "flash::" in n or "softmax" in n
            or "exp_kernel" in n or "masked_fill" in n or "prepare_varlen" in n):
        return "attention"
    # fp8 quant/dequant (per-token dynamic scale + fp8 cast, incl. fused
    # quant+norm triton kernels that carry 'to_copy_abs_clamp').
    if ("scaled_fp8_quant" in n or "segmented_max_reduction" in n
            or "to_copy_abs_clamp" in n or "cutlass_scaled_mm" in n):
        return "fp8_quant"
    # Norm / residual / rope.
    if ("rms_norm" in n or "rmsnorm" in n or "layernorm" in n
            or "rotary" in n or "add_rms" in n):
        return "norm_rope"
    # KV write.
    if "reshape_and_cache" in n or "cache_flash" in n:
        return "kv_write"
    # Dense linears (qkv / o projection): fp8 cutlass GEMM.
    if ("cutlass_3x_gemm" in n or "scaled_mm" in n
            or "matmul_kernel_persistent" in n or "cutlass::device_kernel" in n
            or "gemm" in n):
        return "dense_linear"
    return "misc"


# --- Segment into 48-layer forwards via reshape_and_cache landmarks. ---
LAYERS = 48
R = [i for i, e in enumerate(gk) if "reshape_and_cache" in e["name"]]
nf = len(R) // LAYERS
print(f"reshape landmarks={len(R)}  => forwards={nf}")


def gap_before(i):
    return gk[i]["ts"] - (gk[i - 1]["ts"] + gk[i - 1].get("dur", 0))


cuts = [R[0] - 1]  # boundary before the first full forward
for f in range(1, nf):
    a, b = R[LAYERS * f - 1], R[LAYERS * f]
    best, bg = a, -1.0
    for i in range(a + 1, b + 1):
        g = gap_before(i)
        if g > bg:
            bg, best = g, i
    cuts.append(best - 1)
# tail boundary
last = R[LAYERS * nf - 1]
cuts.append(min(len(gk), last + (R[1] - R[0]) + 8))


def has(seg, key):
    return any(key in e["name"].lower() for e in seg)


kinds, segs = [], []
for f in range(nf):
    seg = gk[cuts[f] + 1:cuts[f + 1] + 1]
    segs.append(seg)
    kinds.append("T" if has(seg, "flashattnfwd") else "D")
print("kinds:", "".join(kinds))
print(f"draft={kinds.count('D')} target={kinds.count('T')}")


def busy_and_gaps(seg):
    iv = sorted((e["ts"], e["ts"] + e.get("dur", 0)) for e in seg)
    busy, gaps = 0.0, []
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s > ce:
            busy += ce - cs
            gaps.append(s - ce)
            cs = s
        ce = max(ce, e)
    busy += ce - cs
    wall = iv[-1][1] - iv[0][0]
    return wall, busy, gaps


# steady-state draft forwards: drop the first 20% (warm/ramping cycles).
didx = [f for f, k in enumerate(kinds) if k == "D"]
didx = didx[len(didx) // 5:]
stats = []
for f in didx:
    wall, busy, gaps = busy_and_gaps(segs[f])
    stats.append((f, wall, busy, len(segs[f]), gaps))

walls = [s[1] for s in stats]
print("\nsteady DRAFT forward walls (ms): "
      + ", ".join(f"{w/1000:.1f}" for w in walls))
print(f"draft wall  ms: median={statistics.median(walls)/1000:.2f} "
      f"min={min(walls)/1000:.2f} max={max(walls)/1000:.2f}")

# representative = the median-wall draft forward
srt = sorted(stats, key=lambda s: s[1])
sel = srt[len(srt) // 2]
f, wall_us, busy_us, nk, gaps = sel
seg = segs[f]
wall, busy = wall_us / 1000.0, busy_us / 1000.0
active = sum(e.get("dur", 0) for e in seg) / 1000.0
durs = sorted(e.get("dur", 0) for e in seg)

print("\n" + "=" * 64)
print(f"SELECTED steady-state DRAFT forward  (forward #{f})")
print("=" * 64)
print(f"wall                : {wall:8.3f} ms")
print(f"GPU-active (union)  : {busy:8.3f} ms   ({busy/wall*100:.1f}% of wall)")
print(f"GPU-active (sum)    : {active:8.3f} ms")
print(f"idle / gap fraction : {(1-busy/wall)*100:7.1f} %   "
      f"(idle {wall-busy:.3f} ms)")
print(f"# kernels           : {nk}")
print(f"median kernel dur   : {statistics.median(durs):7.3f} us")
print(f"mean kernel dur     : {active*1000/nk:7.3f} us")
print(f"# inter-kernel gaps : {len(gaps)}")
print(f"median inter-gap    : {statistics.median(gaps):7.3f} us")
print(f"mean inter-gap      : {sum(gaps)/len(gaps):7.3f} us")

cat_us, cat_n = collections.Counter(), collections.Counter()
name_us, name_n = collections.Counter(), collections.Counter()
for e in seg:
    c = categorize(e["name"])
    dur = e.get("dur", 0)
    cat_us[c] += dur
    cat_n[c] += 1
    name_us[e["name"][:72]] += dur
    name_n[e["name"][:72]] += 1

print("\n--- per-category split of the draft forward ---")
print(f"{'category':13s} {'ms':>7s} {'% active':>9s} {'% wall':>8s} {'#k':>5s}")
for c in ["MoE", "attention", "dense_linear", "fp8_quant", "norm_rope",
          "kv_write", "misc"]:
    ms = cat_us[c] / 1000.0
    print(f"{c:13s} {ms:7.3f} {ms/active*100:8.1f}% {ms/wall*100:7.1f}% "
          f"{cat_n[c]:5d}")
print(f"{'TOTAL(active)':13s} {active:7.3f} {'100.0%':>9s} "
      f"{active/wall*100:7.1f}% {nk:5d}")

print("\n--- top kernels by GPU-us in this draft forward ---")
for nm, us in name_us.most_common(28):
    print(f"  {us/1000:7.3f}ms x{name_n[nm]:4d} [{categorize(nm):11s}] {nm}")
