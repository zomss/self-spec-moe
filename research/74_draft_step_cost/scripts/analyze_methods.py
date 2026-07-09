#!/usr/bin/env python
"""Per-method DRAFT-forward category breakdown, side by side.

For each trace (base/window/fp8d) segment 48-layer draft forwards, categorize
kernels, take the median draft (short-cluster) forward, report ms/forward per
category and the delta vs base. Explains why each method's target slice
reduction is limited.  Usage: analyze_methods.py
"""
import gzip, json, glob, statistics as st
from collections import defaultdict

PHASE = "/data/smcho/self-spec-moe/research/74_draft_step_cost"

def cat(n):
    n = n.lower()
    if any(k in n for k in ('scaled_fp8_quant','to_copy_abs_clamp','fp8_quant','per_token_group_quant','dynamic_scaled')): return 'fp8_quant'
    if any(k in n for k in ('nccl','allgather','all_gather','reducescatter','reduce_scatter','alltoall','all_to_all')): return 'ep_comm'
    if any(k in n for k in ('fused_moe','topkgating','act_and_mul','moe_align','moe_sum','count_and_sort','grouped_gemm')): return 'moe'
    # FA3 attention runs as cutlass::device_kernel (1/layer); check before GEMM.
    if any(k in n for k in ('flashattn','flash::','fmha','scaled_dot','softmax','bmm_kernel','attention','fa3','cutlass::device_kernel','device_kernel')): return 'attention'
    if any(k in n for k in ('matmul_kernel_persistent','nvjet','cutlass','scaled_mm','matmul','addmm','gemm','splitkreduce')): return 'dense_gemm'
    if any(k in n for k in ('rms_norm','rmsnorm','rotary','add_rms','layernorm')): return 'norm_rope'
    if 'reshape_and_cache' in n or 'cache_flash' in n: return 'kv_write'
    if any(k in n for k in ('memcpy','copy','cat_','index','gather','scatter','elementwise','fill','contiguous','pad','arange','narrow')): return 'copy_reshape'
    return 'misc'

def draft_catmap(path):
    ev = json.load(gzip.open(path))["traceEvents"]
    gk = sorted([e for e in ev if e.get("cat") in {"kernel","gpu_memset","gpu_memcpy"} and e.get("pid")==0], key=lambda e:e["ts"])
    R = [i for i,e in enumerate(gk) if "reshape_and_cache" in e["name"].lower()]
    L=48; nf=len(R)//L
    def gb(i): return gk[i]["ts"]-(gk[i-1]["ts"]+gk[i-1].get("dur",0))
    cuts=[R[0]-1]
    for f in range(1,nf):
        a,b=R[L*f-1],R[L*f]; best,bg=a,-1
        for i in range(a+1,b+1):
            g=gb(i)
            if g>bg: bg,best=g,i
        cuts.append(best-1)
    cuts.append(min(len(gk)-1,R[L*nf-1]+(R[1]-R[0])+8))
    segs=[gk[cuts[f]+1:cuts[f+1]+1] for f in range(nf)]
    walls=[((s[-1]["ts"]+s[-1].get("dur",0))-s[0]["ts"])/1000 if s else 0 for s in segs]
    med=st.median([w for w in walls if w>0])
    short=[segs[i] for i in range(nf) if 0<walls[i]<=med*1.15]
    agg=defaultdict(float)
    for s in short:
        for e in s: agg[cat(e["name"])]+=e.get("dur",0)/1000
    n=len(short)
    return {k:v/n for k,v in agg.items()}, sum(agg.values())/n

ARMS=['base','window','fp8d']
maps={}; tots={}
for a in ARMS:
    g=glob.glob(f"{PHASE}/data/trace_m_{a}_16k_b8/*.pt.trace.json.gz")
    if not g: print(f"MISSING trace: {a}"); continue
    maps[a],tots[a]=draft_catmap(g[0])
cats=sorted({c for m in maps.values() for c in m}, key=lambda c:-maps.get('base',{}).get(c,0))
print(f"DRAFT forward GPU ms/forward @16k b8 (batch-invariant OFF)\n")
print(f"{'category':12} " + " ".join(f"{a:>9}" for a in ARMS) + "   | delta vs base")
print("-"*60)
for c in cats:
    row=" ".join(f"{maps.get(a,{}).get(c,0):9.2f}" for a in ARMS)
    b=maps.get('base',{}).get(c,0)
    deltas=" ".join(f"{a}:{maps.get(a,{}).get(c,0)-b:+.2f}" for a in ARMS if a!='base')
    print(f"{c:12} {row}   | {deltas}")
print("-"*60)
print(f"{'TOTAL':12} " + " ".join(f"{tots.get(a,0):9.2f}" for a in ARMS))
for a in ARMS:
    if a in tots: print(f"  {a} draft_fwd total = {tots[a]:.2f} ms  ({tots[a]/tots['base']:.2f}x base)")
