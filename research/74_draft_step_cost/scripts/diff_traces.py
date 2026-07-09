#!/usr/bin/env python
"""Diff the per-category GPU time of a DRAFT forward vs a NO-SPEC forward.

Both traces: segment into 48-layer forwards (reshape_and_cache landmark),
categorize kernels, take the median (short-cluster) forward, and print the
category tables side by side with the delta. Proves whether the ~5ms draft
overhead is copy/reshape data-movement absent in the fused no-spec graph.

Usage: diff_traces.py <draft_trace.gz> <nospec_trace.gz>
"""
import gzip, json, sys, statistics as st
from collections import defaultdict

def cat(name):
    n = name.lower()
    if any(k in n for k in ('fused_moe','topkgating','act_and_mul','moe_align',
            'count_and_sort','moe_forward','moe_sum','grouped_gemm')): return 'MoE'
    if any(k in n for k in ('flashattn','flash::','fmha','scaled_dot','softmax',
            'bmm_kernel','attention','fa3','mha')): return 'attention'
    if any(k in n for k in ('rms_norm','rmsnorm','rotary','add_rms','layernorm')): return 'norm_rope'
    if 'reshape_and_cache' in n or 'cache_flash' in n: return 'kv_write'
    if any(k in n for k in ('gemm','cutlass','scaled_mm','matmul','linear')): return 'dense_linear'
    if any(k in n for k in ('memcpy','copy','cat_','index','gather','scatter',
            'elementwise','fill','pad','arange','narrow','slice','view','contiguous')): return 'copy/reshape'
    return 'misc'

def load_forwards(path):
    ev = json.load(gzip.open(path))["traceEvents"]
    GPU = {"kernel","gpu_memset","gpu_memcpy"}
    gk = sorted([e for e in ev if e.get("cat") in GPU and e.get("pid")==0], key=lambda e:e["ts"])
    R = [i for i,e in enumerate(gk) if "reshape_and_cache" in e["name"].lower()]
    L = 48; nf = len(R)//L
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
    return [segs[i] for i in range(nf) if 0<walls[i]<=med*1.15], med

def catmap(segs):
    agg=defaultdict(float); n=len(segs)
    for s in segs:
        for e in s: agg[cat(e["name"])]+=e.get("dur",0)
    return {k: v/1000/n for k,v in agg.items()}, n

draft_segs,_ = load_forwards(sys.argv[1])
nospec_segs,_ = load_forwards(sys.argv[2])
D,nd = catmap(draft_segs); N,nn = catmap(nospec_segs)
cats = sorted(set(D)|set(N), key=lambda k:-(D.get(k,0)+N.get(k,0)))
print(f"{'category':14} {'DRAFT ms':>9} {'NOSPEC ms':>10} {'delta ms':>9}")
print("-"*46)
td=tn=0
for k in cats:
    d,n=D.get(k,0),N.get(k,0); td+=d; tn+=n
    print(f"{k:14} {d:9.3f} {n:10.3f} {d-n:+9.3f}")
print("-"*46)
print(f"{'TOTAL':14} {td:9.3f} {tn:10.3f} {td-tn:+9.3f}")
print(f"\ndraft fwds={nd}  nospec fwds={nn}")
print(f"draft is {td/tn:.2f}x the no-spec forward GPU time; delta = the overhead.")
