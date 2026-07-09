#!/usr/bin/env python
"""Phase 74: draft-step-cost lever stack analysis.

Per (arm, ctx, batch) reads:
  - harness JSON  data/w72n_p74_<arm>_ctx<ctx>_b<batch>_spec_*.json
      -> accept_len, tok_s_mean, decode_s
  - profiler dir  data/prof_<arm>_ctx<ctx>_b<batch>/self_spec_profile_*.json
      -> draft_forward / draft_forward_first / verify / draft_chain (ms)
  - engine log    logs/<arm>_ctx<ctx>_b<batch>_try*.log
      -> KV pool tokens/rank + kv-window engagement (residency evidence)

Prints one table per arm (batch-scaling) and a cross-arm comparison at the
common batches. Economic bar: draft step D vs verify V -> speedup
accept/(K*(D/V)+1). Usage: analyze_stack.py [data_dir]
"""
import glob
import json
import os
import re
import sys

PHASE = "/data/smcho/self-spec-moe/research/74_draft_step_cost"
DATA = sys.argv[1] if len(sys.argv) > 1 else f"{PHASE}/data"
LOGS = f"{PHASE}/logs"
K = 4


def _prof_region(prof_dir, label):
    """Sample-weighted mean_ms of a region across all rank dumps."""
    num = den = 0.0
    for p in glob.glob(os.path.join(prof_dir, "self_spec_profile_*.json")):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        r = d.get("summary", {}).get(label)
        if r and r.get("mean_ms") is not None and r.get("n"):
            num += r["mean_ms"] * r["n"]
            den += r["n"]
    return (num / den) if den else None


def _accept_tps(arm, ctx, batch):
    for f in glob.glob(f"{DATA}/w72n_p74_{arm}_ctx{ctx}_b{batch}_spec*.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for _k, rows in d.get("by_k", {}).items():
            for r in rows:
                if r.get("batch") == batch:
                    return r.get("accept_len"), r.get("tok_s_mean"), r.get("decode_s")
    return None, None, None


def _pool_and_window(arm, ctx, batch):
    """tokens/rank KV pool + kv-window engaged seq, from the engine log."""
    pool, win, kvq = None, None, None
    for lg in sorted(glob.glob(f"{LOGS}/{arm}_ctx{ctx}_b{batch}_try*.log")):
        txt = open(lg, errors="ignore").read()
        m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", txt)
        if m:
            pool = int(m.group(1).replace(",", ""))
        m = re.findall(r"\[kv-window\].*?win_seq[=\s]+(\d+)", txt)
        if m:
            win = int(m[-1])
        if "kv_cache_dtype=fp8" in txt or "cache_dtype='fp8'" in txt:
            kvq = "fp8"
    return pool, win, kvq


def _discover():
    pts = {}
    for f in glob.glob(f"{DATA}/w72n_p74_*_spec*.json"):
        m = re.search(r"w72n_p74_(\w+?)_ctx(\d+)_b(\d+)_spec", os.path.basename(f))
        if m:
            arm, ctx, b = m.group(1), int(m.group(2)), int(m.group(3))
            pts.setdefault(arm, set()).add((ctx, b))
    return pts


def main():
    pts = _discover()
    if not pts:
        print(f"no p74 result JSONs under {DATA}")
        return
    order = [a for a in ("base", "window", "kvq") if a in pts] + \
            [a for a in pts if a not in ("base", "window", "kvq")]
    hdr = (f"{'arm':6} {'ctx':>6} {'b':>3} {'pool_tok':>9} {'win':>5} "
           f"{'draft_ms':>8} {'df1_ms':>7} {'verify_ms':>9} {'D/V':>5} "
           f"{'accept':>6} {'tok/s':>7} {'speedup':>7}")
    print(hdr)
    print("-" * len(hdr))
    rows = {}
    for arm in order:
        for ctx, b in sorted(pts[arm]):
            df = _prof_region(f"{DATA}/prof_{arm}_ctx{ctx}_b{b}", "draft_forward")
            df1 = _prof_region(f"{DATA}/prof_{arm}_ctx{ctx}_b{b}", "draft_forward_first")
            vf = _prof_region(f"{DATA}/prof_{arm}_ctx{ctx}_b{b}", "verify")
            acc, tps, _ = _accept_tps(arm, ctx, b)
            pool, win, _ = _pool_and_window(arm, ctx, b)
            # cycle draft cost ~ (K-1)*df + df1 ; economic D ~ mean draft step
            dv = (df / vf) if (df and vf) else None
            spd = (acc / (K * dv + 1)) if (acc and dv) else None
            rows[(arm, ctx, b)] = dict(pool=pool, win=win, df=df, df1=df1,
                                       vf=vf, dv=dv, acc=acc, tps=tps, spd=spd)
            def f(x, p=1): return f"{x:.{p}f}" if isinstance(x, (int, float)) else "-"
            print(f"{arm:6} {ctx:>6} {b:>3} {str(pool or '-'):>9} "
                  f"{str(win or '-'):>5} {f(df):>8} {f(df1):>7} {f(vf):>9} "
                  f"{f(dv,2):>5} {f(acc,2):>6} {f(tps,0):>7} {f(spd,2):>7}")
    # cross-arm at common batches
    print("\n== cross-arm (16k) : tok/s and accept vs base ==")
    base_pts = {(c, b): rows[('base', c, b)] for (a, c, b) in rows if a == 'base'}
    for (arm, ctx, b), r in sorted(rows.items()):
        if arm == 'base' or ctx != 16384:
            continue
        base = base_pts.get((ctx, b))
        rel = (f"{r['tps']/base['tps']:.2f}x" if base and base.get('tps')
               and r.get('tps') else "-")
        print(f"  {arm:6} b{b:<3} tok/s={r['tps'] or '-'} "
              f"(vs base {rel}) accept={r['acc']}")


if __name__ == "__main__":
    main()
