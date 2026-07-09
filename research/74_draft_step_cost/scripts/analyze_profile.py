#!/usr/bin/env python
"""Phase 74 profiling analysis: is the DRAFT step optimized?

Per (ctx, batch):
  D          = draft_forward (spec profiler, one draft chain step, B*1 tok, EAGER)
  V          = verify        (spec profiler, target forward, B*(K+1) tok, graphed)
  nospec_fwd = B / nospec_tok_s   (a graphed forward over B*1 tok -- the reference)
  D/ref      = draft overhead vs an equivalent graphed forward (>1 => not optimized)
  D/V        = regime indicator (small => draft cheap vs verify)
Usage: analyze_profile.py [data_dir]
"""
import glob, json, os, re, sys

PHASE = "/data/smcho/self-spec-moe/research/74_draft_step_cost"
DATA = sys.argv[1] if len(sys.argv) > 1 else f"{PHASE}/data"
K = 4


def prof(dr, lbl):
    num = den = 0.0
    for p in glob.glob(f"{DATA}/prof_{dr}/self_spec_profile_*.json"):
        try: s = json.load(open(p)).get("summary", {}).get(lbl) or {}
        except Exception: continue
        if s.get("mean_ms") and s.get("n"): num += s["mean_ms"]*s["n"]; den += s["n"]
    return num/den if den else None


def nospec_tps(ctx):
    """{batch: tok_s} from the batch-list nospec run at this ctx."""
    out = {}
    for f in glob.glob(f"{DATA}/w72n_p74_nospec_ctx{ctx}_nospec*.json"):
        try: d = json.load(open(f))
        except Exception: continue
        for _k, rows in d.get("by_k", {}).items():
            for r in rows:
                if r.get("batch") and r.get("tok_s_mean"):
                    out[r["batch"]] = r["tok_s_mean"]
    return out


def spec_tps(ctx, b):
    for f in glob.glob(f"{DATA}/w72n_p74_base_ctx{ctx}_b{b}_spec*.json"):
        try: d = json.load(open(f))
        except Exception: continue
        for _k, rows in d.get("by_k", {}).items():
            for r in rows:
                if r.get("batch") == b:
                    return r.get("tok_s_mean"), r.get("accept_len")
    return None, None


def main():
    ctxs = sorted({int(m.group(1)) for f in glob.glob(f"{DATA}/prof_base_ctx*")
                   if (m := re.search(r"prof_base_ctx(\d+)_b\d+", f))})
    batches = sorted({int(m.group(1)) for f in glob.glob(f"{DATA}/prof_base_ctx*")
                      if (m := re.search(r"_b(\d+)$", f))})
    hdr = (f"{'ctx':>6} {'b':>3} {'D(draft)':>9} {'V(verify)':>10} "
           f"{'nospec_fwd':>11} {'D/ref':>6} {'D/V':>6} {'spec_tps':>9} "
           f"{'nospec_tps':>11} {'accept':>6} {'speedup':>8}")
    print(hdr); print("-"*len(hdr))
    for ctx in ctxs:
        nt = nospec_tps(ctx)
        for b in batches:
            D = prof(f"base_ctx{ctx}_b{b}", "draft_forward")
            V = prof(f"base_ctx{ctx}_b{b}", "verify")
            ntps = nt.get(b)
            ref = (b/ntps*1000) if ntps else None   # ms per graphed B*1 forward
            stps, acc = spec_tps(ctx, b)
            dref = (D/ref) if (D and ref) else None
            dv = (D/V) if (D and V) else None
            spd = (stps/ntps) if (stps and ntps) else None
            def f(x, p=2): return f"{x:.{p}f}" if isinstance(x,(int,float)) else "-"
            print(f"{ctx:>6} {b:>3} {f(D):>9} {f(V):>10} {f(ref):>11} "
                  f"{f(dref):>6} {f(dv):>6} {f(stps,0):>9} {f(ntps,0):>11} "
                  f"{f(acc):>6} {f(spd):>8}")
    print("\nD/ref > 1 => draft carries eager/propose overhead vs a graphed forward.")
    print("D/V small + accept high => spec-favorable (memory-bound); D/V ~1 => draft~verify.")


if __name__ == "__main__":
    main()
