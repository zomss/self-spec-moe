"""Phase-43 K-retune analysis: measured + draft-shielded speedup per K.

Spec tok/s for K in {2,3} from research/43_kretune/data; K=4 spec and the
K-independent no-spec baselines reused from research/42_comm_sweep/data.

speedup(K,d,b) = spec_tok/s(K,d,b) / nospec_tok/s(d,b)   (same delay+batch).

Draft-shielded projection (true comm-free multi-node: draft pays 0 A2A, only the
full-EP verify pays it, amortized over accept_len). Per (K,b) fit inverse-throughput
(proportional to per-batch decode time) linear in delay d:
    T_mode(d) = a_mode + s_mode * d      (least squares on 1/tok_s)
Shielded spec keeps its d=0 compute a_sp but replaces its delay-slope with the
verify-only, accept-amortized slope taken from the measured nospec slope:
    s_shielded = s_ns / accept_len
    spec_shielded_tok/s(d) = 1 / (a_sp + s_shielded * d)
Analytic shielded crossover: a_ns + s_ns*d = a_sp + s_sh*d
    d* = (a_sp - a_ns) / (s_ns - s_sh).
"""
import glob
import json
import os

KRDATA = "/data/smcho/self-spec-moe/research/43_kretune/data"
SWDATA = "/data/smcho/self-spec-moe/research/42_comm_sweep/data"
DELAYS = [0, 100, 250, 500]
BATCHES = [int(x) for x in os.environ.get("KR_BATCHES", "64").split(",")]
KS = [2, 3, 4]


def _rows(path):
    with open(path) as f:
        d = json.load(f)
    out = {}
    for r in d.get("results", []):
        if isinstance(r, dict) and "batch" in r and "error" not in r:
            out[r["batch"]] = r
    return out


def _merge(paths):
    m = {}
    for p in paths:
        m.update(_rows(p))
    return m


def nospec(a2a):
    files = glob.glob(os.path.join(SWDATA, f"w7fp8_sweep_a2a{a2a}_nospec_*_nospec.json"))
    return _merge(files)


def spec(a2a, k):
    if k == 4:
        files = glob.glob(os.path.join(SWDATA, f"w7fp8_sweep_a2a{a2a}_fp8_spec_*_K4.json"))
    else:
        files = glob.glob(os.path.join(KRDATA, f"w7fp8_kr_a2a{a2a}_fp8_spec_*_K{k}.json"))
    return _merge(files)


def _lin_fit(xs, ys):
    n = len(xs)
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return sy / n, 0.0
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    return a, b


def interp_crossover(pts):
    pts = sorted(pts)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if (y0 - 1.0) * (y1 - 1.0) <= 0 and y1 != y0:
            return x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0), "interp"
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0
    if slope > 1e-9:
        return x1 + (1.0 - y1) / slope, "extrap"
    return None, "none"


def main():
    NS = {a: nospec(a) for a in DELAYS}
    SP = {(k, a): spec(a, k) for k in KS for a in DELAYS}

    # accept_len per (K, b): average over delays (stable).
    accept = {}
    for k in KS:
        for b in BATCHES:
            als = [SP[(k, a)][b].get("accept_len") for a in DELAYS
                   if b in SP[(k, a)] and SP[(k, a)][b].get("accept_len")]
            accept[(k, b)] = sum(als) / len(als) if als else None

    for b in BATCHES:
        print(f"\n########## BATCH {b} ##########")
        print("=== MEASURED speedup table (spec/nospec) ===")
        print(f"{'a2a_us':>7}{'nospec':>10}"
              + "".join(f"{'K'+str(k)+'_tok':>10}{'K'+str(k)+'_su':>9}" for k in KS))
        for a in DELAYS:
            nsv = NS[a].get(b, {}).get("tok_s_mean")
            row = f"{a:>7}{(nsv or 0):>10.1f}"
            for k in KS:
                sp = SP[(k, a)].get(b, {}).get("tok_s_mean")
                su = (sp / nsv) if (sp and nsv) else None
                row += f"{(sp or 0):>10.1f}" + (f"{su:>9.3f}" if su else f"{'-':>9}")
            print(row)

        print("\naccept_len per K (avg over delays):")
        for k in KS:
            al = accept.get((k, b))
            print(f"  K={k}: accept_len = {al:.3f}" if al else f"  K={k}: n/a")

        print("\n=== Best K at each delay (measured) ===")
        for a in DELAYS:
            nsv = NS[a].get(b, {}).get("tok_s_mean")
            sus = {}
            for k in KS:
                sp = SP[(k, a)].get(b, {}).get("tok_s_mean")
                if sp and nsv:
                    sus[k] = sp / nsv
            if sus:
                bestk = max(sus, key=sus.get)
                detail = ", ".join(f"K{k}={sus[k]:.3f}" for k in sorted(sus))
                print(f"  a2a={a:>4}us: best=K{bestk} ({sus[bestk]:.3f})   [{detail}]")

        # per-K delay-slope fits + shielded projection
        print("\n=== Delay-slope fits + DRAFT-SHIELDED projection per K ===")
        # nospec fit (shared)
        ds_ns = [a for a in DELAYS if NS[a].get(b, {}).get("tok_s_mean")]
        a_ns, s_ns = _lin_fit(ds_ns, [1.0 / NS[a][b]["tok_s_mean"] for a in ds_ns])
        print(f"  nospec: a_ns={a_ns:.3e}  s_ns={s_ns:.3e}/us")
        print(f"{'K':>3}{'a_sp':>11}{'s_sp':>12}{'s_sp/s_ns':>10}"
              f"{'accept':>8}{'s_sh':>12}{'d*_shield':>11}{'meas_xover':>12}")
        shield = {}
        for k in KS:
            ds = [a for a in DELAYS if SP[(k, a)].get(b, {}).get("tok_s_mean")]
            a_sp, s_sp = _lin_fit(ds, [1.0 / SP[(k, a)][b]["tok_s_mean"] for a in ds])
            al = accept.get((k, b)) or 3.0
            s_sh = s_ns / al
            d_star = (a_sp - a_ns) / (s_ns - s_sh) if (s_ns - s_sh) != 0 else None
            # measured crossover
            mpts = [(a, SP[(k, a)][b]["tok_s_mean"] / NS[a][b]["tok_s_mean"])
                    for a in DELAYS
                    if SP[(k, a)].get(b, {}).get("tok_s_mean") and NS[a].get(b, {}).get("tok_s_mean")]
            mcx, mkind = interp_crossover(mpts) if len(mpts) >= 2 else (None, "none")
            shield[k] = (a_sp, s_sp, s_sh, al, a_ns, s_ns)
            dss = f"{d_star:.0f}us" if (d_star and d_star > 0) else "none/neg"
            mcxs = f"{mcx:.0f}({mkind})" if mcx else "none"
            print(f"{k:>3}{a_sp:>11.3e}{s_sp:>12.3e}{s_sp/s_ns:>10.3f}"
                  f"{al:>8.2f}{s_sh:>12.3e}{dss:>11}{mcxs:>12}")

        print("\n=== SHIELDED speedup table per K (draft pays 0 A2A) ===")
        grid = DELAYS + [1000]
        print(f"{'a2a_us':>7}" + "".join(f"{'K'+str(k):>9}" for k in KS))
        for d in grid:
            row = f"{d:>7}"
            for k in KS:
                a_sp, s_sp, s_sh, al, a_ns, s_ns = shield[k]
                t_ns = a_ns + s_ns * d
                t_sp = a_sp + s_sh * d
                row += f"{(t_ns / t_sp):>9.3f}"
            print(row)

        print("\n=== SHIELDED speedup @ realistic multi-node delays ===")
        print(f"{'a2a_us':>7}" + "".join(f"{'K'+str(k):>9}" for k in KS))
        for d in (30, 100, 150, 200, 300):
            row = f"{d:>7}"
            for k in KS:
                a_sp, s_sp, s_sh, al, a_ns, s_ns = shield[k]
                row += f"{((a_ns + s_ns * d) / (a_sp + s_sh * d)):>9.3f}"
            print(row)


if __name__ == "__main__":
    main()
