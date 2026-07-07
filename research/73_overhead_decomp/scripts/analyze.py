#!/usr/bin/env python
"""Phase 73: draft-forward overhead decomposition.

Reads per-batch SelfSpecProfiler dumps (data/prof_ctx<CTX>_b<BATCH>/) + the
w7_2node harness JSONs (accept_len, tok/s), builds the batch-scaling table, and
fits draft_forward(b_per_rank) = F_fixed + b*m (least squares) plus the
per-token amortization curve.

Per-rank decode batch = W7_BATCHES  (MEASURED: replicated-batch DP -- each of the
8 DP ranks independently runs the full W7_BATCHES-prompt batch. Confirmed this
phase: W7_BATCHES=8 -> steady draft bs=8, KV usage 6.9% = 8 reqs x 2051 tok /
239k pool, "Running: 8 reqs"/engine; W7_BATCHES=16 -> bs=16, 14.1%. So the batch
label b<N> == per-rank decode batch N, NOT N/8.)

Usage: analyze.py            # all points
"""
import glob
import json
import os
import re

PHASE = "/h/v-sukmincho/self-spec-moe/research/73_overhead_decomp/data"
GEMM_FLOOR_MS = 5.11  # P72: fused_moe 2.249 + bmm SDPA 1.912 + cutlass fp8 0.944


def pick_rank(prof_dir):
    """Profiler dump of the busiest (most `verify` samples) rank."""
    best, best_n = None, -1
    for path in glob.glob(os.path.join(prof_dir, "self_spec_profile_*.json")):
        with open(path) as f:
            d = json.load(f)
        n = d.get("counts", {}).get("verify", 0)
        if n > best_n:
            best, best_n = d, n
    return best


def region_mean(d, label):
    r = d["summary"].get(label)
    if not r or r.get("mean_ms") is None:
        return None
    return r["mean_ms"], r.get("n", 0)


def region_mean_allranks(prof_dir, label):
    """Sample-weighted mean of a region across ALL rank dumps (robust vs
    busiest-only). Weights each rank's post-warmup mean by its sample count."""
    num = den = 0.0
    for path in glob.glob(os.path.join(prof_dir, "self_spec_profile_*.json")):
        with open(path) as f:
            d = json.load(f)
        r = d["summary"].get(label)
        if r and r.get("mean_ms") is not None and r.get("n"):
            num += r["mean_ms"] * r["n"]
            den += r["n"]
    return (num / den) if den else None


def accept_for(ctx, batch):
    """accept_len + tok/s from the harness JSON for this ctx/batch/K."""
    for f in glob.glob(f"{PHASE}/w72n_p73_ctx{ctx}_b{batch}_spec_cg_K*.json"):
        d = json.load(open(f))
        for _k, rows in d.get("by_k", {}).items():
            for r in rows:
                if r.get("batch") == batch and r.get("accept_len"):
                    return r.get("accept_len"), r.get("tok_s_mean")
    return None, None


def lstsq_fit(xs, ys):
    """y = F + m*x least squares. Returns (F, m)."""
    n = len(xs)
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    m = (n * sxy - sx * sy) / denom
    F = (sy - m * sx) / n
    return F, m


def main():
    dirs = sorted(glob.glob(f"{PHASE}/prof_ctx*_b*"))
    pts = []  # (ctx, batch, per_rank, draft_fwd, draft_first, verify, chain, acc, toks, n)
    for d in dirs:
        mobj = re.search(r"prof_ctx(\d+)_b(\d+)", os.path.basename(d))
        if not mobj:
            continue
        ctx, batch = int(mobj.group(1)), int(mobj.group(2))
        j = pick_rank(d)
        if j is None:
            print(f"  [skip] {d}: no profiler dumps")
            continue
        df = region_mean_allranks(d, "draft_forward")
        d0 = region_mean_allranks(d, "draft_forward_first")
        vf = region_mean_allranks(d, "verify")
        dc = region_mean_allranks(d, "draft_chain")
        df_busiest = region_mean(j, "draft_forward")
        acc, toks = accept_for(ctx, batch)
        per_rank = batch  # replicated-batch DP: per-rank decode batch == W7_BATCHES
        pts.append({
            "ctx": ctx, "batch": batch, "per_rank": per_rank,
            "draft_fwd": df,
            "draft_fwd_n": df_busiest[1] if df_busiest else 0,
            "draft_first": d0,
            "verify": vf,
            "chain": dc,
            "accept": acc, "toks": toks,
            "verify_n": j.get("counts", {}).get("verify", 0),
        })

    pts.sort(key=lambda r: (r["ctx"], r["batch"]))

    print("=" * 104)
    print("Phase 73 batch-scaling table (per-rank decode batch = W7_BATCHES, "
          "replicated-batch DP)")
    print("=" * 104)
    hdr = (f"{'ctx':>6} {'b_tot':>6} {'b/rank':>7} {'draft_fwd':>10} "
           f"{'df/tok':>8} {'df_first':>9} {'verify':>8} {'vf/tok':>8} "
           f"{'chain':>8} {'accept':>7} {'tok/s':>8} {'n_df':>6}")
    print(hdr)
    print("-" * 104)
    for r in pts:
        pr = r["per_rank"]
        dtok = r["draft_fwd"] / pr if (r["draft_fwd"] and pr) else None
        vtok = r["verify"] / pr if (r["verify"] and pr) else None
        def f(x, w=8, p=2):
            return f"{x:{w}.{p}f}" if x is not None else " " * (w - 2) + "--"
        print(f"{r['ctx']:>6} {r['batch']:>6} {pr:>7.2f} {f(r['draft_fwd'],10)} "
              f"{f(dtok,8)} {f(r['draft_first'],9)} {f(r['verify'],8)} "
              f"{f(vtok,8)} {f(r['chain'],8)} {f(r['accept'],7,3)} "
              f"{f(r['toks'],8,1)} {r['draft_fwd_n']:>6}")

    # ---- Fit on the 2k points (free-scaling regime) ----
    fit_pts = [r for r in pts if r["ctx"] == 2048 and r["draft_fwd"] is not None]
    if len(fit_pts) >= 2:
        xs = [r["per_rank"] for r in fit_pts]
        ys = [r["draft_fwd"] for r in fit_pts]
        F, m = lstsq_fit(xs, ys)
        # R^2
        ybar = sum(ys) / len(ys)
        ss_tot = sum((y - ybar) ** 2 for y in ys)
        ss_res = sum((y - (F + m * x)) ** 2 for x, y in zip(xs, ys))
        r2 = 1 - ss_res / ss_tot if ss_tot else 1.0
        print("\n" + "=" * 104)
        print("FIT  draft_forward(b_per_rank) = F_fixed + b * m   (2k points, lstsq)")
        print("=" * 104)
        print(f"  F_fixed = {F:7.3f} ms   (batch-independent overhead)")
        print(f"  m       = {m:7.3f} ms/token   (marginal per-token draft work)")
        print(f"  R^2     = {r2:7.4f}")
        b1 = F + m * 1
        print(f"  => at b1/rank: draft={b1:.2f} ms; F_fixed is {100*F/b1:.0f}% of it")
        print(f"  GEMM floor (P72, sequential b1 GEMM sum) = {GEMM_FLOOR_MS:.2f} ms")
        print(f"  F_fixed vs GEMM floor: F_fixed/{GEMM_FLOOR_MS:.2f} = "
              f"{F/GEMM_FLOOR_MS:.2f}x")
        # verify fit for comparison
        vys = [r["verify"] for r in fit_pts if r["verify"] is not None]
        vxs = [r["per_rank"] for r in fit_pts if r["verify"] is not None]
        if len(vxs) >= 2:
            Fv, mv = lstsq_fit(vxs, vys)
            print(f"\n  verify_forward fit (all): F={Fv:.3f} ms  m={mv:.3f} ms/token")

        # ---- Segmented fit: the curve is concave (marginal falls with batch).
        # low-batch intercept = the true batch-independent floor; high-batch
        # slope = the asymptotic per-token marginal.
        def seg(lo, hi):
            s = [r for r in fit_pts if lo <= r["per_rank"] <= hi]
            if len(s) >= 2:
                return lstsq_fit([r["per_rank"] for r in s],
                                 [r["draft_fwd"] for r in s]), s
            return None, s
        print("\n  --- segmented draft_forward fit (concave curve) ---")
        (fl, seglo) = seg(1, 8)
        (fh, seghi) = seg(16, 1e9)
        if fl:
            print(f"  low-batch (b<=8):  F_fixed={fl[0]:.3f} ms  m={fl[1]:.3f} ms/tok"
                  f"  (n={len(seglo)})")
        if fh:
            print(f"  high-batch (b>=16): F_int ={fh[0]:.3f} ms  m={fh[1]:.3f} ms/tok"
                  f"  (n={len(seghi)})")
        b1meas = next((r["draft_fwd"] for r in fit_pts if r["per_rank"] == 1), None)
        b64meas = next((r["draft_fwd"] for r in fit_pts if r["per_rank"] == 64), None)
        if b1meas and b64meas:
            print(f"  MEASURED b1={b1meas:.2f} ms -> b64={b64meas:.2f} ms "
                  f"(only {b64meas/b1meas:.2f}x for 64x tokens => fixed-dominated)")
            print(f"  draft per-token: b1={b1meas:.2f} -> b64={b64meas/64:.3f} ms/tok "
                  f"({b1meas/(b64meas/64):.0f}x amortization)")
        # accessibility: interpolate draft at the 16k pool ceiling (per-rank 12)
        s8 = next((r["draft_fwd"] for r in fit_pts if r["per_rank"] == 8), None)
        s16 = next((r["draft_fwd"] for r in fit_pts if r["per_rank"] == 16), None)
        if s8 and s16:
            d12 = s8 + (s16 - s8) * (12 - 8) / (16 - 8)
            print(f"\n  16k ACCESSIBILITY (pool ceiling per-rank~12, Phase66 fp8-replica):")
            print(f"    draft(b12) ~= {d12:.2f} ms, per-token ~= {d12/12:.3f} ms/tok "
                  f"({b1meas/(d12/12):.1f}x amortized vs b1)" if b1meas else "")

    # ---- 16k cross-check ----
    x2k = {r["batch"]: r for r in pts if r["ctx"] == 2048}
    for r in pts:
        if r["ctx"] == 16384 and r["batch"] in x2k:
            a = x2k[r["batch"]]["draft_fwd"]; b = r["draft_fwd"]
            if a and b:
                print(f"\n16k xcheck b{r['batch']}: draft_fwd 2k={a:.2f} 16k={b:.2f} "
                      f"-> {100*(b-a)/a:+.1f}% (ctx-invariant if within ~10%)")


if __name__ == "__main__":
    main()
