"""Phase 47 (sweep2) analysis: definitive speedup-vs-A2A win curve, fully-optimized
comm-free self-spec (genuine replica + PIECEWISE + all fixes), K=2.

speedup = spec tok/s (Phase 47, piecewise K=2) / no-spec tok/s (REUSED Phase 42).

Six deliverables:
  1. Speedup table (A2A x batch) + measured crossover (speedup=1.0).
  2. Map A2A -> f (comm fraction) from the no-spec decode-time budget, then
     speedup-vs-f, compared against the cost-model ideal accept_len/(K(1-f)+1).
  3. vs pre-piecewise (Phase 42 K=4) delta at each A2A.
  4. Draft-shielding check (measured A2A counts total/active/real from logs) +
     shielded speedup curve (draft pays 0 A2A; verify slope = s_ns/accept_len).
  5. Multi-node projection at realistic f in [0.6, 0.8].
  6. Sanity: 1.008x at a2a=0 b64.
"""
import glob
import json
import os
import re

REPO = "/data/smcho/self-spec-moe"
SPEC_DATA = os.environ.get("SPEC_DATA", f"{REPO}/research/47_sweep2/data")
NOSPEC_DATA = os.environ.get("NOSPEC_DATA", f"{REPO}/research/42_comm_sweep/data")
PRE_PW_DATA = os.environ.get("PRE_PW_DATA", f"{REPO}/research/42_comm_sweep/data")
LOGD = os.environ.get("LOGD", f"{REPO}/research/47_sweep2/logs")
DELAYS = [0, 100, 250, 500, 1000]
BATCHES = [32, 64, 128]
K = 2
K_PREPW = 4


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


def nospec_rows(a2a):
    files = glob.glob(
        os.path.join(NOSPEC_DATA, f"w7fp8_sweep_a2a{a2a}_nospec_*_nospec.json")
    )
    return _merge(files)


def spec_rows_pw(a2a):
    # Phase 47 spec (piecewise K=2): w7fp8_sweep2_a2a{a2a}_fp8_spec_*_K2.json
    files = glob.glob(
        os.path.join(SPEC_DATA, f"w7fp8_sweep2_a2a{a2a}_fp8_spec_*_K{K}.json")
    )
    m = _merge(files)
    # Prefer the clean 5-iter rerun for the noisy a2a=0 b64 point if present.
    if a2a == 0:
        rr = glob.glob(
            os.path.join(SPEC_DATA, f"w7fp8_sweep2rerun_a2a0_fp8_spec_*_K{K}.json")
        )
        m.update(_merge(rr))
    return m


def spec_rows_prepw(a2a):
    # Phase 42 spec (pre-piecewise K=4).
    files = glob.glob(
        os.path.join(PRE_PW_DATA, f"w7fp8_sweep_a2a{a2a}_fp8_spec_*_K{K_PREPW}.json")
    )
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


def parse_a2a_counts():
    """Parse 'Self-spec AgRs all2all count: total=.. active=.. real=..' from
    the spec logs -> return dict a2a_us -> (total,active,real) from any rank.

    These are per-destroy cumulative counts summing dispatch/dispatch_logits/
    combine over the whole run. total = every _emulate_exposed_a2a_delay call
    (incl. the comm-free draft's local-route branch); real = only the true
    all_gatherv/reduce_scatterv (0 for a genuine comm-free draft's MoE)."""
    out = {}
    rx = re.compile(r"all2all count: total=(\d+) active=(\d+) real=(\d+)")
    for a in DELAYS:
        best = None
        for p in glob.glob(os.path.join(LOGD, f"*a2a{a}.log")):
            with open(p, errors="ignore") as f:
                for line in f:
                    m = rx.search(line)
                    if m:
                        t, ac, r = int(m[1]), int(m[2]), int(m[3])
                        # keep the max (rank 0 / full run)
                        if best is None or t > best[0]:
                            best = (t, ac, r)
        if best:
            out[a] = best
    return out


def main():
    # ---- load ----
    NS = {a: nospec_rows(a) for a in DELAYS}
    SP = {a: spec_rows_pw(a) for a in DELAYS}
    PP = {a: spec_rows_prepw(a) for a in DELAYS}

    delays = [a for a in DELAYS if NS[a].get(64) and SP[a].get(64)]

    print("=" * 78)
    print("1. MEASURED SPEEDUP (piecewise K=2 spec / Phase-42 no-spec)")
    print("=" * 78)
    hdr = (f"{'a2a_us':>7} {'batch':>6} {'nospec':>9} {'spec_pw':>9} "
           f"{'accept':>7} {'speedup':>8} {'flag':>5}")
    print(hdr)
    print("-" * len(hdr))
    speedup = {}
    accept = {}
    for a in delays:
        for b in BATCHES:
            nr = NS[a].get(b); sr = SP[a].get(b)
            ns = nr["tok_s_mean"] if nr else None
            sp = sr["tok_s_mean"] if sr else None
            al = sr.get("accept_len") if sr else None
            spd = (sp / ns) if (ns and sp) else None
            speedup[(a, b)] = spd
            accept[(a, b)] = al
            flag = "S" if (nr or {}).get("suspect") or (sr or {}).get("suspect") else ""
            print(f"{a:>7} {b:>6} {ns or 0:>9.1f} {sp or 0:>9.1f} "
                  f"{al or 0:>7.2f} {(spd or 0):>8.3f} {flag:>5}")
        print()

    print("--- Measured crossover (speedup=1.0) per batch ---")
    for b in BATCHES:
        pts = [(a, speedup[(a, b)]) for a in delays if speedup.get((a, b))]
        if len(pts) < 2:
            continue
        cx, kind = interp_crossover(pts)
        mx = max(y for _, y in pts)
        mn = min(y for _, y in pts)
        if cx is None:
            print(f"  b{b}: no crossover; range {mn:.3f}-{mx:.3f}")
        else:
            allwin = all(y >= 1.0 for _, y in pts)
            note = " (>=1.0x EVERYWHERE incl a2a=0)" if allwin else ""
            print(f"  b{b}: crossover ~= {cx:.0f}us ({kind}); "
                  f"range {mn:.3f}-{mx:.3f}{note}")
    print()

    # ---- 2. A2A -> f mapping + cost-model ideal ----
    # f = fraction of no-spec per-token decode time that is A2A (comm). Estimate
    # from the no-spec inverse-throughput linear fit vs delay: T_ns(d)=a_ns+s_ns*d
    # (per fixed token-batch). At delay d, comm time = s_ns*d*C_eff, compute = a_ns.
    # But s_ns already aggregates ALL C collectives' sleep per token. So the A2A
    # fraction of the no-spec decode step at delay d is:
    #     f(d) = s_ns*d / (a_ns + s_ns*d).
    # This is the physically-meaningful comm fraction the cost model uses.
    print("=" * 78)
    print("2. A2A -> f (comm fraction) and speedup-vs-f  vs cost-model ideal")
    print("   ideal = accept_len / (K*(1-f) + 1)   [draft comm-free; verify pays f]")
    print("=" * 78)
    fit_ns = {}
    for b in BATCHES:
        ds = [a for a in delays if NS[a].get(b)]
        a_ns, s_ns = _lin_fit(ds, [1.0 / NS[a][b]["tok_s_mean"] for a in ds])
        fit_ns[b] = (a_ns, s_ns)
    hdr2 = (f"{'a2a_us':>7} {'batch':>6} {'f':>7} {'speedup':>8} "
            f"{'ideal':>8} {'meas/ideal':>11}")
    print(hdr2)
    print("-" * len(hdr2))
    fmap = {}
    for a in delays:
        for b in BATCHES:
            a_ns, s_ns = fit_ns[b]
            f = (s_ns * a) / (a_ns + s_ns * a) if (a_ns + s_ns * a) > 0 else 0.0
            fmap[(a, b)] = f
            spd = speedup.get((a, b))
            al = accept.get((a, b)) or 0.0
            ideal = al / (K * (1 - f) + 1) if al else None
            ratio = (spd / ideal) if (spd and ideal) else None
            print(f"{a:>7} {b:>6} {f:>7.3f} "
                  f"{(spd or 0):>8.3f} {(ideal or 0):>8.3f} "
                  f"{(ratio or 0):>11.3f}")
        print()

    # ---- 3. vs pre-piecewise (Phase 42, K=4) ----
    print("=" * 78)
    print("3. vs PRE-PIECEWISE (Phase 42 K=4): the delta piecewise bought")
    print("=" * 78)
    hdr3 = (f"{'a2a_us':>7} {'batch':>6} {'sp_K4_pre':>9} {'sp_K2_pw':>9} "
            f"{'delta':>7} {'x-factor':>8}")
    print(hdr3)
    print("-" * len(hdr3))
    for a in delays:
        for b in BATCHES:
            nr = NS[a].get(b)
            ns = nr["tok_s_mean"] if nr else None
            pr = PP[a].get(b)
            pre = (pr["tok_s_mean"] / ns) if (pr and ns) else None
            now = speedup.get((a, b))
            delta = (now - pre) if (pre and now) else None
            xf = (now / pre) if (pre and now) else None
            print(f"{a:>7} {b:>6} {(pre or 0):>9.3f} {(now or 0):>9.3f} "
                  f"{(delta or 0):>+7.3f} {(xf or 0):>8.2f}")
        print()

    # ---- 4. Draft-shielding check ----
    print("=" * 78)
    print("4. DRAFT-SHIELDING CHECK (measured A2A counts + shielded curve)")
    print("=" * 78)
    counts = parse_a2a_counts()
    if any(counts.values()):
        print("Measured AgRs all2all counts at engine destroy (per spec run):")
        print(f"  {'a2a_us':>7} {'total':>8} {'active':>8} {'real':>8}  interp")
        for a in delays:
            c = counts.get(a)
            if c:
                t, ac, r = c
                interp = ("draft comm-free (real=0)" if r == 0
                          else f"real={r} true collectives")
                print(f"  {a:>7} {t:>8} {ac:>8} {r:>8}  {interp}")
    else:
        print("Structural: draft real collectives = 0 (Phase-41 bit-identical to plain")
        print("MoE). Runtime destroy-log count NOT captured: V1 force-kills workers on")
        print("shutdown before AgRsAll2AllManager.destroy() fires. Over-charge measured")
        print("empirically from the delay-slopes above (s_sp vs s_sh).")

    # Shielded: draft pays 0 injected sleep; only verify pays it, amortized over
    # accept_len. Shielded spec slope = s_ns/accept_len; d=0 compute unchanged.
    fit_sp = {}
    for b in BATCHES:
        ds = [a for a in delays if SP[a].get(b)]
        a_sp, s_sp = _lin_fit(ds, [1.0 / SP[a][b]["tok_s_mean"] for a in ds])
        fit_sp[b] = (a_sp, s_sp)

    # Empirical over-charge: measured spec delay-slope vs shielded verify-only slope.
    print("\n  Empirical over-charge (delay-slope fit; measured spec vs shielded):")
    print(f"    {'batch':>6}{'s_ns':>11}{'s_sp':>11}{'s_sp/s_ns':>10}"
          f"{'s_sh':>11}{'overcharge':>11}")
    for b in BATCHES:
        a_ns, s_ns = fit_ns[b]
        a_sp, s_sp = fit_sp[b]
        als = [accept[(d, b)] for d in delays if accept.get((d, b))]
        al = sum(als) / len(als) if als else 3.0
        s_sh = s_ns / al
        oc = (s_sp / s_sh - 1) if s_sh else 0
        print(f"    {b:>6}{s_ns:>11.2e}{s_sp:>11.2e}{s_sp / s_ns:>10.3f}"
              f"{s_sh:>11.2e}{oc:>10.0%}")
    print("    (s_sp/s_ns<1: spec less delay-sensitive than no-spec; "
          "s_sp>s_sh: draft over-charge)")

    print("\n  DRAFT-SHIELDED speedup (draft pays 0 A2A; verify slope=s_ns/accept):")
    hdr4 = f"  {'a2a_us':>7}" + "".join(f"{'b'+str(b):>10}" for b in BATCHES)
    print(hdr4)
    shielded = {}
    for a in delays + [1500, 2000]:
        row = f"  {a:>7}"
        for b in BATCHES:
            a_ns, s_ns = fit_ns[b]
            a_sp, _ = fit_sp[b]
            al = accept.get((min(delays, key=lambda d: abs(d - a)), b)) or 4.6
            # use per-batch mean accept over delays
            als = [accept[(d, b)] for d in delays if accept.get((d, b))]
            al = sum(als) / len(als) if als else 3.0
            s_sh = s_ns / al
            t_ns = a_ns + s_ns * a
            t_sp = a_sp + s_sh * a
            sh = t_ns / t_sp
            shielded[(a, b)] = sh
            row += f"{sh:>10.3f}"
        print(row)
    print("  Shielded crossover per batch:")
    for b in BATCHES:
        pts = [(a, shielded[(a, b)]) for a in delays]
        cx, kind = interp_crossover(pts)
        mx = max(y for _, y in pts)
        cxs = f"{cx:.0f}us ({kind})" if cx else "everywhere >=1.0" if all(
            y >= 1 for _, y in pts) else "none"
        print(f"    b{b}: shielded max {mx:.3f}; crossover {cxs}")

    # ---- 5. Multi-node projection at realistic f ----
    print("\n" + "=" * 78)
    print("5. MULTI-NODE PROJECTION at realistic f in [0.6, 0.8]")
    print("=" * 78)
    print("   Using the cost-model ideal at the measured accept_len (per batch):")
    print("     speedup_ideal(f) = accept_len / (K*(1-f) + 1)")
    hdr5 = f"  {'f':>6}" + "".join(f"{'b'+str(b):>10}" for b in BATCHES)
    print(hdr5)
    for f in (0.6, 0.7, 0.8):
        row = f"  {f:>6.2f}"
        for b in BATCHES:
            als = [accept[(d, b)] for d in delays if accept.get((d, b))]
            al = sum(als) / len(als) if als else 3.0
            ideal = al / (K * (1 - f) + 1)
            row += f"{ideal:>10.3f}"
        print(row)
    print("  (also: measured-interpolated speedup at the a2a that maps to each f,")
    print("   per batch, from Section 2's f-map -- see the f column there.)")

    # ---- 6. Sanity ----
    print("\n" + "=" * 78)
    print("6. SANITY: 1.008x at a2a=0 b64 (native, piecewise K=2)")
    print("=" * 78)
    s0 = speedup.get((0, 64))
    ns0 = NS[0].get(64, {}).get("tok_s_mean")
    sp0 = SP[0].get(64, {}).get("tok_s_mean")
    al0 = accept.get((0, 64))
    print(f"  no-spec b64 a2a0 = {ns0:.1f} (Phase-42 reuse; expect 1857.6)")
    print(f"  spec_pw b64 a2a0 = {sp0:.1f}  accept_len={al0:.3f}")
    print(f"  speedup = {s0:.4f}  (target ~1.008x)")


if __name__ == "__main__":
    main()
