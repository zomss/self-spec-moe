"""Phase-42 crossover + draft-shielded projection from the measured sweep.

Inputs are the measured (nospec, spec) tok/s at each (delay_us, batch), hardcoded
from the run logs (single source of truth; the raw JSONs are in data/).

Two analyses:

1) MEASURED speedup vs delay, and interpolated/extrapolated crossover (speedup=1.0).
   In the measured curve BOTH the no-spec verify AND the comm-free draft eat the
   injected per-collective sleep (the draft's MoE routes through naive_dp_ep ->
   get_ep_group().dispatch/combine, whose _emulate_exposed_a2a_delay fires even on
   the comm-free local-route branch). So the measured curve UNDER-states World A.

2) DRAFT-SHIELDED projection (the true multi-node comm-free semantics): a real
   comm-free draft on a multi-node fabric pays ZERO inter-node A2A latency; only the
   full-EP VERIFY pays it. We remove the draft's injected-delay share analytically.

   Model per-decode-batch wall time as linear in delay d:
     T_mode(d) = T0_mode + S_mode * d
   where S_mode = (#injected collectives that run the sleep, per timed batch) *
   (their serialization on the stream). We estimate S_mode by least-squares on
   1/tok_s (proportional to per-batch decode time up to the constant token count).

   The draft-shielded spec time removes the draft's collective share. The draft runs
   K forwards/step vs 1 verify forward; each full MoE forward issues the same #
   collectives C. Over a step the injected-delay collectives are ~ (K*C + C) for
   measured spec vs C for shielded spec (verify only). So the shielded delay-slope is
   S_spec_shielded = S_spec * C / (K*C + C) = S_spec / (K + 1)  (first-order).
   T0 (the d=0 compute) is unchanged (the sleep is 0 at d=0). We then recompute
   shielded spec tok/s(d) = tok_s_spec(0) reference scaled by T0/(T0 + S_shield*d).
"""

import glob
import json
import os

DATA = os.environ.get(
    "W7_OUT", "/data/smcho/self-spec-moe/research/42_comm_sweep/data"
)
K = 4
BATCHES = [32, 64, 128]
DELAYS_ALL = [0, 100, 250, 500, 1000]
# Measured accept_len per batch (stable across delays; ~4.68-4.76).
ACCEPT = {32: 4.75, 64: 4.69, 128: 4.67}


def _rows(path):
    with open(path) as f:
        d = json.load(f)
    out = {}
    for r in d.get("results", []):
        if isinstance(r, dict) and "batch" in r and "error" not in r:
            out[r["batch"]] = r["tok_s_mean"]
    return out


def _load_M():
    M = {}
    for a in DELAYS_ALL:
        ns, sp = {}, {}
        for p in glob.glob(os.path.join(DATA, f"w7fp8_sweep_a2a{a}_nospec_*_nospec.json")):
            ns.update(_rows(p))
        for p in glob.glob(os.path.join(DATA, f"w7fp8_sweep_a2a{a}_fp8_spec_*_K*.json")):
            sp.update(_rows(p))
        M[a] = {b: (ns.get(b), sp.get(b)) for b in BATCHES}
    return M


# Measured tok/s (system throughput, output tokens): delay_us -> {batch -> (ns, sp)}
M = _load_M()


def _lin_fit(xs, ys):
    """Least squares y = a + b x. Returns (a, b)."""
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
    """pts: sorted [(delay, speedup)]. Return (crossover, kind)."""
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if (y0 - 1.0) * (y1 - 1.0) <= 0 and y1 != y0:
            return x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0), "interp"
    # extrapolate from last two
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0
    if slope > 1e-6:
        return x1 + (1.0 - y1) / slope, "extrap"
    return None, "none"


def main():
    delays = [d for d in DELAYS_ALL if M[d][64][0] is not None]
    print("=== MEASURED speedup table (spec/nospec) ===")
    print(f"{'a2a_us':>7}" + "".join(f"{'b'+str(b):>10}" for b in BATCHES))
    for d in delays:
        row = f"{d:>7}"
        for b in BATCHES:
            ns, sp = M[d][b]
            row += f"{(sp/ns):>10.3f}" if (ns and sp) else f"{'-':>10}"
        print(row)

    print("\n=== MEASURED crossover (speedup=1.0) per batch ===")
    for b in BATCHES:
        pts = [(d, M[d][b][1] / M[d][b][0]) for d in delays
               if M[d][b][0] and M[d][b][1]]
        pts.sort()
        cx, kind = interp_crossover(pts)
        mx = max(y for _, y in pts)
        if cx is None:
            print(f"  b{b}: NO crossover <= {max(delays)}us; plateau max "
                  f"speedup {mx:.3f}; not increasing.")
        else:
            print(f"  b{b}: crossover ~= {cx:.0f}us ({kind}); "
                  f"max measured speedup {mx:.3f}.")

    # Fit per-batch delay slope on inverse-throughput (proportional to decode
    # time per fixed token count): time = a + s*d.
    print("\n=== Measured delay-slopes (inverse-throughput, s/tok-batch per us) ===")
    print(f"{'batch':>6}{'a_ns':>10}{'s_ns':>12}{'a_sp':>10}{'s_sp':>12}"
          f"{'s_sp/s_ns':>10}{'accept':>8}")
    fit = {}
    for b in BATCHES:
        ds = [d for d in delays if M[d][b][0] and M[d][b][1]]
        a_ns, s_ns = _lin_fit(ds, [1.0 / M[d][b][0] for d in ds])
        a_sp, s_sp = _lin_fit(ds, [1.0 / M[d][b][1] for d in ds])
        al = ACCEPT.get(b, 4.68)
        fit[b] = (a_ns, s_ns, a_sp, s_sp, al)
        print(f"{b:>6}{a_ns:>10.2e}{s_ns:>12.3e}{a_sp:>10.2e}{s_sp:>12.3e}"
              f"{s_sp / s_ns:>10.3f}{al:>8.2f}")
    print("  (s_sp/s_ns << 1 => spec already far less delay-sensitive than nospec:")
    print("   the comm-free draft's injected-sleep contribution to spec is small;")
    print("   spec's residual delay cost is the VERIFY, amortized over accept_len.)")

    # DRAFT-SHIELDED projection. Model: a true comm-free multi-node draft pays ZERO
    # inter-node A2A latency; ONLY the full-EP verify pays it. The verify is a
    # full-EP forward (same collective count as a nospec token) run ONCE per step
    # and amortized over accept_len output tokens. So shielded spec delay-slope =
    # s_ns / accept_len (assumption-light; uses the measured nospec slope directly).
    # d=0 compute (a_sp) is unchanged.
    print("\n=== DRAFT-SHIELDED projection (true comm-free multi-node: draft pays 0 "
          "A2A; verify slope = s_ns/accept_len) ===")
    print(f"{'a2a_us':>7}" + "".join(f"{'b'+str(b):>10}" for b in BATCHES))
    shielded = {b: {} for b in BATCHES}
    for b in BATCHES:
        a_ns, s_ns, a_sp, s_sp, al = fit[b]
        s_sh = s_ns / al
        for d in delays + [1500, 2000]:
            t_ns = a_ns + s_ns * d
            t_sp = a_sp + s_sh * d
            shielded[b][d] = (t_sp, t_ns, t_ns / t_sp)
        shielded[b]['fit'] = (a_ns, s_ns, a_sp, s_sp, s_sh)
    for d in delays:
        row = f"{d:>7}"
        for b in BATCHES:
            row += f"{shielded[b][d][2]:>10.3f}"
        print(row)

    print("\n=== Projected speedup at realistic multi-node delays (shielded) ===")
    print(f"{'delay_us':>9}" + "".join(f"{'b'+str(b):>10}" for b in BATCHES))
    for d in (300, 400, 500):
        row = f"{d:>9}"
        for b in BATCHES:
            a_ns, s_ns, a_sp, s_sp, s_sh = shielded[b]['fit']
            row += f"{(a_ns + s_ns * d) / (a_sp + s_sh * d):>10.3f}"
        print(row)

    print("\n=== DRAFT-SHIELDED crossover per batch ===")
    for b in BATCHES:
        pts = [(d, shielded[b][d][2]) for d in delays]
        pts.sort()
        cx, kind = interp_crossover(pts)
        a_ns, s_ns, a_sp, s_sp, s_sh = shielded[b]['fit']
        # analytic crossover: a_ns + s_ns*d = a_sp + s_sh*d  ->  d* = (a_sp-a_ns)/(s_ns-s_sh)
        d_star = (a_sp - a_ns) / (s_ns - s_sh) if (s_ns - s_sh) != 0 else None
        mx = max(y for _, y in pts)
        cxs = f"{cx:.0f}us ({kind})" if cx else "none"
        dss = f"{d_star:.0f}us" if (d_star and d_star > 0) else "none/neg"
        print(f"  b{b}: shielded max speedup {mx:.3f}; grid-crossover {cxs}; "
              f"analytic crossover d*={dss}")


if __name__ == "__main__":
    main()
