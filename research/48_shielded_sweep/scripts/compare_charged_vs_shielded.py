"""Phase 48: direct charged-vs-shielded comparison.

speedup = spec tok/s / Phase-42 no-spec tok/s at each (a2a_us, batch).
  charged  = Phase 47 spec data (draft paid the emulated A2A sleep)
  shielded = Phase 48 spec data (gate fixed; draft pays no emulated A2A)
Also prints the Phase-47 analyzer's *model*-shielded prediction alongside, so
the rerun validates the model (measured-shielded ~= model-shielded).
"""

import glob
import json
import os

REPO = "/data/smcho/self-spec-moe"
NOSPEC = f"{REPO}/research/42_comm_sweep/data"
CHARGED = f"{REPO}/research/47_sweep2/data"
SHIELDED = f"{REPO}/research/48_shielded_sweep/data"
DELAYS = [0, 100, 250, 500, 1000]
BATCHES = [32, 64, 128]
K = 2


def rows(pattern):
    out = {}
    for p in glob.glob(pattern):
        with open(p) as f:
            d = json.load(f)
        for r in d.get("results", []):
            if isinstance(r, dict) and "batch" in r and "error" not in r:
                out[r["batch"]] = r
    return out


def lin_fit(xs, ys):
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    den = n * sxx - sx * sx
    b = (n * sxy - sx * sy) / den if den else 0.0
    return (sy - b * sx) / n, b


def main():
    ns = {a: rows(f"{NOSPEC}/w7fp8_sweep_a2a{a}_nospec_*_nospec.json") for a in DELAYS}
    ch = {a: rows(f"{CHARGED}/w7fp8_sweep2_a2a{a}_fp8_spec_*_K{K}.json") for a in DELAYS}
    # Phase 47's clean 5-iter rerun for the noisy a2a=0 point.
    ch[0].update(rows(f"{CHARGED}/w7fp8_sweep2rerun_a2a0_fp8_spec_*_K{K}.json"))
    sh = {a: rows(f"{SHIELDED}/w7fp8_sweep2_a2a{a}_fp8_spec_*_K{K}.json") for a in DELAYS}

    print(f"{'a2a_us':>7} {'batch':>6} {'nospec':>9} "
          f"{'charged':>9} {'su_ch':>7} {'shielded':>9} {'su_sh':>7} "
          f"{'model_sh':>9} {'acc_sh':>7}")
    # Model-shielded per Phase-47 analyzer: T_sp(d) = a_sp + (s_ns/accept)*d,
    # with a_sp from the charged fit at d=0 and s_ns from the no-spec fit.
    fits = {}
    for b in BATCHES:
        ds = [a for a in DELAYS if ns[a].get(b) and ch[a].get(b)]
        a_ns, s_ns = lin_fit(ds, [1.0 / ns[a][b]["tok_s_mean"] for a in ds])
        a_sp, _ = lin_fit(ds, [1.0 / ch[a][b]["tok_s_mean"] for a in ds])
        als = [ch[a][b].get("accept_len") for a in ds if ch[a][b].get("accept_len")]
        al = sum(als) / len(als) if als else 2.9
        fits[b] = (a_ns, s_ns, a_sp, al)

    for a in DELAYS:
        for b in BATCHES:
            n = ns[a].get(b, {}).get("tok_s_mean")
            c = ch[a].get(b, {}).get("tok_s_mean")
            s = sh[a].get(b, {}).get("tok_s_mean")
            acc = sh[a].get(b, {}).get("accept_len")
            a_ns, s_ns, a_sp, al = fits[b]
            t_ns, t_sh = a_ns + s_ns * a, a_sp + (s_ns / al) * a
            model = t_ns / t_sh if t_sh > 0 else 0.0
            print(f"{a:>7} {b:>6} {n or 0:>9.1f} "
                  f"{c or 0:>9.1f} {(c / n if c and n else 0):>7.3f} "
                  f"{s or 0:>9.1f} {(s / n if s and n else 0):>7.3f} "
                  f"{model:>9.3f} {acc or 0:>7.2f}")
        print()


if __name__ == "__main__":
    main()
