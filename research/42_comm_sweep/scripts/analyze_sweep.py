"""Aggregate Phase-42 comm-sweep JSONs into the speedup-vs-A2A-delay table.

Reads research/42_comm_sweep/data/w7fp8_sweep_a2a*.json (written by
w7_fp8_timing.py) and prints, per (a2a_delay_us, batch): nospec tok/s, spec
tok/s, accept_len, speedup = spec/nospec. Then interpolates the crossover delay
where speedup == 1.0 for each batch.

Speedup = spec tok/s / nospec tok/s at the SAME (a2a_us, batch).
"""
import glob
import json
import os

DATA = os.environ.get(
    "W7_OUT", "/data/smcho/self-spec-moe/research/42_comm_sweep/data"
)
DELAYS = [int(x) for x in os.environ.get("DELAYS", "0 100 250 500 1000").split()]
BATCHES = [int(x) for x in os.environ.get("BATCHES", "32,64,128").split(",")]


def load(path):
    with open(path) as f:
        return json.load(f)


def rows_of(path):
    d = load(path)
    out = {}
    for r in d.get("results", []):
        if isinstance(r, dict) and "batch" in r and "error" not in r:
            out[r["batch"]] = r
    return out


def _merge(paths):
    m = {}
    for p in paths:
        m.update(rows_of(p))
    return m


def nospec_rows(a2a):
    # w7fp8_sweep_a2a{a2a}_nospec_{suffix}_nospec.json ; suffix carries a2a tag.
    pats = [
        os.path.join(DATA, f"w7fp8_sweep_a2a{a2a}_nospec_*_nospec.json"),
        os.path.join(DATA, f"w7fp8_sweep_a2a{a2a}_nospec_*.json"),
    ]
    files = []
    for pat in pats:
        files += glob.glob(pat)
    # prefer the per-K (nospec) files (have per-batch results), skip the summary
    files = [f for f in files if "_nospec.json" in f]
    return _merge(files)


def spec_rows(a2a, k=None):
    # w7fp8_sweep_a2a{a2a}_fp8_spec_{suffix}_K{K}.json
    pat = os.path.join(DATA, f"w7fp8_sweep_a2a{a2a}_fp8_spec_*_K*.json")
    files = glob.glob(pat)
    if k is not None:
        files = [f for f in files if f.endswith(f"_K{k}.json")]
    return _merge(files)


def main():
    print(f"data dir: {DATA}")
    print(f"delays: {DELAYS}  batches: {BATCHES}\n")

    # table: rows per (a2a, batch)
    table = {}  # (a2a,batch) -> dict
    for a2a in DELAYS:
        nrows = nospec_rows(a2a)
        srows = spec_rows(a2a)
        for b in BATCHES:
            nr = nrows.get(b)
            sr = srows.get(b)
            ntps = nr["tok_s_mean"] if nr else None
            stps = sr["tok_s_mean"] if sr else None
            al = sr.get("accept_len") if sr else None
            sp = (stps / ntps) if (ntps and stps) else None
            table[(a2a, b)] = dict(
                nospec=ntps, spec=stps, accept_len=al, speedup=sp,
                nsuspect=(nr or {}).get("suspect"),
                ssuspect=(sr or {}).get("suspect"),
            )

    hdr = f"{'a2a_us':>7} {'batch':>6} {'nospec':>10} {'spec':>10} {'accept':>7} {'speedup':>8} {'flag':>6}"
    print(hdr)
    print("-" * len(hdr))
    for a2a in DELAYS:
        for b in BATCHES:
            t = table[(a2a, b)]
            ns = f"{t['nospec']:.1f}" if t['nospec'] else "  -  "
            ss = f"{t['spec']:.1f}" if t['spec'] else "  -  "
            al = f"{t['accept_len']:.2f}" if t['accept_len'] else "  -  "
            sp = f"{t['speedup']:.3f}" if t['speedup'] else "  -  "
            flag = "S" if (t['nsuspect'] or t['ssuspect']) else ""
            print(f"{a2a:>7} {b:>6} {ns:>10} {ss:>10} {al:>7} {sp:>8} {flag:>6}")
        print()

    # crossover interpolation per batch
    print("=== Crossover delay (speedup == 1.0), linear interp per batch ===")
    for b in BATCHES:
        pts = [(a2a, table[(a2a, b)]['speedup']) for a2a in DELAYS
               if table[(a2a, b)]['speedup'] is not None]
        pts.sort()
        cross = None
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if (y0 - 1.0) * (y1 - 1.0) <= 0 and y1 != y0:
                cross = x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0)
                break
        if cross is not None:
            print(f"  batch={b}: crossover ~= {cross:.0f} us/collective  "
                  f"(bracketed by {pts})")
        elif pts:
            # extrapolate from last two points if trending up
            if len(pts) >= 2:
                (x0, y0), (x1, y1) = pts[-2], pts[-1]
                slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0
                if slope > 0:
                    xstar = x1 + (1.0 - y1) / slope
                    print(f"  batch={b}: NO crossover <= {max(DELAYS)}us; "
                          f"max speedup {max(y for _, y in pts):.3f}; "
                          f"linear-extrapolated crossover ~= {xstar:.0f} us "
                          f"(slope {slope:.2e}/us; EXTRAPOLATION, treat as lower bound)")
                else:
                    print(f"  batch={b}: NO crossover; speedup not increasing "
                          f"with delay ({pts})")
            else:
                print(f"  batch={b}: insufficient points {pts}")
        else:
            print(f"  batch={b}: no data")


if __name__ == "__main__":
    main()
