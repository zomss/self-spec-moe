"""Analyze W7 micro-bench dumps: reconstruct the spec cycle and attribute it.

Reads mb_spec_*.json and mb_nospec_*.json from the data dir and, for each
(batch, K, model) spec point with a matching nospec point, prints:

  K * t_draft_forward                 (fundamental draft compute)
  chain_overhead = t_draft_chain
                   - draft_forward_first - (K-1)*draft_forward
                                       (fixable CPU orchestration)
  t_verify                            (the verify forward)
  reconstructed cycle = t_draft_chain + t_verify
  ideal cycle (zero overhead) = K*t_draft_forward + t_verify
  implied speedup = accept_len * t_nospec_step / cycle

Run:
  .venv/bin/python w7_microbench_analyze.py [DATA_DIR]
"""
import glob
import json
import os
import sys

DATA = sys.argv[1] if len(sys.argv) > 1 else (
    "/data/smcho/ssm-mb/research/34_worldA_system/data"
)


def _mean(prof, label):
    if not prof:
        return None
    s = prof.get("summary", {})
    d = s.get(label)
    if not d:
        return None
    return d.get("mean_ms")


def load(mode):
    out = {}
    for f in sorted(glob.glob(os.path.join(DATA, f"mb_{mode}_*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        # Key must include K so multiple K at the same (tag, batch) don't collide.
        key = (d["tag"], d["batch"], d.get("a2a_us"), d["K"])
        out[key] = d
    return out


def main():
    spec = load("spec")
    nospec = load("nospec")

    # nospec indexed by (tag, batch, a2a) -- K is 0 / irrelevant.
    nospec_by_bk = {(t, b, a): d for (t, b, a, _k), d in nospec.items()}

    rows = []
    for (tag, batch, a2a, _kk), d in sorted(spec.items()):
        K = d["K"]
        prof = d.get("rank0_profile")
        df_first = _mean(prof, "draft_forward_first")
        df = _mean(prof, "draft_forward")
        chain = _mean(prof, "draft_chain")
        verify = _mean(prof, "verify")
        rr = d.get("run_result") or {}
        accept_len = rr.get("accept_len")

        # nospec match by (tag, batch, a2a)
        nd = nospec_by_bk.get((tag, batch, a2a))
        nstep = None
        nverify = None
        if nd:
            nstep = (nd.get("run_result") or {}).get("nospec_step_ms")
            nverify = _mean(nd.get("rank0_profile"), "verify")

        # K total draft forwards = 1 first + (K-1) loop
        k_fwd = None
        if df is not None and df_first is not None:
            k_fwd = df_first + (K - 1) * df
        chain_overhead = None
        if chain is not None and k_fwd is not None:
            chain_overhead = chain - k_fwd
        cycle = None
        if chain is not None and verify is not None:
            cycle = chain + verify
        ideal = None
        if k_fwd is not None and verify is not None:
            ideal = k_fwd + verify

        def spd(denom):
            if denom and accept_len and nstep:
                return accept_len * nstep / denom
            return None

        rows.append(dict(
            tag=tag, batch=batch, K=K, a2a=a2a, accept_len=accept_len,
            df_first=df_first, df=df, k_fwd=k_fwd, chain=chain,
            chain_overhead=chain_overhead, verify=verify, cycle=cycle,
            ideal=ideal, nstep=nstep, nverify=nverify,
            speedup_measured=spd(cycle), speedup_ideal=spd(ideal),
        ))

    def fmt(x, w=8):
        return ("%.3f" % x).rjust(w) if isinstance(x, (int, float)) else \
            str(x).rjust(w)

    hdr = ["tag", "b", "K", "a2a", "acc", "df1", "df", "Kfwd", "chain",
           "ovhd", "verify", "cycle", "ideal", "nstep", "spd_m", "spd_i"]
    print("  ".join(h.rjust(8) for h in hdr))
    for r in rows:
        print("  ".join([
            r["tag"].rjust(8), fmt(r["batch"]), fmt(r["K"]), fmt(r["a2a"]),
            fmt(r["accept_len"]), fmt(r["df_first"]), fmt(r["df"]),
            fmt(r["k_fwd"]), fmt(r["chain"]), fmt(r["chain_overhead"]),
            fmt(r["verify"]), fmt(r["cycle"]), fmt(r["ideal"]),
            fmt(r["nstep"]), fmt(r["speedup_measured"]),
            fmt(r["speedup_ideal"]),
        ]))

    # Attribution for each row (% of reconstructed cycle).
    print("\nAttribution (% of reconstructed cycle = chain + verify):")
    for r in rows:
        if not r["cycle"]:
            continue
        c = r["cycle"]
        kf = r["k_fwd"] or 0
        ov = r["chain_overhead"] or 0
        vf = r["verify"] or 0
        print(f"  {r['tag']} b{r['batch']} K{r['K']} a2a{r['a2a']}: "
              f"cycle={c:.2f}ms | K-forward={kf:.2f}ms ({100*kf/c:.0f}%) | "
              f"chain_overhead={ov:.2f}ms ({100*ov/c:.0f}%) | "
              f"verify={vf:.2f}ms ({100*vf/c:.0f}%)")


if __name__ == "__main__":
    main()
