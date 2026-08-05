#!/usr/bin/env python3
"""Score E0: the window-switching envelope on real data.

  S(regime, window) = toks(window) / toks(AR)          per-boot AR anchor
  konly             = the single best window, by aggregate  (today's C3)
  envelope          = per-regime max over windows           (any switcher's
                                                             upper bound)
  gate              = envelope / konly - 1  >= 1%  (pre-registered)

Regimes are split by whether the axis under test can act at all:

  discriminating  R4, R5, R5cot, R8   end ctx > 2048, crosses both windows
  partial         R1                  end ctx 1087, crosses 512 only
  null control    R6                  end ctx 318, crosses NEITHER

P8: the null control's envelope must be < half the discriminating group's.
R6 cannot benefit from window switching by construction, so whatever it
shows is this metric's noise floor.
"""
import json
import statistics as st
import sys
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
DATA = PHASE / "data"
WINDOWS = ["512", "2048", "none"]
GROUPS = {
    "discriminating": ["R4", "R5", "R5cot", "R8"],
    "partial": ["R1"],
    "null_control": ["R6"],
}


def hmean(v):
    return len(v) / sum(1.0 / x for x in v)


def load(arch, seed):
    arms = {}
    for w in ["off"] + WINDOWS:
        f = DATA / f"e0_{arch}_w{w}_s{seed}.json"
        if f.exists():
            arms[w] = json.loads(f.read_text())
    return arms


def score(arch, seed):
    arms = load(arch, seed)
    missing = [w for w in ["off"] + WINDOWS if w not in arms]
    if "off" in missing:
        print(f"[{arch} s{seed}] no AR anchor -- skipped")
        return None
    rids = [r for r in arms["off"]["regimes"]
            if all(r in arms[w]["regimes"] for w in arms)]

    S = {}          # regime -> window -> S
    for r in rids:
        ar = arms["off"]["regimes"][r]["toks"]
        S[r] = {w: arms[w]["regimes"][r]["toks"] / ar
                for w in WINDOWS if w in arms}

    def agg(sel, pick):
        v = [pick(S[r]) for r in sel if r in S]
        return st.mean(v) if v else None

    out = {"arch": arch, "seed": seed, "missing_arms": missing,
           "regimes": {}, "groups": {}}
    print(f"\n===== {arch} seed={seed} "
          f"({'all arms' if not missing else 'MISSING ' + ','.join(missing)}) =====")
    print(f"  {'regime':8s} {'AR tok/s':>9s} " +
          " ".join(f"{'w'+w:>8s}" for w in WINDOWS) +
          f" {'best':>6s} {'map says':>9s}")
    for r in rids:
        ar = arms["off"]["regimes"][r]["toks"]
        best_w = max(S[r], key=S[r].get)
        out["regimes"][r] = {
            "ar_toks": ar,
            "S": {w: round(s, 4) for w, s in S[r].items()},
            "best_window": best_w,
            "accept": {w: arms[w]["regimes"][r].get("accept") for w in S[r]},
            "batch": arms["off"]["regimes"][r]["batch"],
        }
        print(f"  {r:8s} {ar:9.1f} " +
              " ".join(f"{S[r].get(w, float('nan')):8.4f}" for w in WINDOWS) +
              f" {best_w:>6s}")

    for gname, sel in GROUPS.items():
        present = [r for r in sel if r in S]
        if not present:
            continue
        # konly: one window fixed for the whole deployment, chosen on this group
        konly_w = max(WINDOWS, key=lambda w: agg(present, lambda d: d.get(w, 0)))
        konly = agg(present, lambda d: d.get(konly_w, 0))
        env = agg(present, lambda d: max(d.values()))
        out["groups"][gname] = {
            "regimes": present,
            "konly_window": konly_w,
            "konly_S": round(konly, 4),
            "envelope_S": round(env, 4),
            "envelope_over_konly_pct": round((env / konly - 1) * 100, 2),
            "hmean_konly": round(hmean([S[r][konly_w] for r in present]), 4),
            "hmean_envelope": round(
                hmean([max(S[r].values()) for r in present]), 4),
        }
        g = out["groups"][gname]
        print(f"  [{gname:14s}] konly=w{konly_w} {konly:.4f}  "
              f"envelope {env:.4f}  -> {g['envelope_over_konly_pct']:+.2f}%")

    d = out["groups"].get("discriminating")
    n = out["groups"].get("null_control")
    if d and n:
        de, ne = d["envelope_over_konly_pct"], n["envelope_over_konly_pct"]
        out["P8_null_control_ok"] = bool(ne < de / 2)
        out["gate_pass"] = bool(de >= 1.0 and ne < de / 2)
        print(f"  P8 null control: R6 {ne:+.2f}% vs discriminating {de:+.2f}%"
              f"  -> {'OK' if out['P8_null_control_ok'] else 'FAILS (noise)'}")
        print(f"  GATE (>= +1% and P8 ok): "
              f"{'PASS -> build E1' if out['gate_pass'] else 'FAIL -> do not build E1'}")
    return out


def main():
    seeds = [int(x) for x in (sys.argv[1:] or ["0"])]
    res = []
    for arch in ("dense", "llama"):
        for s in seeds:
            r = score(arch, s)
            if r:
                res.append(r)
    if res:
        (DATA / "e0_envelope.json").write_text(json.dumps(
            {"windows": WINDOWS, "groups": GROUPS, "results": res}, indent=1))
        print(f"\nwrote {DATA}/e0_envelope.json")


if __name__ == "__main__":
    main()
