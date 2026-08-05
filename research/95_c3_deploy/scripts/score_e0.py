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

TWO BIAS CORRECTIONS, because `envelope` is a max over three noisy arms --
the same winner's-curse structure C2 measured on its own search:

1. NULL-CONTROL SUBTRACTION. R6's envelope is pure max-of-noise (the window
   cannot act there), so it estimates the upward bias directly. The headline
   is the CORRECTED envelope = discriminating - null, not the raw max.

2. CROSS-SEED SELECTION (`--cross-seed a b`). Choose each regime's window on
   seed a, then score that choice on seed b. Selection and evaluation land on
   independent draws, so the winner's curse is removed rather than estimated
   -- the same fix C2's 2-sample truth scoring applied to its rankers. This
   is the preferred estimate whenever two seeds exist.

Note the gate metric is AR-INDEPENDENT: envelope/konly is a ratio of two spec
arms that share the same AR denominator, so boot-to-boot AR variance cancels
exactly and cannot move the gate decision (it moves only the absolute S).
"""
import json
import statistics as st
import sys
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
DATA = PHASE / "data"
# The switching set is {512, 2048}, NOT {512, 2048, none}. Full-context
# drafting is not realizable in the deployed stack: VLLM_SELF_SPEC_DRAFT_FULLCG
# raises "requires VLLM_SELF_SPEC_DRAFT_KV_WINDOW > 0 (the scratchpad
# materialises the sinks+window key set)". Phase 94 handled this by running
# w-none PLAIN and disclosing the realization split (measured worth -0.5%
# mean); running it plain HERE would compare across stacks inside the very
# metric under test, so the none arm is dropped and the constraint recorded.
WINDOWS = ["512", "2048"]
# GROUPING, corrected by measurement (2026-08-05). P8 registered R6 as a null
# control on the premise that a window cannot matter where it does not bind.
# MEASURED FALSE: at R6 (end ctx 318) dense runs +5.0% faster on w512 with
# accept IDENTICAL (4.733 vs 4.734), llama +12.0%. The mechanism is
# _scratchpad_n_kept_blocks: the gather materialises
# n_sink + ceil((window+K)/block_size) blocks EVERY draft step regardless of
# the live context length, so the window sets a fixed per-draft-step COST
# floor. R6 discriminates through cost even though it cannot through accept.
#
# Consequences: (a) there is no window-inert regime, so no valid null control
# exists and P8 is retired as refuted; (b) bias control falls entirely to
# cross-seed selection; (c) all_regimes is the honest aggregate.
GROUPS = {
    "all_regimes": ["R4", "R5", "R5cot", "R8", "R1", "R6"],
    "accept_binding": ["R4", "R5", "R5cot", "R8"],   # window binds semantically
    "cost_only": ["R1", "R6"],                       # window acts via cost only
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

    a = out["groups"].get("all_regimes")
    if a:
        out["raw_envelope_pct"] = a["envelope_over_konly_pct"]
        print(f"  RAW envelope (all regimes) = "
              f"{a['envelope_over_konly_pct']:+.2f}%  -- BIASED UP (per-regime "
              "argmax selected on the same draw it is scored on).")
        print("  No null control exists (P8 refuted): the gate is decided by "
              "cross-seed selection, not by this number.")
    return out


def cross_seed(arch, sa, sb):
    """Pick each regime's window on seed sa, score that pick on seed sb.

    Selection and evaluation on independent draws -> no winner's curse.
    """
    A, B = load(arch, sa), load(arch, sb)
    if "off" not in A or "off" not in B:
        return None
    rids = [r for r in A["off"]["regimes"]
            if all(r in A.get(w, {"regimes": {}})["regimes"] and
                   r in B.get(w, {"regimes": {}})["regimes"] for w in WINDOWS)]
    if not rids:
        return None
    SA = {r: {w: A[w]["regimes"][r]["toks"] / A["off"]["regimes"][r]["toks"]
              for w in WINDOWS} for r in rids}
    SB = {r: {w: B[w]["regimes"][r]["toks"] / B["off"]["regimes"][r]["toks"]
              for w in WINDOWS} for r in rids}
    out = {"arch": arch, "select_seed": sa, "eval_seed": sb, "groups": {}}
    print(f"\n----- {arch}: cross-seed (select s{sa}, evaluate s{sb}) -----")
    for gname, sel in GROUPS.items():
        present = [r for r in sel if r in SA]
        if not present:
            continue
        konly_w = max(WINDOWS,
                      key=lambda w: st.mean([SA[r][w] for r in present]))
        konly = st.mean([SB[r][konly_w] for r in present])
        picks = {r: max(SA[r], key=SA[r].get) for r in present}
        switched = st.mean([SB[r][picks[r]] for r in present])
        out["groups"][gname] = {
            "konly_window": konly_w, "konly_S": round(konly, 4),
            "switched_S": round(switched, 4),
            "gain_pct": round((switched / konly - 1) * 100, 2),
            "picks": picks,
        }
        print(f"  [{gname:14s}] konly=w{konly_w} {konly:.4f} -> switched "
              f"{switched:.4f}  {out['groups'][gname]['gain_pct']:+.2f}%"
              f"   picks={picks}")
    d, n = out["groups"].get("discriminating"), out["groups"].get("null_control")
    if d and n:
        out["corrected_gain_pct"] = round(d["gain_pct"] - n["gain_pct"], 2)
        print(f"  CORRECTED cross-seed gain = {out['corrected_gain_pct']:+.2f}%"
              "   <- unbiased estimate of the switching headroom")
    return out


def main():
    args = sys.argv[1:]
    xs = None
    if "--cross-seed" in args:
        i = args.index("--cross-seed")
        xs = (int(args[i + 1]), int(args[i + 2]))
        args = args[:i] + args[i + 3:]
    seeds = [int(x) for x in (args or ["0"])]
    res, cross = [], []
    for arch in ("dense", "llama"):
        for s in seeds:
            r = score(arch, s)
            if r:
                res.append(r)
        if xs:
            c = cross_seed(arch, *xs)
            if c:
                cross.append(c)
            c2 = cross_seed(arch, xs[1], xs[0])   # both directions
            if c2:
                cross.append(c2)
    if res:
        (DATA / "e0_envelope.json").write_text(json.dumps(
            {"windows": WINDOWS, "groups": GROUPS, "results": res,
             "cross_seed": cross}, indent=1))
        print(f"\nwrote {DATA}/e0_envelope.json")


if __name__ == "__main__":
    main()
