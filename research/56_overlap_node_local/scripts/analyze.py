"""Phase 56 Stage A analysis: cycle-time delta + hidden fraction.

Reads the harness JSONs (w72n_*.json) for paired AHEAD=0/1 runs and, when
present, the profiler dumps (prof_*/ *.json) for component attribution.

cycle_ms = 1000 * batch * accept_len / tok_s   (per-rank decode cycle)
delta_ms = cycle_on - cycle_off                (exposed ahead cost)
hidden   = (T_ahead_serial - delta) / T_ahead_serial
floor    = max(0, T_ahead_serial - T_verify)   (bounded-overlap best case)
"""
import glob
import json
import os
import sys

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def load(tagsub: str):
    out = {}
    for p in glob.glob(os.path.join(DATA, f"w72n_*{tagsub}*_spec_*_K2.json")):
        j = json.load(open(p))
        for r in j.get("results") or []:
            if "error" in r:
                out[r["batch"]] = r
                continue
            out[r["batch"]] = r
    return out


def prof_summary(profdir: str):
    best = {}
    for p in glob.glob(os.path.join(profdir, "*.json")):
        j = json.load(open(p))
        s = j.get("summary", j)
        n = sum(v.get("n_total", 0) or 0 for v in s.values() if isinstance(v, dict))
        if not best or n > best[0]:
            best = (n, s)
    return best[1] if best else {}


def main():
    off_tag, on_tag = sys.argv[1], sys.argv[2]
    off, on = load(off_tag), load(on_tag)
    print(f"{'b':>4} {'off tok/s':>10} {'on tok/s':>10} {'ratio':>6} "
          f"{'off cyc ms':>10} {'on cyc ms':>10} {'delta ms':>9} "
          f"{'acc off':>7} {'acc on':>7}")
    for b in sorted(set(off) & set(on)):
        o, n = off[b], on[b]
        if "error" in o or "error" in n:
            print(f"{b:>4} ERROR off={o.get('error','')[:40]} "
                  f"on={n.get('error','')[:40]}")
            continue
        co = 1000 * b * (o.get("accept_len") or 0) / o["tok_s_mean"]
        cn = 1000 * b * (n.get("accept_len") or 0) / n["tok_s_mean"]
        print(f"{b:>4} {o['tok_s_mean']:>10.1f} {n['tok_s_mean']:>10.1f} "
              f"{n['tok_s_mean']/o['tok_s_mean']:>6.3f} {co:>10.1f} "
              f"{cn:>10.1f} {cn-co:>9.1f} "
              f"{o.get('accept_len') or 0:>7.3f} {n.get('accept_len') or 0:>7.3f}")
    for tag in (off_tag, on_tag):
        for d in glob.glob(os.path.join(DATA, f"prof_*{tag}*")):
            s = prof_summary(d)
            if s:
                print(f"\nprofiler {os.path.basename(d)}:")
                for k in ("verify", "draft_chain", "draft_forward",
                          "draft_forward_first", "ahead_chain", "shadow_chain"):
                    if k in s and s[k].get("mean_ms") is not None:
                        print(f"  {k:>20}: {s[k]['mean_ms']:8.2f} ms "
                              f"(n={s[k]['n']})")


if __name__ == "__main__":
    main()
