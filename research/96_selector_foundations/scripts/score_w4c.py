#!/usr/bin/env python3
"""W4c scorer: R per (content, style) from matched off/spec boots."""
import json
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "data" / "w4"
K = 4

off = json.loads((D / "w4c_off.json").read_text())
spec = json.loads((D / "w4c_spec.json").read_text())


def rate(entry, which):
    """Extract a scalar rate from a style entry (episode-rejected)."""
    if "rate" in entry:                     # compile style
        return entry["rate"], entry.get("accept")
    rr = [r for r in entry["rounds"] if r.get(which)]
    ref = max(r[which] for r in rr)
    keep = sorted(r[which] for r in rr if r[which] >= 0.95 * ref)
    med = keep[len(keep) // 2]
    accs = [r["accept"] for r in entry["rounds"] if r.get("accept")]
    return med, (sum(accs) / len(accs) if accs else None)


print(f"{'content':6s} {'style':13s} {'basis':8s} | {'off':>7s} {'spec':>7s} "
      f"{'S':>6s} {'tau':>6s} | {'R':>6s}")
results = {}
for cname in ("cmap", "r5"):
    o_styles = {e["style"]: e for e in off["contents"][cname]}
    s_styles = {e["style"]: e for e in spec["contents"][cname]}
    for style in ("compile-N160", "compile-N512", "serving", "serving-ie"):
        bases = (("rate",),) if style.startswith("compile") else (
            ("e2e_rate",), ("dec_rate_req",))
        for (basis,) in bases:
            o_r, _ = rate(o_styles[style], basis)
            s_r, tau = rate(s_styles[style], basis)
            s = s_r / o_r
            r = ((tau / s) - 1) / K if tau else float("nan")
            results[(cname, style, basis)] = r
            print(f"{cname:6s} {style:13s} {basis:8s} | {o_r:7.1f} "
                  f"{s_r:7.1f} {s:6.3f} {tau or 0:6.3f} | {r:6.3f}")

print("\nverdicts:")
rc160 = results.get(("cmap", "compile-N160", "rate"))
rr160 = results.get(("r5", "compile-N160", "rate"))
rcs = results.get(("cmap", "serving-ie", "dec_rate_req"))
rrs = results.get(("r5", "serving", "dec_rate_req"))
rc512 = results.get(("cmap", "compile-N512", "rate"))
if all(v == v for v in (rc160, rr160, rcs, rrs, rc512)):
    proto = (rc160 - rcs) / rc160
    content = abs(rc160 - rr160) / rc160
    print(f"  protocol axis (cmap: N160 vs serving-ie dec): "
          f"{proto:+.1%}  | content axis (N160 cmap vs r5): {content:.1%}")
    print(f"  P-W4c1 protocol dominates: "
          f"{'CONFIRMED' if proto > 0.10 > content else 'REFUTED/MIXED'}")
    amort = (rc160 - rc512) / rc160
    print(f"  P-W4c2 amortization N160->N512: {amort:+.1%} "
          f"({'toward serving' if amort > 0.05 else 'FLAT -- not startup'})")
sv = results.get(("r5", "serving", "dec_rate_req"))
si = results.get(("r5", "serving-ie", "dec_rate_req"))
if sv and si and sv == sv and si == si:
    print(f"  P-W4c3 EOS effect on R (r5): {abs(sv - si) / sv:.1%} "
          f"({'not the mechanism' if abs(sv - si) / sv < 0.05 else 'MATTERS'})")
