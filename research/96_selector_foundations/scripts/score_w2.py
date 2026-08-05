#!/usr/bin/env python3
"""W2 scorer: both currencies per (arch, regime, window) + episode audit.

For each cell against the arch's AR anchor:
  S_e2e   = e2e rate ratio          (deployment currency)
  S_dec   = per-request decode-rate ratio (map's currency, live proxy)
  phi     = prefill share of request time (bridge)
  queue_s = e2e_lat - prefill - decode    (derived; definitional gap audit)

Episode rejection (results_w1.md protocol): reference = max e2e round in
the cell; reject rounds >5% below; a cell must keep >=2 rounds. Rejection
counts are printed with every cell.
"""
import json
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "data" / "w2"

ARCHS = ["dense", "llama"]
WINDOWS = ["off", "512", "2048"]
REGIMES = ["R4", "R5", "R5cot", "R8", "R1", "R6"]


def load(arch, win):
    p = D / f"w2_{arch}_w{win}_notune_s0.json"
    return json.loads(p.read_text())["regimes"] if p.exists() else None


def cell_stats(reg):
    """Episode-rejected medians over rounds."""
    rounds = reg["rounds"]
    ref = max(r["e2e_rate"] for r in rounds)
    keep = [r for r in rounds if r["e2e_rate"] >= 0.95 * ref]
    rej = len(rounds) - len(keep)
    med = sorted(keep, key=lambda r: r["e2e_rate"])[len(keep) // 2]
    return med, rej, len(keep)


for arch in ARCHS:
    tabs = {w: load(arch, w) for w in WINDOWS}
    if not tabs["off"]:
        print(f"{arch}: no AR anchor yet")
        continue
    print(f"\n{'=' * 100}\n{arch.upper()}  (episode-rejected medians; "
          f"rej = rejected rounds)\n{'=' * 100}")
    print(f"{'regime':7s} {'AR e2e':>8s} {'phi':>6s} | "
          f"{'S_e2e':>7s} {'S_dec':>7s} {'phi':>6s} {'acc':>5s} {'rej':>3s} | "
          f"{'S_e2e':>7s} {'S_dec':>7s} {'phi':>6s} {'acc':>5s} {'rej':>3s} | "
          f"{'currency gap w512':>18s}")
    print(f"{'':7s} {'':>8s} {'':>6s} | {'w512':>36s} | {'w2048':>36s} |")
    for rid in REGIMES:
        if rid not in tabs["off"]:
            continue
        ar, ar_rej, _ = cell_stats(tabs["off"][rid])
        row = f"{rid:7s} {ar['e2e_rate']:>8.1f} {ar['phi_req']:>6.3f} |"
        gap = ""
        for w in ("512", "2048"):
            if not tabs[w] or rid not in tabs[w]:
                row += f" {'—':>36s} |"
                continue
            sp, rej, kept = cell_stats(tabs[w][rid])
            s_e2e = sp["e2e_rate"] / ar["e2e_rate"]
            s_dec = (sp["decode_rate_req"] / ar["decode_rate_req"]
                     if sp["decode_rate_req"] and ar["decode_rate_req"]
                     else float("nan"))
            row += (f" {s_e2e:>7.3f} {s_dec:>7.3f} {sp['phi_req']:>6.3f} "
                    f"{sp['accept'] or 0:>5.2f} {rej + ar_rej:>3d} |")
            if w == "512":
                gap = f"{(s_dec - s_e2e) * 100:>+17.1f}%"
        print(row + gap)

# queue-time audit: how much request time is neither prefill nor decode
print(f"\n{'-' * 60}\nqueue/gap audit (AR arms, per-regime, seconds "
      f"per round):")
for arch in ARCHS:
    t = load(arch, "off")
    if not t:
        continue
    for rid in REGIMES:
        if rid not in t:
            continue
        med, _, _ = cell_stats(t[rid])
        q = med["e2e_lat_sum_s"] - med["prefill_time_s"] - med["decode_time_s"]
        print(f"  {arch:6s} {rid:6s} e2e_lat {med['e2e_lat_sum_s']:>8.1f} "
              f"prefill {med['prefill_time_s']:>7.2f} "
              f"decode {med['decode_time_s']:>8.1f} "
              f"other {q:>7.2f} ({q / max(med['e2e_lat_sum_s'], 1e-9):>6.1%})")
