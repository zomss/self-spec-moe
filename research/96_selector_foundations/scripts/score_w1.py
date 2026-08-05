#!/usr/bin/env python3
"""W1 scorer: between-boot swing per (arm, tune) cell + tune contrast.

Reads data/w1/w1_llama_{arm}_{tune}_b{i}.json. For each cell:
  toks per boot (median-of-iters, e0 convention), between-boot swing
  (min/max - 1), within-boot spread (per-boot max iter / min iter - 1),
  accept per boot.

Verdicts against the pre-registered predictions (run_w1_boot.py header):
  P-W1a: notune between-boot swing < 5% in BOTH arms
  P-W1b: tune swing pattern -- spec-only vs engine-wide
  P-W1c: if notune swing >= 5%, autotune is not the cause
"""
import json
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "data" / "w1"

ARMS = ["off", "w2048"]
TUNES = ["tune", "notune"]


def cell(arm, tune):
    boots = []
    for b in range(3):
        p = D / f"w1_llama_{arm}_{tune}_b{b}.json"
        if p.exists():
            boots.append(json.loads(p.read_text()))
    return boots


def pct(x):
    return f"{100 * x:+.1f}%"


print(f"{'cell':16s} {'boot toks':>26s} {'between-boot':>12s} "
      f"{'within-boot':>22s} {'accept':>18s}")
swings = {}
for arm in ARMS:
    for tune in TUNES:
        boots = cell(arm, tune)
        if not boots:
            print(f"{arm}/{tune:6s}  (no data)")
            continue
        toks = [b["toks"] for b in boots]
        within = [max(b["all"]) / min(b["all"]) - 1 for b in boots]
        accepts = [b.get("accept") for b in boots]
        swing = max(toks) / min(toks) - 1 if len(toks) > 1 else 0.0
        swings[(arm, tune)] = swing
        print(f"{arm}/{tune:6s} n={len(boots)} "
              f"{str([round(t, 1) for t in toks]):>22s} "
              f"{pct(swing):>12s} "
              f"{str([pct(w) for w in within]):>22s} "
              f"{str(accepts):>18s}")

print()
for arm in ARMS:
    t, nt = swings.get((arm, "tune")), swings.get((arm, "notune"))
    if t is not None and nt is not None:
        print(f"[{arm:6s}] tune swing {pct(t)} -> notune swing {pct(nt)}")

nt_off = swings.get(("off", "notune"))
nt_spec = swings.get(("w2048", "notune"))
t_off = swings.get(("off", "tune"))
t_spec = swings.get(("w2048", "tune"))
if None not in (nt_off, nt_spec, t_off, t_spec):
    a = nt_off < 0.05 and nt_spec < 0.05
    print(f"\nP-W1a (notune swing <5% both arms): "
          f"{'CONFIRMED' if a else 'REFUTED'}")
    if t_spec >= 0.05 or t_off >= 0.05:
        loc = ("spec-only (draft-side kernels)"
               if t_spec >= 0.05 > t_off else
               "engine-wide" if t_off >= 0.05 else "AR-only (?)")
        print(f"P-W1b (localization under tune): {loc}")
    else:
        print("P-W1b: tune swing <5% in this matrix -- the I4 swing did "
              "not REPRODUCE; scoring cannot attribute it")
    if not a:
        print("P-W1c: autotune is NOT (the whole) cause -- check "
              "gpu_snapshot_* in the boot JSONs for contention/clocks")
