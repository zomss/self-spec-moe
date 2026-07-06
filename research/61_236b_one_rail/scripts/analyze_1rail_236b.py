"""Phase 61 analysis: 236B 1-rail two-way table + implied f.

Reads the Phase 61 data JSONs (1-rail nospec + World A K=1) and compares
against the fixed 8-rail references (Phase 53 nospec, Phase 55/54
run_step3_095 World A node-local K=1).

Implied comm fraction at 1 rail, from the no-spec step-time ratio
(comm grew ~8x, compute did not):  f_1rail ~= 1 - t_8rail/t_1rail.
"""

import glob
import json
import os

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data")

# 8-rail references at b8/32/64 (tok/s over 128 decode tokens/req/rank).
REF8_NOSPEC = {8: 264.0, 32: 756.0, 64: 1130.0}          # Phase 53, 0.90
REF8_WORLDA = {8: 81.0, 32: 273.4, 64: 378.6}            # Phase 55 paired K=1
REF8_ACCEPT = {8: 1.916, 32: 1.870, 64: 1.881}


def load(tag_glob):
    paths = sorted(glob.glob(os.path.join(DATA, tag_glob)))
    if not paths:
        return None
    with open(paths[-1]) as f:
        d = json.load(f)
    rows = {}
    for r in d.get("results", []):
        if "batch" in r and "tok_s_mean" in r:
            rows[r["batch"]] = r
    return rows


def step_ms(row):
    # decode_s_mean covers OUTLEN-SHORTLEN = 128 decode steps.
    n_steps = row["out_len"] - row["short_len"]
    return 1000.0 * row["decode_s_mean"] / n_steps


def main():
    nospec = load("w72n_dsv2_1rail_nospec_nospec_cg_nospec.json")
    worlda = load("w72n_dsv2_1rail_worldA_spec_cg_K1.json")
    if not nospec:
        print("no nospec data yet")
        return

    print("== implied f at 1 rail (no-spec step ratio vs 8-rail refs) ==")
    print(f"{'batch':>6} {'t8 ms':>8} {'t1 ms':>8} {'ratio':>7} {'f_1rail':>8}")
    for b in sorted(nospec):
        t1 = step_ms(nospec[b])
        # tok/s = b*128/decode_s -> t_step_ms = decode_s*1000/128 = b*1000/tok_s.
        t8 = b * 1000.0 / REF8_NOSPEC[b]
        f = 1.0 - t8 / t1
        print(f"{b:>6} {t8:>8.1f} {t1:>8.1f} {t1 / t8:>7.2f} {f:>8.3f}")

    print()
    print("== two-way table (1 rail): tok/s (accept), speedup vs 1-rail nospec ==")
    hdr = f"{'batch':>6} {'nospec':>12} {'WorldA-K1':>22} {'speedup':>9}"
    print(hdr)
    for b in sorted(nospec):
        ns = nospec[b]["tok_s_mean"]
        cell = "-"
        spd = "-"
        if worlda and b in worlda:
            w = worlda[b]
            al = w.get("accept_len")
            cell = f"{w['tok_s_mean']:.1f}+-{w['tok_s_std']:.1f} ({al:.3f})"
            spd = f"{w['tok_s_mean'] / ns:.3f}x"
        print(f"{b:>6} {ns:>12.1f} {cell:>22} {spd:>9}")

    print()
    print("== 8-rail references (fixed) ==")
    for b in (8, 32, 64):
        print(f"  b{b}: nospec {REF8_NOSPEC[b]:.0f}, WorldA-K1 {REF8_WORLDA[b]:.1f} "
              f"(accept {REF8_ACCEPT[b]:.3f}) -> {REF8_WORLDA[b] / REF8_NOSPEC[b]:.3f}x")


if __name__ == "__main__":
    main()
