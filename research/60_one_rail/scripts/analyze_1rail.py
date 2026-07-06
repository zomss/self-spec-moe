"""Phase 60: build the 1-rail three-way table + implied-f estimates.

Reads the phase-60 1-rail JSONs and the phase-57 8-rail chat references
(same harness, same prompts, same batches) and prints:
  - implied f at 1 rail per batch: f ~= 1 - t_8rail/t_1rail (no-spec step time;
    same token count both sides, so t ratio == inverse tok/s ratio);
  - tok/s (accept) per {config x batch} with speedup vs 1-rail no-spec;
  - the b64 ranking.
"""

import json
import os

P60 = "/h/v-sukmincho/self-spec-moe/research/60_one_rail/data"
P57 = "/h/v-sukmincho/self-spec-moe/research/57_large_ep_spec_strategy/data"

FILES_1RAIL = {
    "no-spec": f"{P60}/w72n_q30b_1rail_nospec_nospec_cg_nospec.json",
    "EAGLE3 K=1": f"{P60}/w72n_q30b_1rail_eagle_spec_cg_K1.json",
    "EAGLE3 K=2": f"{P60}/w72n_q30b_1rail_eagle_spec_cg_K2.json",
    "WorldA K=2": f"{P60}/w72n_q30b_1rail_worldA_spec_cg_K2.json",
}
FILES_8RAIL = {
    "no-spec": f"{P57}/w72n_q30b_nospec_2n_chat_nospec_cg_nospec.json",
    "EAGLE3 K=1": f"{P57}/w72n_q30b_eagle_2n_chat_spec_cg_K1.json",
    "EAGLE3 K=2": f"{P57}/w72n_q30b_eagle_2n_chat_spec_cg_K2.json",
}
BATCHES = [8, 32, 64]


def load(path):
    if not os.path.exists(path):
        return {}
    rows = {}
    for r in json.load(open(path))["results"]:
        if "error" not in r:
            rows[r["batch"]] = r
    return rows


def main():
    r1 = {k: load(v) for k, v in FILES_1RAIL.items()}
    r8 = {k: load(v) for k, v in FILES_8RAIL.items()}

    print("== implied f at 1 rail (no-spec step-time ratio) ==")
    print(f"{'batch':>6} {'t8(s)':>8} {'t1(s)':>8} {'t1/t8':>6} {'f_1rail':>8}")
    for b in BATCHES:
        a = r8["no-spec"].get(b)
        c = r1["no-spec"].get(b)
        if not (a and c):
            print(f"{b:>6}  (missing)")
            continue
        t8, t1 = a["decode_s_mean"], c["decode_s_mean"]
        print(f"{b:>6} {t8:>8.3f} {t1:>8.3f} {t1 / t8:>6.2f} {1 - t8 / t1:>8.2f}")

    print("\n== 1-rail three-way table: tok/s (accept) [speedup vs no-spec] ==")
    hdr = f"{'config':<12}" + "".join(f"{'b' + str(b):>24}" for b in BATCHES)
    print(hdr)
    for cfg in FILES_1RAIL:
        cells = []
        for b in BATCHES:
            r = r1[cfg].get(b)
            base = r1["no-spec"].get(b)
            if not r:
                cells.append(f"{'--':>24}")
                continue
            al = r.get("accept_len")
            al_s = f" ({al:.2f})" if al else ""
            sp = f" [{r['tok_s_mean'] / base['tok_s_mean']:.2f}x]" if base else ""
            cells.append(f"{r['tok_s_mean']:>10.1f}{al_s}{sp:>12}")
        print(f"{cfg:<12}" + "".join(cells))

    print("\n== 8-rail chat reference (Phase 57) ==")
    for cfg in FILES_8RAIL:
        cells = []
        for b in BATCHES:
            r = r8[cfg].get(b)
            cells.append(f"{r['tok_s_mean']:>10.1f}" if r else f"{'--':>10}")
        print(f"{cfg:<12}" + "".join(cells))

    print("\n== b64 ranking (1 rail) ==")
    ranked = sorted(
        ((cfg, r1[cfg][64]["tok_s_mean"]) for cfg in r1 if 64 in r1[cfg]),
        key=lambda t: -t[1],
    )
    for i, (cfg, v) in enumerate(ranked, 1):
        print(f"{i}. {cfg:<12} {v:.1f} tok/s")


if __name__ == "__main__":
    main()
