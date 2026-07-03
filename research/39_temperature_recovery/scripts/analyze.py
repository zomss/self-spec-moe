"""Phase 39: build the temperature-sweep table from data/temp_*.json and the
losslessness comparison from data/lossless_*.json.

Usage:
  python analyze.py sweep   # proxy temperature sweep table
  python analyze.py lossless # proxy losslessness compare (nospec vs spec @ temp)
  python analyze.py q30      # qwen3-30b accept + speedup table
"""
import glob
import json
import os
import sys

DATA = "/data/smcho/self-spec-moe/research/39_temperature_recovery/data"


def _load(path):
    with open(path) as f:
        return json.load(f)


def sweep():
    rows = []
    for p in sorted(glob.glob(os.path.join(DATA, "temp_sweep_*_dp8_K4.json"))):
        d = _load(p)
        cfg = d.get("cfg", "A")
        if "error" in d:
            rows.append((cfg, d.get("temp"), "ERR", d["error"]))
            continue
        rows.append((cfg, d.get("temp"), d.get("accept_len"),
                     d.get("per_tok"), d.get("per_pos_rate"), d.get("probabilistic")))
    print("=== PROXY TEMPERATURE SWEEP (DP=8) ===")
    print("cfg A = full-replica comm-free draft; cfg C = EP-full draft (control)")
    print(f"{'cfg':>3} {'temp':>5} {'prob':>5} {'accept_len':>11} {'per_tok':>8}"
          "  per_pos_rate")
    for r in sorted(rows, key=lambda x: (x[0], x[1] if x[1] is not None else 0)):
        if len(r) == 4:
            print(f"{r[0]:>3} {r[1]:>5}  ERROR: {r[3]}")
            continue
        cfg, temp, al, pt, pp, prob = r
        pps = "[" + ", ".join(f"{x:.3f}" if x is not None else "?"
                              for x in (pp or [])) + "]"
        al_s = f"{al:.3f}" if al else "None"
        pt_s = f"{pt:.3f}" if pt else "None"
        print(f"{cfg:>3} {temp:>5} {str(prob):>5} {al_s:>11} {pt_s:>8}  {pps}")


def lossless():
    files = {}
    for p in glob.glob(os.path.join(DATA, "lossless_*.json")):
        d = _load(p)
        s = d["summary"]
        files[(s["mode"], s.get("temp"))] = (d, p)
    temps = sorted({k[1] for k in files})
    print("=== PROXY LOSSLESSNESS UNDER TEMPERATURE (fixed seed) ===")
    for t in temps:
        ns = files.get(("nospec", t))
        sp = files.get(("spec", t))
        if not (ns and sp):
            print(f"temp={t}: missing nospec or spec run")
            continue
        ns_res = {r["prompt"]: r["token_ids"] for r in ns[0]["results"]}
        sp_res = {r["prompt"]: r["token_ids"] for r in sp[0]["results"]}
        exact = 0
        tot_tok = 0
        agree_tok = 0
        n = 0
        first_div = []
        for prompt, ns_ids in ns_res.items():
            sp_ids = sp_res.get(prompt)
            if sp_ids is None:
                continue
            n += 1
            if ns_ids == sp_ids:
                exact += 1
            L = min(len(ns_ids), len(sp_ids))
            # first divergence index
            fd = next((i for i in range(L) if ns_ids[i] != sp_ids[i]), L)
            first_div.append(fd if fd < L else None)
            for i in range(L):
                tot_tok += 1
                if ns_ids[i] == sp_ids[i]:
                    agree_tok += 1
        sp_summary = sp[0]["summary"]
        fd_clean = [x for x in first_div if x is not None]
        print(f"temp={t}: exact_seq={exact}/{n}  per_tok_agree="
              f"{agree_tok}/{tot_tok}={agree_tok / tot_tok:.3f}  "
              f"spec_accept_len={sp_summary.get('mean_accept_length')}  "
              f"spec_accept_rate={sp_summary.get('acceptance_rate')}")
        if fd_clean:
            print(f"          first-divergence indices (min/median/max): "
                  f"{min(fd_clean)}/"
                  f"{sorted(fd_clean)[len(fd_clean) // 2]}/{max(fd_clean)} "
                  f"(of {len(fd_clean)} diverging prompts)")


def q30():
    print("=== QWEN3-30B TEMPERATURE + SPEEDUP (DP=8 EP=8) ===")
    spec = {}
    nospec = {}
    for p in sorted(glob.glob(os.path.join(DATA, "q30t_*.json"))):
        d = _load(p)
        res = d.get("results") or []
        for row in res:
            if "error" in row:
                print(f"  {os.path.basename(p)} batch={row.get('batch')} "
                      f"ERROR {row['error']}")
                continue
            key = (d["temp"], d.get("full_cg"), row["batch"], d.get("this_k"))
            if d["mode"] == "spec":
                spec[key] = (d, row)
            else:
                nospec[(d["temp"], row["batch"])] = (d, row)
    print(f"{'temp':>5} {'fcg':>5} {'batch':>5} {'K':>2} {'accept_len':>11} "
          f"{'spec_tps':>9} {'nospec_tps':>10} {'speedup':>8}")
    for key in sorted(spec):
        temp, fcg, batch, k = key
        d, row = spec[key]
        ns = nospec.get((temp, batch))
        ns_tps = ns[1]["tok_s_mean"] if ns else None
        sp_tps = row["tok_s_mean"]
        spd = (sp_tps / ns_tps) if (ns_tps and sp_tps) else None
        al = row.get("accept_len")
        print(f"{temp:>5} {str(fcg):>5} {batch:>5} {k:>2} "
              f"{(f'{al:.3f}' if al else 'None'):>11} {sp_tps:>9.1f} "
              f"{(f'{ns_tps:.1f}' if ns_tps else 'NA'):>10} "
              f"{(f'{spd:.3f}' if spd else 'NA'):>8}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    {"sweep": sweep, "lossless": lossless, "q30": q30}[which]()
