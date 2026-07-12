#!/usr/bin/env python3
"""V3: predict the ENTIRE deferred MLA cost tier from config constants.

REGISTERED BEFORE `e1_sweep.sh mla` RUNS. Transfer rule (documented, fixed):
  - MLA constants from DeepSeek-V2-Lite's config alone: W_dense 2.6 GB
    (attn/MLA proj + embeddings + first dense layer + 2 always-on shared
    experts), routed experts 64/layer x 26 MoE layers x 17.3 MB -> exp_bytes
    450 MB rolled across layers, residents/rank 16, top-6; KV 576 latent
    x 2 B x 27 layers = 31104 B/token (FLASH_ATTN_MLA reads the latent);
    P_active 2.4e9.
  - From the moe fit (corrected units): BW_eff, h (KV-management), comm c0/c1,
    quant kappas transfer AS-IS; the launch floor f0, f1 scales by the layer
    ratio 27/48 (floor ~ kernel count ~ layers).
Gate (README): median |err| <= 15% AND crossovers placed correctly.
"""

import json
import sys
from pathlib import Path

PHASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PHASE / "scripts"))
import fit_cost_model as M  # noqa: E402

M.CONST["mla"] = dict(
    W_dense=2.6e9, n_res=16, E=64, topk=6, exp_bytes=450.0e6,
    kv_tok=31104.0, P_active=2.4e9, dp=4,
    ctx_of={2: 2048, 16: 16384, 32: 32768})
M.KAPPAS["mla"] = ["fp8m", "fib", "a2atile"]
ARMS_MLA = {
    "ds_bf16": dict(comm=True),
    "ds_fp8marlin": dict(w=0.5, kappa="fp8m", comm=True),
    "ds_fp8block": dict(w=0.5, kappa="fib", comm=True),
    "ds_win": dict(win=512 + 16, comm=True),
    "ds_skip50": dict(ls=0.5, comm=True),
    "ds_kvq": dict(kvs=0.5, comm=True),   # vs the FLASHMLA-pinned bf16 pair
}
for a, spec in ARMS_MLA.items():
    M.ARMS[("mla", a)] = spec
LAYER_RATIO = 27 / 48


def main() -> int:
    blob = json.loads((PHASE / "data/fit.json").read_text())
    p = blob["moe/additive"]["params"]
    pv = [p["f0"] * LAYER_RATIO, p["f1"] * LAYER_RATIO, p["BW_GBs"],
          p["h_ms_per_Mtok"], p["c0"], p["c1"],
          p["kappa_fp8m"], p["kappa_fib"], p["kappa_a2atile"]]
    out = {"transfer": dict(layer_ratio=round(LAYER_RATIO, 3), moe_params=p),
           "cells": {}}
    for arm in ARMS_MLA:
        for b in (4, 8, 32):
            for c in (2, 16, 32):
                t = M.predict("mla", pv, arm, b, c, "additive")
                t_bf = M.predict("mla", pv, "ds_bf16", b, c, "additive")
                out["cells"][f"{arm}_b{b}_c{c}k"] = dict(
                    tpot=round(t, 2), R=round(t / t_bf, 3))
    # headline registered claims
    win = {k: v["R"] for k, v in out["cells"].items() if k.startswith("ds_win")}
    out["claims"] = [
        f"window is a WEAK lever on MLA (KV latent tiny): R stays "
        f"{min(win.values()):.2f}-{max(win.values()):.2f} across the grid "
        "(vs 0.35-0.65 on GQA-MoE at long ctx)",
        "quant arms ~parity plus kernel tax (as on GQA-MoE)",
        "=> combined with 77's beta, NVLink MLA has NO strong self-spec lever; "
        "its levers await the comm-bound fabric (shared-expert local-route).",
    ]
    f = PHASE / "data/v3_predictions_mla.json"
    f.write_text(json.dumps(out, indent=1))
    for k in ("ds_bf16", "ds_win", "ds_fp8marlin", "ds_skip50", "ds_kvq"):
        row = {c: out["cells"][f"{k}_b{b}_c{c}k"]["R"]
               for b in (8,) for c in (2, 16, 32)}
        print(f"{k:14s} R@b8: " + "  ".join(f"c{c}k={v:.3f}" for c, v in row.items()))
    print("\n".join(out["claims"]))
    print(f"REGISTERED -> {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
