#!/usr/bin/env python3
"""Phase 80 E0: measured beta for COMBINED levers (the composition law).

Stacks 77's arm mechanisms in ONE draft forward against the same stage-A
refs (paired with the singles): window slice x fake-quant weights x layer
skip x kv-quant x router mask. Writes data/beta_combo.csv; analyze_combo.py
tests beta_combo against product(beta_i) and min(beta_i).

Combo grammar: "win512+q_fp8+kvq_fp8" etc. At most one q_* per combo
(weight mutation is destructive -> combos grouped by quant kind, model
reloaded between kinds, quant groups run in sequence).

Cells: law pairs at ctx {2048, 32768}; the three E3-candidate combos at all
three ctx. 12 prompts x 96 positions per cell (77's stability requirement).
"""

import argparse
import contextlib
import csv
import gc
import json
import sys
from pathlib import Path

import torch

PHASE = Path(__file__).resolve().parent.parent
P77 = PHASE.parent / "77_acceptance_map"
sys.path.insert(0, str(P77 / "scripts"))
from score_accept import (  # noqa: E402
    MODELS, LayerSkip, RouterMask, fake_quant_, kv_quant, load_model, view_cache)

LAW_CTX = [2048, 32768]
ALL_CTX = [2048, 16384, 32768]
# combo -> ctx list; E3-candidates get ALL_CTX
COMBOS = {
    "dense": {
        "win512+q_fp8": LAW_CTX, "win512+q_int4": ALL_CTX,
        "win512+skip125": LAW_CTX, "win512+skip25": LAW_CTX,
        "q_fp8+skip125": LAW_CTX, "win512+kvq_fp8": LAW_CTX,
        "q_fp8+kvq_fp8": LAW_CTX, "win512+q_int4+kvq_fp8": LAW_CTX,
    },
    "moe": {
        "win512+q_fp8": LAW_CTX, "win512+skip125": LAW_CTX,
        "q_fp8+skip125": LAW_CTX, "win512+kvq_fp8": LAW_CTX,
        "win512+lr50": ALL_CTX, "q_fp8+lr50": LAW_CTX,
        "skip125+lr50": LAW_CTX, "win512+lr50+q_fp8": ALL_CTX,
    },
    "mla": {
        "skip125+q_fp8": ALL_CTX, "win512+q_fp8": LAW_CTX,
        "skip125+lr50": LAW_CTX, "q_fp8+lr50": LAW_CTX,
        "win512+skip125": LAW_CTX, "skip125+q_fp8+lr50": LAW_CTX,
    },
}


def parse(combo: str):
    spec = dict(window=None, skip=None, lr=None, kvq=False, quant=None)
    for part in combo.split("+"):
        if part.startswith("win"):
            spec["window"] = int(part[3:]) + 16          # + sinks
        elif part.startswith("skip"):
            spec["skip"] = int(part[4:]) / (1000 if len(part) > 6 else 100)
        elif part.startswith("lr"):
            spec["lr"] = int(part[2:]) / 100
        elif part == "kvq_fp8":
            spec["kvq"] = True
        elif part.startswith("q_"):
            spec["quant"] = part[2:]
    return spec


@torch.no_grad()
def score_combo(model, cfg, combo: str, ctx: int, refdir: Path, prompts: int):
    spec = parse(combo)
    n_layers = model.config.num_hidden_layers
    dev = "cuda"
    mans = []
    if spec["skip"]:
        mans.append(LayerSkip(model, cfg["layers"], spec["skip"]))
    if spec["lr"]:
        mans.append(RouterMask(model, spec["lr"]))
    matches, overlaps = [], []
    for pi in range(prompts):
        meta = torch.load(refdir / f"p{pi}_meta.pt", weights_only=False)
        kv_cpu = torch.load(refdir / f"p{pi}_cache.pt", weights_only=False)
        kv = {li: tuple(t.to(dev) for t in kv_cpu[li]) for li in kv_cpu}
        if spec["kvq"]:
            kv = kv_quant(kv)
        T = meta["prompt_ids"].shape[1]
        with contextlib.ExitStack() as st:
            for m in mans:
                st.enter_context(m)
            for step, cur in enumerate(meta["gen_tokens"]):
                seqlen = T + step
                dcache = view_cache(kv, n_layers, seqlen, window=spec["window"])
                out = model(input_ids=torch.tensor([[cur]], device=dev),
                            past_key_values=dcache, use_cache=True,
                            position_ids=torch.tensor([[seqlen]], device=dev),
                            cache_position=torch.tensor(
                                [dcache.get_seq_length()], device=dev))
                dp = torch.softmax(out.logits[:, -1].float(), -1)
                tp = meta["probs"][step].to(dev).float()
                matches.append(float(int(dp.argmax()) == int(tp.argmax())))
                overlaps.append(float(torch.minimum(dp[0], tp).sum()))
                del dcache, out
        del kv, kv_cpu
        gc.collect()
        torch.cuda.empty_cache()
    return dict(model=None, combo=combo, ctx=ctx,
                beta_greedy=round(sum(matches) / len(matches), 4),
                overlap_t1=round(sum(overlaps) / len(overlaps), 4),
                positions=len(matches), prompts=prompts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--prompts", type=int, default=12)
    a = ap.parse_args()
    cfg = MODELS[a.model]
    out = PHASE / "data/beta_combo.csv"
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["model"], r["combo"], r["ctx"])
                for r in csv.DictReader(out.open())}

    # group by quant kind: none first, then each kind with a fresh model
    kinds = sorted({parse(c)["quant"] for c in COMBOS[a.model]},
                   key=lambda k: (k is not None, k or ""))
    model = load_model(cfg)
    mutated = False
    for kind in kinds:
        todo = [(c, ctx) for c, ctxs in COMBOS[a.model].items()
                if parse(c)["quant"] == kind for ctx in ctxs
                if (a.model, c, str(ctx)) not in done]
        if not todo:
            continue
        if kind is not None:
            if mutated:
                del model
                gc.collect(); torch.cuda.empty_cache()
                model = load_model(cfg)
            fake_quant_(model, kind)
            mutated = True
        for combo, ctx in todo:
            refdir = P77 / f"data/refs/{a.model}_c{ctx}"
            r = score_combo(model, cfg, combo, ctx, refdir, a.prompts)
            r["model"] = a.model
            print(f"[E0] RESULT {json.dumps(r)}", flush=True)
            exists = out.exists()
            with out.open("a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(r.keys()))
                if not exists:
                    w.writeheader()
                w.writerow(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
