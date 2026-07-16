#!/usr/bin/env python3
"""E1-ngram: prompt-lookup drafting gate -- CPU-only, on 77's cached refs.

Prompt-lookup (PLD/vLLM ngram): match the last n generated tokens against
the sequence so far; if found, propose the tokens that followed the most
recent earlier occurrence. No draft model: R = 0 and the draft is
FLOOR-FREE by construction (speedup = tau alone).

Metrics per (model, ctx): coverage (a proposal exists), accuracy
(proposal == target token | proposed), and beta_eff (abstain counted as
miss -- the map-compatible scalar); plus the multi-token accepted-run
length when proposing k=4 (PLD proposes a block).
"""

import csv
import sys
from pathlib import Path

import torch

PHASE = Path(__file__).resolve().parents[1]
P77 = PHASE.parent / "77_acceptance_map"
OUT = PHASE / "data/ngram_gate.csv"
(PHASE / "data").mkdir(parents=True, exist_ok=True)

NGRAM = 2          # vLLM default prompt_lookup window
BLOCK = 4          # proposed block length for the run-length metric


def propose(seq: list, n: int, k: int):
    """Most-recent n-gram match; return the k tokens that followed, or None."""
    if len(seq) < n + 1:
        return None
    tail = seq[-n:]
    # scan backwards, excluding the tail itself
    for start in range(len(seq) - n - 1, -1, -1):
        if seq[start:start + n] == tail:
            nxt = seq[start + n:start + n + k]
            return nxt if nxt else None
    return None


def main() -> int:
    rows = []
    for model, ctx in (("dense", 16384), ("moe", 16384), ("mla", 16384)):
        refdir = P77 / f"data/refs/{model}_c{ctx}"
        if not (refdir / "p0_meta.pt").exists():
            continue
        cov = acc = eff = n_pos = 0
        run_lens = []
        for pi in range(12):
            meta = torch.load(refdir / f"p{pi}_meta.pt", weights_only=False)
            prompt = meta["prompt_ids"][0].tolist()
            gen = [int(t) for t in meta["gen_tokens"]]
            for step in range(len(gen)):
                seq = prompt + gen[:step]
                prop = propose(seq, NGRAM, BLOCK)
                n_pos += 1
                if prop is None:
                    continue
                cov += 1
                ok = prop[0] == gen[step]
                acc += ok
                eff += ok
                # accepted run length within the proposed block
                r = 0
                for j, t in enumerate(prop):
                    if step + j < len(gen) and t == gen[step + j]:
                        r += 1
                    else:
                        break
                run_lens.append(r)
        row = dict(model=model, ctx=ctx, n_pos=n_pos,
                   coverage=round(cov / n_pos, 4),
                   acc_given_proposed=round(acc / max(cov, 1), 4),
                   beta_eff=round(eff / n_pos, 4),
                   mean_run_at_k4=round(sum(run_lens) / max(len(run_lens), 1), 3))
        rows.append(row)
        print(row, flush=True)
    with OUT.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
