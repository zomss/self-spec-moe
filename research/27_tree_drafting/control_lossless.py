#!/usr/bin/env python3
"""Control: is the tree's low greedy_match a verify bug or batched-numerics+cascade?

greedy_match vs the GLOBAL reference cascades: one numerics flip (B1) makes the whole
suffix 'mismatch'. This isolates PER-CYCLE correctness: each cycle, compare the tree
verify's accepted tokens to a FRESH sequential bf16 greedy from THAT cycle's committed
prefix. If per-cycle match is ~1.0 (modulo the B1 ~1.5% rate), the tree verify is
correct and the low aggregate was cascade; if it's low, the tree verify is wrong.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import tree_verify as tv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-tokens", type=int, default=32)
    ap.add_argument("--cache-frac", type=float, default=0.5)
    ap.add_argument("--schedule", default="2,2")
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--local-files-only", action="store_true")
    a = ap.parse_args()
    dev = a.device
    sched = [int(x) for x in a.schedule.split(",")]

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained(a.model, local_files_only=a.local_files_only)
    E = cfg.num_experts; C = int(round(a.cache_frac * E))
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=a.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="eager", local_files_only=a.local_files_only).to(dev).eval()
    gates = [m for m in model.modules() if type(m).__name__ == "Qwen3MoeTopKRouter"]
    print(f"[cfg] schedule={sched} C={C}(0.5E)")

    per_cycle_match, per_cycle_tot = 0, 0
    for p in tv.PROMPTS[:3]:
        ids = tok(p, return_tensors="pt").input_ids.to(dev)
        masks = tv.build_masks(model, ids, E, C, dev)
        cur, produced = ids, 0
        while produced < a.n_tokens:
            nodes = tv.build_tree(model, gates, masks, cur, sched)
            acc_seq, _ = tv.verify_tree(model, cur, nodes)
            # FRESH sequential greedy from THIS cycle's prefix (no cascade)
            ref = tv.reference_greedy(model, cur, len(acc_seq))
            for t, r in zip(acc_seq, ref):
                per_cycle_match += int(t == r); per_cycle_tot += 1
            cur = torch.cat([cur, torch.tensor([acc_seq], device=dev)], dim=1)
            produced += len(acc_seq)

    rate = per_cycle_match / max(per_cycle_tot, 1)
    print(f"\nper-cycle (fresh-reference, NO cascade) match: {rate:.3f} "
          f"({per_cycle_match}/{per_cycle_tot})")
    print("interpretation: ~0.95+ => tree verify correct, low aggregate was cascade; "
          "low => tree-verify bug")
    a.output_json.write_text(json.dumps(
        {"schedule": sched, "per_cycle_match": round(rate, 4),
         "n": per_cycle_tot}, indent=2))
    print(f"[saved] {a.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
