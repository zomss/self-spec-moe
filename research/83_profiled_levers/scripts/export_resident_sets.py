#!/usr/bin/env python3
"""Export expert_usage.pt (per-layer counts) -> resident-set files for the
VLLM_SELF_SPEC_DRAFT_RESIDENT_SETS env ({layer_idx: LongTensor top-frac ids})."""

import sys
from pathlib import Path

import torch

PHASE = Path(__file__).resolve().parents[1]
counts = torch.load(PHASE / "data/expert_usage.pt", weights_only=False)
for frac, name in ((0.5, "flr50"), (0.25, "flr25")):
    sets = {li: c.argsort(descending=True)[: max(1, int(len(c) * frac))].clone()
            for li, c in enumerate(counts)}
    out = PHASE / f"data/resident_sets_{name}.pt"
    torch.save(sets, out)
    print(f"{out}: {len(sets)} layers x {len(sets[0])} experts")
