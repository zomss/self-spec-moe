#!/usr/bin/env python3
"""C7/R3: retrieval (needle) bank -- the window-stress workload axis.

Each prompt: W7-style filler with N planted facts at controlled depths
(all >2k tokens from the END, i.e. outside any 512-window), then a
question requiring one planted fact. The teacher-forced continuation
(the answer) needs long-range attention -> window beta should collapse,
kvq (full-context, half-byte) should hold. 16 prompts, one line each.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "77_acceptance_map/scripts"))
from anchor_gate import FILLER  # noqa: E402

random.seed(86)
FACTS = [
    ("the harbor master's ledger", "the ship 'Meridian Star' carried {} crates of saffron"),
    ("the observatory's log", "the comet was recorded at {} degrees above the horizon"),
    ("the merchant's contract", "the agreed price was {} silver coins per bale"),
    ("the census scroll", "the village of Thornfield counted {} households"),
]
OUT = Path(__file__).resolve().parents[1] / "data/prompts_retrieval.txt"
lines = []
for i in range(16):
    key, tmpl = FACTS[i % len(FACTS)]
    val = random.randint(137, 977)
    fact = f"According to {key}, {tmpl.format(val)}."
    q = (f"Based on the document above, state exactly what {key} recorded, "
         f"including the precise number, and explain where it appeared.")
    # the harness/refs builder appends filler AFTER this line up to ctx;
    # embed the fact INSIDE the prompt line followed by a long spacer cue.
    lines.append(f"{fact} [This fact appears early in a long document.] {q}")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines) + "\n")
print("wrote", OUT, len(lines), "prompts")
