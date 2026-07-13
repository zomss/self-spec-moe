"""Second-distribution beta sweep (math reasoning) for the paper's robustness
check: same harness as 77/score_accept.py, prompt bank swapped to AIME-derived
math prompts, refs/output redirected into this phase (data/refs/, data/beta.csv).

Usage: identical CLI to score_accept.py, e.g.
  .venv/bin/python beta_dist2.py --model dense --ctx 16384 --arms win512,q_fp8
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
P77 = REPO / "research/77_acceptance_map/scripts"
sys.path.insert(0, str(P77))

import anchor_gate  # noqa: E402
import score_accept  # noqa: E402

anchor_gate.PROMPTS = REPO / "research/79_paper/data/prompts_math.txt"
score_accept.PHASE = REPO / "research/79_paper"

if __name__ == "__main__":
    raise SystemExit(score_accept.main())
