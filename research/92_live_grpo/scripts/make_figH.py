#!/usr/bin/env python3
"""Fig H: real long-horizon GRPO — stale-drafter accept vs training step,
with the policy's entropy/KL trajectory underneath. Two panels, one story:
the drafter frozen at step 0 does not go stale; accept RISES as the
policy sharpens (entropy collapse dominates weight drift)."""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASE = Path(__file__).resolve().parents[1]
CURVE = PHASE / "data" / "e1_curve.jsonl"
LOG = PHASE / "logs" / "stage1_no-sd.log"
OUT = Path("/data/smcho/self-spec-moe/paper/figures/figH_live_staleness.png")

# ---- accept curve (long-run rows + step0 anchor), mean over seeds ----
acc = {}   # step -> list of accepts (stale drafter@0)
fresh = {}
for line in CURVE.read_text().splitlines():
    r = json.loads(line)
    tgt = str(r.get("target"))
    tag = r["tag"]
    if tag == "e2-step0":
        acc.setdefault(0, []).append(r["accept"])
    elif "92_grpo_long" in tgt and tag.endswith("-stale"):
        s = int(re.search(r"e2-step(\d+)-", tag).group(1))
        acc.setdefault(s, []).append(r["accept"])
    elif "92_grpo_long" in tgt and tag.endswith("-fresh"):
        s = int(re.search(r"e2-step(\d+)-", tag).group(1))
        fresh.setdefault(s, []).append(r["accept"])

steps = sorted(acc)
mean = [sum(acc[s]) / len(acc[s]) for s in steps]
lo = [min(acc[s]) for s in steps]
hi = [max(acc[s]) for s in steps]

# ---- entropy / KL from the training log ----
ent, kl = {}, {}
pat = re.compile(r"step:(\d+) - actor/entropy:([\d.]+).*?actor/kl_loss:np\.float64\(([\d.e-]+)\)")
for m in pat.finditer(LOG.read_text()):
    s = int(m.group(1))
    ent[s] = float(m.group(2))
    kl[s] = float(m.group(3))
tsteps = sorted(ent)

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(7.2, 5.4), sharex=True,
    gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12})

ax1.fill_between(steps, lo, hi, alpha=0.25, color="#2b6cb0", lw=0)
ax1.plot(steps, mean, "o-", color="#2b6cb0", lw=2, ms=5,
         label="stale drafter@0 (never refreshed)")
if fresh:
    fs = sorted(fresh)
    ax1.plot(fs, [sum(fresh[s]) / len(fresh[s]) for s in fs], "s",
             color="#c05621", ms=7, label="fresh drafter@step (requantized)")
ax1.axhline(3.6, color="#b91c1c", ls="--", lw=1.2)
ax1.text(2, 3.615, "refresh gate (S<1 boundary)", color="#b91c1c",
         fontsize=8, va="bottom")
ax1.set_ylabel("accept length @ K=4")
ax1.set_ylim(3.4, 4.5)
ax1.legend(loc="lower right", fontsize=8, frameon=False)
ax1.set_title("Real GRPO (their recipe, run live): the frozen drafter never "
              "goes stale —\naccept RISES as the policy sharpens",
              fontsize=10)

ax2.plot(tsteps, [ent[s] for s in tsteps], "o-", color="#4a5568", lw=1.5,
         ms=3, label="policy entropy")
ax2.set_ylabel("entropy", color="#4a5568")
ax2.set_xlabel("GRPO training step (batch 32, lr 5e-7, MATH lv.3-5)")
ax2b = ax2.twinx()
ax2b.plot(tsteps, [kl[s] for s in tsteps], "s-", color="#805ad5", lw=1.5,
          ms=3, label="KL(policy || base)")
ax2b.set_ylabel("KL loss", color="#805ad5")
ax2.legend(loc="center right", fontsize=8, frameon=False)
ax2b.legend(loc="upper center", fontsize=8, frameon=False)

for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(True)
fig.savefig(OUT, dpi=180, bbox_inches="tight")
print("wrote", OUT)
print("stale:", {s: round(sum(a)/len(a), 3) for s, a in sorted(acc.items())})
print("fresh:", {s: round(sum(a)/len(a), 3) for s, a in sorted(fresh.items())})
