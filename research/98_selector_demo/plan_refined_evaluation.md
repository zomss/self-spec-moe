# Refined plan: baselines, the LI/LO/LIO/SS grid, and what we measure next

Written 2026-08-18 on the VM branch, after porting Phase 100 from
`research/self-spec-moe-baremetal`. Supersedes the "candidate next steps"
table in `status_vm_branch.md` items B and C; item A (MoE) is deferred to
after this arc, extended to TP=2.

## 1. What was ported, and why it changes the plan

`research/100_baselines/` now exists here in full:

* `baseline_survey.md` — the published field grouped by lever. Main set is
  families **A (layer-skip), B (KV-sparsity draft), C (quantized draft)**;
  model-free drafters (PLD/ngram) deferred by decision as a drafter
  *replacement* rather than a lever composition.
* `dataset_map.md`, `final_eval_design.md` — the scoring protocol.
* `results_campaign1_interim.md` + `data/campaign1/` — 31 of 35 boots
  already measured on h103/h104.

Two things in there change what we should do next.

**The evaluation grid is length-quadrant, not regime-labelled**, and it
retires `ignore_eos` from scored runs:

| cell | dataset | input | output (natural) | batch sweep |
| --- | --- | --- | --- | --- |
| **LI** | GovReport | ~16K | ~0.3–1K | 1 / 8 / 16 |
| **LO** | AIME 2024+2025 | ~0.1–0.5K | ~5–25K | 1 / 8 / 16 / 32 |
| **LIO** | BookSum (≥8K chapters) | ~8–16K | ~1–2K | 1 / 8 / 16 |
| **SS** | GSM8K concise | ~0.3K | ~0.1–0.3K | 1 / 8 / 32 / 64 |

Retiring `ignore_eos` is nearly free because every arm is
distribution-preserving: at T=0 all arms emit identical tokens and stop at
the same EOS, so equal work is preserved by losslessness rather than
enforced. It also buys a **free correctness gate** — any cross-arm output
divergence at T=0 fails the run.

**Campaign 1 already shows the switching case.** Scored against stock:

| arm | b8 LO | b16 LO | b32 LO |
| --- | --- | --- | --- |
| w4a16 | **1.099** | 1.014 | 1.090 |
| w1024 | 0.911 | **1.271** | 1.484 |
| magicdec512 | 0.843 | 1.190 | **1.504** |

The winner moves with batch inside one cell, and at LI/LIO every armed arm
loses to stock (0.61–0.94). That is exactly the shape a selector should
profit from: a different lever per cell, and whole cells where the right
answer is not to speculate. It is a far better test bed than the six R-
regimes, where per-regime switching was worth only +1.4%.

## 2. Consequence for item B, which needs restating

`status_vm_branch.md` framed B as "measure acceptance on this box instead of
transferring it". **That framing is wrong and this plan corrects it.**
Acceptance is deterministic given checkpoints and prompts — G98-F measured
bit-identical accept patterns across boxes — so re-measuring the same
content here reproduces h104's numbers exactly and changes nothing. The R6
transfer error is **content generalisation** (calibrated on seeds 4–5,
applied to seeds 2–3), not a box effect.

What "Round 2 proper" must therefore mean is: **acceptance measured on the
distribution the evaluation actually runs.** Under the refined grid that is
not a refinement but a prerequisite, because of the u-axis:

* G98-D binned acceptance by generated-suffix length with edges
  `[256, 1024, 3072]` and, generating 640 tokens, populated **only buckets
  0 and 1**.
* The LO cell generates **5–25K tokens**. Buckets 2 and 3 — every step past
  1K of generation, which is where most LO tokens live — have **never been
  measured**.

So the acceptance input to the selector is currently undefined over the
regime the refined grid is built to stress. B is the work of defining it.

## 3. Item C, likewise sharpened

C was "spend the burn-in budget where the time is". Under the R-regimes that
meant R1 (59% of time). Under the refined grid the time shares are dominated
by **LO**, which is also the cell with the longest generations and therefore
the strongest u-dependence. C and B collapse into one question — *what is
acceptance as a function of how far into a generation you are, and how much
of it must be measured online* — which the refined grid can answer directly
rather than by proxy.

## 4. Sequence

1. **B (now).** Measure acceptance as a function of u out to the lengths the
   LO cell reaches, for the candidate set, on this box. Deliverable: a
   u-resolved acceptance profile per candidate, and a re-scored prediction
   map that uses it. Free first step: re-derive what G98-D's existing bucket
   data already implies before spending GPU time.
2. **Refined evaluation of ours.** Our selector on the LI/LO/LIO/SS grid
   under the Phase-100 protocol (natural EOS, batch sweep, divergence gate),
   directly comparable to the seven Campaign-1 arms.
3. **MoE, extended to TP=2.** Deferred to after the above; the TP=1
   discriminator is folded in as its first step.

## 5. What this box may and may not claim here

Unchanged from `status_vm_branch.md` §1: armed-vs-armed is stable to ~1%,
anything-vs-OFF moves 9–16%. Campaign 1 scores every arm against **stock**
(an armed reference), not against OFF, which is the same discipline and
means the ported numbers and ours are compatible on that axis. Cross-box
absolute comparisons remain inadmissible; Campaign 1's own record already
notes its b8/b16/b32 groups ran on h103 and b64 on h104, with every ratio
taken within a group.
