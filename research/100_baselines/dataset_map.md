# Dataset map: what every in-scope baseline actually evaluates on

Record of 2026-08-16. Sources: paper full texts fetched this session
(arXiv HTML/ar5iv); our side read from
`98_selector_demo/scripts/w98d2_regime_datasets.py` (the Phase 88 E0
canonical loader). Per-paper numbers below are as extracted today —
**re-verify exact sample counts against the PDFs before anything is
preregistered**, since HTML extraction can drop table details.

---

## Our regimes, for reference

| regime | data | context | gen | batch | T |
| --- | --- | --- | --- | --- | --- |
| R1 | GSM8K + AIME mixed, CoT prompt | short | 1024 | 1 | 0 |
| R4 | CNN/DM **packed multi-article** to ~8K | 8K | default | 8 | 0 |
| R5 | NQ-open question over **C4 filler docs** | 14K | default | 8 | 0 |
| R5cot | AIME over C4 filler docs | 14K | 3072 | 8 | 0 |
| R6 | GSM8K, "answer concisely" | short | 256 | 32 | 0 |
| R8 | AIME CoT | short | 2048 | 16 | 1.0 |
| RKS (dormant) | Phase-57 prompts over C4 filler | 16K | 160 | 1 | 0 |

Unused but already in the loader: R2 MT-Bench, R3 HumanEval, R7 WMT14
de-en — three of Spec-Bench's six subtasks are a config flag away.

## Their evaluation sets

### Layer-skip family

| paper | datasets | samples | gen len | T | models |
| --- | --- | --- | --- | --- | --- |
| **Self-SD** (2309.08168) | CNN/DM (1-shot), XSum (1-shot), HumanEval; app'x: GSM8K, MT-bench | 1000 / 1000 / all | 512 target; K=12 | 0 and 0.2 (0.6 code) | LLaMA-2-13B/70B, CodeLLaMA-13B |
| **SWIFT** (2410.06916) | CNN/DM (1-shot), GSM8K (5-shot), TinyStories, HumanEval; analysis: Alpaca, WMT14, NQ (500 ea.) | 1000 ea. | **64** / 64 / 128 / 512 | 0 (0.6 pass@10) | LLaMA-2-13B/70B, CodeLLaMA-13B/34B |
| **CLaSp** (2505.24196) | **Spec-Bench**: MT-bench, WMT14 de-en, CNN/DM, NQ, GSM8K, DPR | 80 per subtask | max seq 1024 | 0 and 1 | LLaMA3 8B/70B (+13B/405B variants) |
| **KnapSpec** (2602.20217) | reasoning: **AIME24/25, MMLU-Pro**; summarization: **GovReport (~16K in), PG19, BookSum** | not extracted | **32K** (AIME), 4K (MMLU-Pro) | 0; sampling 0.7/k50/p0.95 | reasoning: **Qwen3-4B/8B/14B/32B**; summ.: Llama3-1B/3B/8B/70B |

### KV-sparsity family

| paper | datasets | context | gen len | batch | models |
| --- | --- | --- | --- | --- | --- |
| **MagicDec** (2408.11049) | **PG-19** (main); RULER (passkey, CWE, QA-1) | 1K–100K (8 points) | **96** | **16–256** | Llama-3.1-8B, Llama-2-7B-32K, Qwen-2.5-7B/32B, Mistral-7B, … |
| **TriForce** (2404.11912) | PG-19, NarrativeQA | 122K–127K (to 512K) | 256 (to 2048) | 1 (pairs to 10) | Llama2-7B/13B-128K, LWM-Text |
| **QuantSpec** (2502.10424) | PG-19, ∞Bench-Sum (~171K), Multi-LexSum (~90K) | 4K–128K | **90** | 1 | Llama-2-7B-32K-Inst, LWM-Text-Chat-128k |

### Quantized family

| paper | datasets | gen len | batch | baseline |
| --- | --- | --- | --- | --- |
| **QSpec** (2410.11305) | accel: GSM8K, MATH, MBPP, HumanEval, ShareGPT, LMSys (100 ea.) | 200 | 8–32 | **W4A16/W4A4/W16A16 deployments + EAGLE** — never FP16 AR |

QSpec's baseline column confirms the border-exclusion: its comparison
universe is quantized serving, not an FP16 target.

---

## What the map shows

### 1. The field has two data currencies, and they split by family

**Skip family → short-context instruction/reasoning sets** (CNN/DM 1-shot,
GSM8K, HumanEval, Spec-Bench), generation 64–512 tokens.
**KV family → PG-19 long prose** plus a long-QA garnish (RULER,
NarrativeQA), generation 90–256 tokens. KnapSpec is the bridge — it moved
the skip lever onto the KV family's data (GovReport/PG19/BookSum) *and*
onto long-generation reasoning (AIME at 32K out). That bridging is the
paper's actual novelty in evaluation terms, and reproducing it means
running on both currencies.

### 2. A direct-overlap cell we did not know existed: KnapSpec × Qwen3-8B

KnapSpec's reasoning column runs **Qwen3-8B — our exact model — on
AIME24/25 and MMLU-Pro**. Our R1/R8 already draw from AIME. This is the
single most valuable reproduction cell in the phase: same model, same task
family, their published number on one side and our measured lattice on the
other, no scale excuse in either direction. Two protocol deltas to close:
their AIME runs to 32K generated tokens (ours cap at 1024–3072) and their
dynamic depth uses tau_conf=0.7 with re-optimization every T=64 steps.

### 3. Every shared dataset is used differently — the traps are in protocol

* **CNN/DM**: they feed one article, 1-shot, and generate 64–512 tokens;
  our R4 packs multiple articles to 8K. Same dataset name, different
  experiment. A faithful Self-SD/SWIFT cell must be single-article.
* **Generation lengths are tiny across the field**: SWIFT 64, QuantSpec 90,
  MagicDec 96, QSpec 200, TriForce 256. Our protocol is decode-currency
  with `ignore_eos` and long equal-work decodes. Short-gen speedup numbers
  fold in warm-up and per-request overhead amortization; SWIFT's on-the-fly
  optimizer in particular pays its cost inside those 64 tokens. **Both
  protocols must be run**: theirs for fidelity (B3'b needs their band),
  ours for cross-baseline comparability. Register this fork explicitly.
* **Our R5/R5cot/RKS long contexts are C4 filler** — a construction that
  exists in no baseline paper. Fine for lattice-internal comparisons,
  inadmissible for reproduction. Long-context cells for Phase 100 need the
  real corpora: PG-19, GovReport.
* **CLaSp's Spec-Bench is 80 samples/task at max-seq 1024** — small and
  short; its 1.3–1.7x lives there. Three of its six subtasks (MT-Bench,
  HumanEval-adjacent code, WMT14) are already dormant regimes in our
  loader.

### 4. Gaps on our side (datasets we have never run)

PG-19, GovReport, BookSum, XSum, MMLU-Pro, AIME at 32K generation,
NarrativeQA/RULER. Of these, **PG-19 and GovReport are load-bearing**:
PG-19 is the KV family's shared currency (MagicDec + TriForce + QuantSpec
+ KnapSpec all use it), GovReport is KnapSpec's headline cell and Stage
B's 16K target — Stage B should be built on real GovReport, not C4 filler.

## Proposed Phase 100 evaluation grid (to preregister)

| block | data | protocol | serves |
| --- | --- | --- | --- |
| **P1** PG-19 | 8K/16K/32K prefill, gen 96, batch {1, 8, 32} | MagicDec's | MagicDec row (B3'a), later QuantSpec/TriForce cites |
| **P2** AIME24/25 on Qwen3-8B | long generation (their 32K budget, ours capped where needed), b1 | KnapSpec's | the direct-overlap cell |
| **P3** GovReport | ~16K real input, summarization | KnapSpec's | Stage B; KnapSpec summ. column |
| **P4** CNN/DM single-article 1-shot + GSM8K + HumanEval | their short-gen budgets AND our equal-work budget | Self-SD/SWIFT | skip-family fidelity (B3'b) |
| **P5** Spec-Bench 80-sample suite | max-seq 1024 | CLaSp's | CLaSp; R2/R3/R7 wake up |

Sample sizes, exact splits, and the dual-protocol (their-gen vs equal-work)
scoring rule go into the phase barrier before the first scored boot.

## Granularity resolution (closes the survey's open item)

Verified this session from the papers: **Self-SD, SWIFT, KnapSpec skip
attention and MLP sublayers independently** (SWIFT's config figures mark
skipped attention red and skipped MLP blue); **CLaSp skips whole
transformer layers** (50–60% of them). Stage A's one sub-block mask
implementation therefore serves three of the four skip baselines, and
CLaSp needs only our existing whole-layer mask plus its dynamic
reselection machinery.
