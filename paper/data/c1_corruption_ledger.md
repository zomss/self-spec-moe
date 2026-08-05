# cache-key collision: corruption ledger

Root cause: `SpeculativeConfig.compute_hash()` omitted `draft_model_config`,
so two runs sharing a target model but differing in the DRAFT checkpoint
wrote the same `torch_compile_cache/<key>/` directory and loaded each
other's compiled `draft_model` graph. Fixed in `vllm/config/speculative.py`
(commit aaad1c14e), verified by A/B under one shared cache root.

Detection: two independent screens, because they catch different damage.
  - acceptance screen  -- finds corruption that WRECKS drafting (tau << K+1)
  - collision screen   -- parses logs for two checkpoints writing the SAME
                          cache directory; this is the ONLY one that finds
                          cost-side corruption, where tau stays plausible

| artifact | collision | damage | status |
|---|---|---|---|
| stage_a dense w8int8 | W4A16 | tau 1.04 (acceptance) | RESOLVED — re-measured, tau 2.98 |
| stage_a llama w8int8 | W4A16 | tau 1.73 (acceptance) | RESOLVED — re-measured, tau 2.97 |
| stage_a dense fp8dyn | W4A8-gptq | S 0.15 (COST, tau fine) | RESOLVED — re-measured, S ~1.0 |
| stage_a dense w4a8cut | FP8-dynamic | none — was the primer | VERIFIED clean (dS 1.37%) |
| stage_a dense w4a16 | W8INT8 | none — was the primer | VERIFIED clean (control, dS 2.83%) |
| stage_a llama w4a16 | W8INT8 | K2/2k-ctx cells ~3x cost-slow | RESOLVED — v2 re-measure; llama Stage-A winner map built from repaired data |
| stage_b llama w8int8_k2 | W4A16 | accept 1.84, toks -60% | RESOLVED — re-run, accept 2.83 |
| stage_b llama w4a16 | W8INT8 | cost both directions (b8 -9%, b32 +6%) | RESOLVED — k2/k4 re-run; 23->24/33 beats-AR, SUSPECT cell (S 6.65) resolves to real 1.24x win; winning-config set unchanged |
| stage_a mla w4a16 / w8chan | each other | w4a16 S off up to ~4.6x (COST) | RESOLVED — v2 re-measure; **4 Stage-A winners changed** (3 OFF and 1 w8chan cell -> w4a16) |
| stage_a moe w4a16 / w8chan | each other | minor | RESOLVED — v2 re-measure; 0 winners changed |
| c2 oracle llama (12 dirs) | W4A16/W8INT8 | w8int8 rows corrupt | RESOLVED — full post-fix re-measurement (`oracle_llamafix`); paper/c2.md corrected end-to-end (see its correction note) |
| c2 oracle llama R2 | none — isolated roots | none | VERIFIED clean |
| stage_a q3_32b, stage_b {dense,mla,moe,q3_32b} | none | none | VERIFIED clean |
| p75 e2 quant arms (hardware axis) | W4A16-asym / W4A16-sym / W8INT8 share one dir | unknown (W4asym<->W4sym graphs near-identical, W8<->W4 differ) | OPEN — flagged by the global screen; feeds c1's hardware-axis numbers; needs re-measure or A/B to bound |
| p92 e2_sweep (RL drafter refresh) | 92-drafter-step{8,16} <-> static W4A16 (2 dirs) | unknown (same graph shape, likely benign) | OPEN — flagged by the global screen; feeds the refresh-vs-static comparison |
| all other phases WITH retained logs (70, 74, 76, 79-87, 93, 94) | — | — | SCREENED CLEAN (global screen 2026-08-05: 923 draft boots, 574 cache dirs, full-path keyed; the only sharing found is the documented 93/94 pre-fix set + p75/p92 above) |
| phases WITHOUT retained logs (71-73, 77, 78, 88 regime suite, 89-91) | unscreenable | unknown | OPEN — no boot logs kept; the regime-suite (88) rows can only be re-verified by re-measurement |
| oracle R2 replications, all four arches | none — isolated roots verified in-log (36/36 orcR2 dirs on dense; mla/moe likewise) | none | VERIFIED clean (acceptance screen: 0 rows tau<2.0 at K2 on any arch) |

Full-log collision screen, llama stage_a (51 draft boots over 4 logs):
only W4A16<->W8INT8 ever shared a cache directory (2 hashes); fp8dyn sits
alone on its own hash, and every bf16-self-draft arm (window/skip/kvq)
is clean. The llama Stage-A winner map (built post-repair, 16 cells)
awards 5 cells to fp8dyn on screened-clean data.

Impact (supersedes the earlier "no winner changed" claim, which predated
the MLA/MoE and Stage-B re-measures): dense and MoE Stage-A winner maps
are unchanged and llama's winning-config SET is unchanged, but the MLA
Stage-A map changed in 4 of 14 cells (all to w4a16, all high-batch —
softening MLA's Stage-A OFF-dominance and closing part of the Stage-A/
Stage-B cross-layer gap), and llama Stage-B moved 5 cells' beats-AR
verdicts (net 23->24/33), including the physical-sanity SUSPECT cell
resolving into a real win. w8int8 moves from "broken" to near-lossless-
acceptance parity; fp8dyn from a 6.7x slowdown to parity-to-mild-win.
