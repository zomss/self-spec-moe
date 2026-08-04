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
| stage_b llama w8int8_k2 | W4A16 | accept 1.84, toks -60% | RESOLVED — re-run, accept 2.83 |
| stage_a mla w4a16 / w8chan | each other | unknown | RE-MEASURING |
| stage_a moe w4a16 / w8chan | each other | unknown | PENDING (TP2) |
| stage_a llama w4a16 | W8INT8 | unknown (partner) | RE-MEASURING |
| stage_b llama w4a16 | W8INT8 | unknown (partner) | PENDING |
| c2 oracle llama (12 dirs) | W4A16/W8INT8 | w8int8 rows corrupt | RE-MEASURING (offset 0, fixed) |
| c2 oracle llama R2 | none — isolated roots | none | VERIFIED clean |
| stage_a q3_32b, stage_b {dense,mla,moe,q3_32b} | none | none | VERIFIED clean |

Impact so far: every correction changed a lever's CHARACTERIZATION, none
changed a winner. C1's "no universal lever" claim and both winner maps are
unchanged. w8int8 moves from "broken" to near-lossless-acceptance parity;
fp8dyn from a 6.7x slowdown to parity-to-mild-win.
