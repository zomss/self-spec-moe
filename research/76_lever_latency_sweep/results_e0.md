# Phase 76 E0 results — infrastructure-parity preflight: ALL THREE GATES OPEN

Box: cloud-9ezI3Q (8×H100, driver 570.211.01), GPUs **0,1,6,7 only** (2-5
reserved). vllm 0.1.dev17860, torch 2.11.0+cu129, flashinfer 0.6.12.
Tables: `data/e0_table_{dense,moe,mla}.md`; per-arm logs `logs/e0_*.log`;
runner `scripts/e0_preflight.sh` (+ `e0_assert.py`, `e0_probe_cfg.py`,
`e0_rerun_fp8block.sh`).

## Verdict

**Dense 9/9 PASS · MoE-GQA 9/9 PASS · MLA 6/7 PASS + 1 legitimate BROKEN-CELL.**
Every sweep arm rides the same attention backend + FULL_AND_PIECEWISE cudagraph
as its L0 denominator, with only the intended lever delta, asserted from logs —
except the two cells recorded below. The deliberate trap arm (d_kvq_e5m2)
flipped FLASH_ATTN→FLASHINFER and the checker caught it, validating the method.

## Kernel-path map (what each lever arm actually engages)

| lever arm | dense 7B (TP1) | MoE-GQA 30B (DP4+EP4) | MoE-MLA V2-Lite (DP4+EP4) |
|---|---|---|---|
| L0 bf16 | FLASH_ATTN | FLASH_ATTN + TRITON MoE | FLASH_ATTN_MLA + TRITON MoE |
| L1a weight-only | Machete (auto) / Marlin (forced) — asserted via chooser probe | MARLIN Fp8 MoE | MARLIN Fp8 MoE ✓works |
| L1b W8A8-fp8 | quant=fp8 (cutlass), FA3 kept | **FLASHINFER_CUTLASS** Fp8 MoE | **FLASHINFER_CUTLASS** ✓works |
| L1c KV fp8_e4m3 | FA3 kept (fast path) | FA3 kept | **BROKEN: flips to FLASHMLA** |
| L2 window 512 | FA3 kept, cfg plumbed, max_len uncapped | same | cfg plumbed; kernel effect UNVERIFIED |
| L3 skip50 (dummy) | 14 layers ✓ | 24 layers ✓ | 14 layers ✓ |
| L4 local-route / skip-a2a | n/a | **markers fire** (AgRs path) | (not yet armed; same path expected) |

All MoE arms: `MoEPrepareAndFinalizeNaiveDPEPModular` — the AgRs dispatch path
(the collective L4 cuts), identical across arms.

## E0 findings (each one would have silently corrupted the sweep)

1. **Pure TP4+EP has NO dispatch collective.** First run used TP4+EP4 →
   `MoEPrepareAndFinalizeNoDPEPModular`: TP already replicates tokens, so there
   is no a2a at all — L4 was a no-op and the fabric axis vacuous. The MoE
   fabric MUST be attention-DP4+EP4 (P74/P24's). Consequence for the sweep
   primitive: the offline `LLM` API refuses internal DP (`llm.py:298`), so MoE
   cells run via `vllm serve -dp4` + client, not `bench latency`.
2. **MLA + fp8 KV flips the attention backend** (FLASH_ATTN_MLA → FLASHMLA) —
   the P74-e5m2 trap class, now on MLA. As-is the L1c×MLA cell confounds lever
   with backend. Remedy if the cell is wanted: pin `VLLM_ATTENTION_BACKEND=
   FLASHMLA` on BOTH arms of the ratio; else mark excluded.
3. **Trap control validated the checker**: `fp8_e5m2` KV fell off FA3 to
   FLASHINFER and was flagged; `fp8_e4m3` stays on FA3 (dense + GQA) as P74
   documented.
4. **DeepGEMM runtime JIT silently produces no cubin when `nvcc` is not on the
   worker PATH.** fp8_per_block routes its grouped GEMM through FlashInfer's
   embedded DeepGEMM, which `popen()`s bare `nvcc` when CUDA_HOME is unset,
   captures only stdout ("NVCC compilation took 3 ms", empty log), then dies
   with `Assertion failed: !cubin.empty()`. Fix (env_e76.sh):
   `TRTLLM_DG_NVCC_COMPILER=/usr/local/cuda/bin/nvcc`, `CUDA_HOME`,
   cuda bin on PATH. Its cache also defaults to `~/.tensorrt_llm` (root FS) →
   `TRTLLM_DG_CACHE_DIR=/data/smcho/cache/trtllm_dg`.
5. **FlashInfer fused_moe JIT (183 cutlass objects) cannot be built by 4
   racing workers on this box** — and the root FS fills unpredictably (nvcc
   temp → "No space left on device", plus one empty-cubin poisoning of the DG
   cache at the ENOSPC moment). Prebuilt once single-process (`ninja` in
   `~/.cache/flashinfer/.../fused_moe_90`, `TMPDIR=/data/smcho/tmp`) →
   `fused_moe_90.so` cached; both fp8block arms then pass.
6. **Window override plumbing verified on all three models** via config probe
   (`cache_config.sliding_window=512` reaches every Attention layer through the
   fallback at `attention.py:233-238`; `max_model_len` NOT capped). Caveat: on
   MLA this proves config plumbing only — whether the MLA decode kernel honors
   the window needs a functional latency-vs-ctx sanity cell in E1 (a window
   that changes nothing vs bf16 at 32k = silently ignored).
7. **LOCAL_ROUTE / SKIP_A2A were silent branches** — added `info_once` markers
   in `all2all.py` (gather path); both fire under DP4+EP4. The sweep's honest
   L4 arm is LOCAL_ROUTE (real expert_map routing); SKIP_A2A is the
   shape-preserving timing stand-in (P24 B2a).
8. **Serve-log append hazard**: the server holds its log open without O_APPEND,
   so status markers appended to the same file get overwritten by post-kill
   shutdown writes — markers live in `<log>.status` (cost E0 one false-negative
   round).
9. Box changes recorded: user moved `HF_HOME=/data/smcho/huggingface` (old
   `~/.cache/huggingface` GONE), `VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR` to
   /data via `~/.bashrc`; env_e76.sh repeats them for non-login contexts.

## Consequences for the E1 sweep design

- MoE/MLA cells: `vllm serve -dp4 --enable-expert-parallel` + a client that
  measures the two-output-length slope (batch is GLOBAL, keep multiples of 4);
  dense cells: `bench latency` TP1 as in P75-E1.
- L1c on MLA: pin FLASHMLA on both arms or drop the cell (decide at E1).
- Add ONE functional sanity cell per lever family at long ctx before the grid
  (window must move t_step at 32k; skip50 must scale ~linearly at b32/2k).
- Everything else (kernel choices per arm, dummy-load for L3, DP4 fabric) is
  settled above and encoded in `env_e76.sh` + the runner.
