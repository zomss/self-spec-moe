#!/bin/bash
# QUANT-EFFECTIVENESS SWEEP (plain model forward, NOT self-spec) — the 4 READY cells:
#   dense/MoE W8A8 (compute-cut) + dense/MoE KV-fp8 (KV read-cut), vs bf16 baseline.
# Harness: `vllm bench latency` (native fp8, on-the-fly, no checkpoint for these cells).
# Regimes isolate the binding term:
#   decode_long  = long ctx, small batch  -> KV-I/O-bound (KV-fp8 should win, esp MoE)
#   decode_short = short ctx, small batch  -> weight-I/O-bound (dense: weight cut shows)
#   prefill_big  = long prompt, big batch  -> compute-bound (W8A8 compute cut should win)
# Item-1 (dense weight-only fp8) is NOT here: needs a CT W8A16 ckpt — see
# quant_kernel_readiness.md "the one decision".
# DO NOT run while GPUs 4-7 are reserved by others.
set -u
PHASE=/data/smcho/self-spec-moe/research/74_draft_step_cost
export CUDA_VISIBLE_DEVICES=4,5,6,7
export HF_HUB_OFFLINE=1 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
LOGD="$PHASE/logs"; DATED=$(date +%H%M%S)
VLLM=/data/smcho/self-spec-moe/.venv/bin/vllm

# arm: name model par quantflag kvflag  input output batch
run(){
  local NAME=$1 MODEL=$2 PAR=$3 QUANT=$4 KV=$5 IN=$6 OUT=$7 B=$8
  local LOG="$LOGD/qs_${NAME}.log"
  echo "[qs] >>> $NAME  ($MODEL $PAR q='$QUANT' kv='$KV' in=$IN out=$OUT b=$B)"
  timeout 900 "$VLLM" bench latency \
    --model "$MODEL" $PAR $QUANT $KV \
    --input-len "$IN" --output-len "$OUT" --batch-size "$B" \
    --num-iters-warmup 2 --num-iters 5 \
    --gpu-memory-utilization 0.9 --max-model-len $((IN+OUT+256)) \
    > "$LOG" 2>&1
  echo "[qs] $NAME -> $(grep -hiE 'Avg latency' "$LOG" | tail -1)"
  # confirm the intended kernel engaged:
  grep -hiE 'Fp8 MoE backend|CutlassFP8|FLASHINFER_CUTLASS|Marlin|Machete|attention backend|FlashAttention|kv_cache_dtype=' "$LOG" | tail -2
}

DENSE="Qwen/Qwen3-8B";        DPAR="--tensor-parallel-size 1"
MOE="Qwen/Qwen3-30B-A3B";     MPAR="--tensor-parallel-size 4 --enable-expert-parallel"

# ---- DENSE (Qwen3-8B) : bf16 / W8A8(fp8) / KV-fp8 across regimes ----
run d_bf16_declong   "$DENSE" "$DPAR" ""                        ""                        16384 128 1
run d_w8a8_declong   "$DENSE" "$DPAR" "--quantization fp8"      ""                        16384 128 1
run d_kv_declong     "$DENSE" "$DPAR" ""                        "--kv-cache-dtype fp8_e4m3" 16384 128 1
run d_bf16_decshort  "$DENSE" "$DPAR" ""                        ""                          512 128 1
run d_w8a8_decshort  "$DENSE" "$DPAR" "--quantization fp8"      ""                          512 128 1
run d_bf16_prefill   "$DENSE" "$DPAR" ""                        ""                         4096   8 32
run d_w8a8_prefill   "$DENSE" "$DPAR" "--quantization fp8"      ""                         4096   8 32

# ---- DENSE weight-only fp8 (Item 1): W8A16 ckpt + force-marlin (dequant, READ-CUT only) ----
# quant detected from the checkpoint config (no --quantization flag). Marlin gated off on
# SM90 without this env. Expect a WIN at weight-bound decode (decshort), weak at KV-bound.
export VLLM_TEST_FORCE_FP8_MARLIN=1
WO_CK="/data/smcho/ckpts/Qwen3-8B-W8A16-FP8"
run d_wo_declong     "$WO_CK" "$DPAR" ""                        ""                         16384 128 1
run d_wo_decshort    "$WO_CK" "$DPAR" ""                        ""                           512 128 1
unset VLLM_TEST_FORCE_FP8_MARLIN

# ---- DENSE weight-only INT4 (Machete, 4x read cut) — the STRONGER weight-only lever ----
# quant detected from ckpt; int4 Machete/Marlin are NOT gated on Hopper (no force env).
# Head-to-head vs d_wo_* (fp8 2x) and d_bf16_* : does the 4x cut win bigger at weight-bound?
I4_CK="/data/smcho/ckpts/Qwen3-8B-W4A16-INT4"
run d_int4_declong   "$I4_CK" "$DPAR" ""                        ""                         16384 128 1
run d_int4_decshort  "$I4_CK" "$DPAR" ""                        ""                           512 128 1

# ---- MoE (Qwen3-30B-A3B, EP4) : bf16 / W8A8(fp8_per_block) / KV-fp8 across regimes ----
run m_bf16_declong   "$MOE" "$MPAR" ""                          ""                         16384 128 1
run m_w8a8_declong   "$MOE" "$MPAR" "--quantization fp8_per_block" ""                      16384 128 1
run m_kv_declong     "$MOE" "$MPAR" ""                          "--kv-cache-dtype fp8_e4m3" 16384 128 1
run m_bf16_prefill   "$MOE" "$MPAR" ""                          ""                         4096   8 32
run m_w8a8_prefill   "$MOE" "$MPAR" "--quantization fp8_per_block" ""                       4096   8 32
echo "[qs] DONE ($(date +%H:%M:%S))"
