#!/bin/bash
# Phase 75 S0 -- HARD GATE. Run this FIRST on any new server.
#
# The whole phase rests on an int4 W4A16 weight-only kernel actually executing.
# On h106 the Marlin path dies with `cudaErrorUnsupportedPtxVersion` (its PTX was
# built by a newer toolchain than the driver's JIT accepts: driver 580.65.06 vs a
# CUDA-13.0 torch build). bf16 and FlashInfer-CUTLASS ship SM90 cubins and run fine,
# so this failure is INVISIBLE until you touch Marlin. Catch it in 2 minutes here
# rather than 3 hours into S3.
#
# Also prints WHICH mixed-precision kernel vLLM selects. vLLM's CUDA priority list
# (model_executor/kernels/linear/__init__.py:353) puts Machete AHEAD of Marlin, so on
# SM90 we most likely get MACHETE where the paper used MARLIN. That is a favorable
# deviation (Machete is SM90-native) but it MUST be recorded.
#
# Run BY PATH: `bash scripts/preflight.sh`.
set -u
PHASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$PHASE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
mkdir -p "$PHASE/logs"
LOG="$PHASE/logs/preflight.log"
: > "$LOG"

say(){ echo "$@" | tee -a "$LOG"; }
fail(){ say "[S0] FAIL: $*"; say "[S0] GATE CLOSED -- do not run S1-S6 on this box."; exit 1; }

say "=== Phase 75 preflight ($(date)) on $(hostname) ==="

[ -x "$PY" ] || fail "no venv at $PY (see AGENTS.md: uv venv --python 3.12)"
"$PY" -c "import vllm" 2>/dev/null || fail "vllm not importable from $PY"

say "--- versions ---"
"$PY" - <<'PYEOF' 2>&1 | tee -a "$LOG"
import torch, vllm
from vllm.platforms import current_platform
print(f"vllm            {vllm.__version__}")
print(f"torch           {torch.__version__}  (CUDA build {torch.version.cuda})")
print(f"device          {torch.cuda.get_device_name(0)}")
print(f"capability      {torch.cuda.get_device_capability(0)}  sm90={current_platform.is_device_capability(90)}")
PYEOF
say "driver          $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"

say ""
say "--- GATE 1: does an int4 W4A16 kernel actually EXECUTE? (the PTX trap) ---"
"$PY" - <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys, torch
from vllm import _custom_ops as ops
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.machete_utils import (
    query_machete_supported_group_sizes,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_rows, quantize_weights,
)

atype, wtype, gs = torch.bfloat16, scalar_types.uint4b8, 128
print(f"machete group sizes (bf16): {query_machete_supported_group_sizes(atype)}")

status = {}

# GATE 1a: MACHETE (vLLM's first-choice int4 W4A16 kernel on SM90).
# Layout per tests/kernels/quantization/test_machete_mm.py: pack_rows -> COL-MAJOR.
try:
    K, N, M = 4096, 4096, 16
    w = torch.randn((K, N), dtype=atype, device="cuda") / 10
    _, w_q, w_s, _ = quantize_weights(w, wtype, group_size=gs)
    w_q = pack_rows(w_q, wtype.size_bits, *w_q.shape)
    w_q = w_q.t().contiguous().t()                       # col major (required)
    b = ops.machete_prepack_B(w_q, atype, wtype, w_s.dtype)
    a = torch.randn((M, K), dtype=atype, device="cuda")
    out = ops.machete_mm(a=a, b_q=b, b_type=wtype, b_group_scales=w_s,
                         b_group_size=gs, out_type=atype)
    torch.cuda.synchronize()
    status["MACHETE"] = f"OK  out={tuple(out.shape)}"
except Exception as e:
    status["MACHETE"] = f"{type(e).__name__}: {e}"[:200]

# GATE 1b: MARLIN -- the kernel the PAPER used. Informational only.
# On h106 the *MoE* Marlin path dies with cudaErrorUnsupportedPtxVersion (see
# 74/logs/fc2_marlin_*). We probe the dense int4 Marlin GEMM here to learn whether
# that failure is Marlin-wide or MoE-specific. Distinguish three outcomes:
#   OK / KERNEL-FAILED (PTX etc.) / PROBE-UNAVAILABLE (our harness, not the kernel).
try:
    # Resolve helpers + the op BEFORE the kernel try-block, so a missing API is
    # never misreported as a kernel failure. Signature: benchmarks/kernels/benchmark_marlin_p74.py
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
        marlin_quantize, MarlinWorkspace,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        GPTQ_MARLIN_MAX_PARALLEL, GPTQ_MARLIN_MIN_THREAD_N,
    )
    marlin_gemm = getattr(ops, "marlin_gemm")
    K, N, M = 4096, 4096, 16
    w = torch.randn((K, N), dtype=torch.float16, device="cuda") / 10
    _, mq, ms, zp, g_idx, perm = marlin_quantize(w, scalar_types.uint4b8, gs, act_order=False)
    a = torch.randn((M, K), dtype=torch.float16, device="cuda")
    wsp = MarlinWorkspace(N, GPTQ_MARLIN_MIN_THREAD_N, GPTQ_MARLIN_MAX_PARALLEL)
    try:
        out = marlin_gemm(a, None, mq, None, ms, None, None, zp, g_idx, perm,
                          wsp.scratch, scalar_types.uint4b8, M, N, K,
                          True, False, False, False)
        torch.cuda.synchronize()
        status["MARLIN"] = f"OK  out={tuple(out.shape)}"
    except Exception as e:                       # the KERNEL failed -- meaningful
        status["MARLIN"] = f"KERNEL-FAILED {type(e).__name__}: {e}"[:200]
except Exception as e:                           # our probe failed -- NOT a kernel verdict
    status["MARLIN"] = f"PROBE-UNAVAILABLE ({type(e).__name__}: {e})"[:160] + " -- no verdict on the kernel"

print()
for k, v in status.items():
    ptx = " <<< PTX/driver mismatch" if ("PtxVersion" in v or "unsupported toolchain" in v) else ""
    print(f"  {k:8s} {v}{ptx}")

# The phase needs AT LEAST ONE working int4 W4A16 kernel. Machete is what vLLM
# actually selects (it precedes Marlin in the CUDA priority list), so Machete alone
# is sufficient; a Marlin failure is a recorded deviation, not a blocker.
ok = status.get("MACHETE", "").startswith("OK")
marlin = status.get("MARLIN", "")
print(f"\nusable int4 kernel: {'MACHETE' if ok else 'NONE'}")
if ok and marlin.startswith("KERNEL-FAILED"):
    print("NOTE: dense int4 Marlin FAILS here; vLLM selects Machete anyway. Record the deviation.")
elif ok and marlin.startswith("PROBE-UNAVAILABLE"):
    print("NOTE: Marlin unprobed (test helper missing). Do NOT claim Marlin is broken from this;")
    print("      the known h106 PTX failure is in the *MoE* Marlin path (74/logs/fc2_marlin_*).")
sys.exit(0 if ok else 3)
PYEOF
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] || fail "no int4 W4A16 kernel executes on this box (rc=$rc). \
If the error mentions PTX/toolchain: a CUDA-13 torch build needs a driver new enough to \
JIT its PTX. Use a box whose driver matches the torch CUDA build, or reinstall torch/vllm."

say ""
say "--- GATE 2: models present (offline-safe) ---"
"$PY" - <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys
from huggingface_hub import snapshot_download
missing = []
for m in ["Qwen/Qwen2.5-7B-Instruct"]:
    try:
        p = snapshot_download(m, allow_patterns=["config.json"], local_files_only=True)
        print(f"cached   {m}")
    except Exception:
        missing.append(m)
        print(f"MISSING  {m}  -> huggingface-cli download {m}")
sys.exit(1 if missing else 0)
PYEOF
[ "${PIPESTATUS[0]}" -eq 0 ] || say "[S0] WARN: download the missing model(s) before S1 (needs network; unset HF_HUB_OFFLINE)."

say ""
say "--- GATE 3: harness sampling knob (needed for C3 @ temperature 1.0) ---"
if grep -q 'W7_TEMP' "$REPO/research/52_two_node_e2e/scripts/w7_2node.py"; then
  say "w7_2node.py: W7_TEMP present (T=1.0 rollout sampling available)"
else
  fail "w7_2node.py lacks W7_TEMP -- C3 cannot be measured at temperature 1.0"
fi

say ""
say "[S0] GATE OPEN. Record in results_repro.md: GPU, driver, torch CUDA build, and the"
say "[S0] int4 kernel that executed (MACHETE here; the paper used MARLIN)."
say "[S0] log: $LOG"
