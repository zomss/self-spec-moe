# W4A16 decode kernel: which of vLLM's kernels is best for the draft?

Date: 2026-08-12
Status: answered. **Humming is fastest at every M; Marlin second; Machete —
what we run today — is third.** The switch is a kernel-priority change, not new
code, worth ~3.2-3.4 ms/step. A specialised kernel is still worth ~2.9 ms/step
after that.

## Which kernels can even implement our config

Asking the registry directly for the draft's config (group_size 128, `uint4b8`,
bf16 activations, SM90), CUDA priority order:

| kernel | viable | reason |
| --- | --- | --- |
| `CutlassW4A8` | no | FP8 activations only |
| `Machete` | **yes** | selected today |
| `AllSpark` | no | no SM90 support (also `uint8b128` only) |
| `Marlin` | **yes** | |
| `Humming` | **yes** | JIT; needs `ninja` on PATH |
| `Conch` | no | package not installed — **environment, not merit** |
| `Exllama` | no | fp16 activations only |
| `TritonW4A16` | **yes** | |

W4A8 / W8A8 / W8A16 / W4A4 are separate quantization schemes, not alternative
kernels for ours.

## Result

Per-step W4A16 GEMM cost, 28 layers x 4 chain forwards, CUDA-graph timed:

| M | machete (today) | marlin | **humming** | triton | ideal |
| --- | --- | --- | --- | --- | --- |
| 1 | 9.55 ms | 6.67 | **6.32** | 35.70 | 3.33 |
| 4 | 9.64 | 6.71 | **6.28** | 35.27 | 3.33 |
| 8 | 9.73 | 6.72 | **6.27** | 35.26 | 3.33 |
| 32 | 9.22 | 8.43 | **7.11** | 35.32 | 3.33 |

Per shape, µs/call and % of that shape's own bandwidth bound:

| M=1 | ideal | machete | marlin | humming |
| --- | --- | --- | --- | --- |
| qkv_proj | 3.87 | 14.44 (27%) | 9.82 (39%) | **9.40 (41%)** |
| o_proj | 2.58 | 13.76 (19%) | **7.72 (33%)** | 8.49 (30%) |
| gate_up_proj | 15.49 | 36.74 (42%) | 25.96 (60%) | **25.46 (61%)** |
| down_proj | 7.75 | 20.31 (38%) | 16.09 (48%) | **13.11 (59%)** |

| M=32 | ideal | machete | marlin | humming |
| --- | --- | --- | --- | --- |
| qkv_proj | 3.87 | 16.14 (24%) | 13.53 (29%) | **11.32 (34%)** |
| o_proj | 2.58 | 14.66 (18%) | 9.98 (26%) | **9.49 (27%)** |
| gate_up_proj | 15.49 | 29.85 (52%) | 30.51 (51%) | **27.64 (56%)** |
| down_proj | 7.75 | 21.66 (36%) | 21.24 (36%) | **15.00 (52%)** |

Humming wins 7 of 8 (Marlin takes `o_proj` at M=1) and its margin is widest
exactly where Marlin was weakest — `down_proj` (59% vs 48%) and M=32
(7.11 vs 8.43 ms/step, 16%).

`TritonW4A16` is a portability fallback: 5-8% of bound, 5x slower than Humming.

## Two findings about the method, not the kernels

**Stopping at "Marlin beats Machete" would have shipped the second-best
kernel.** The first comparison covered two of eight kernels. Widening it to
everything the registry admits changed the answer.

**Two of the three exclusions were environment artifacts, not facts about the
kernels.** Humming failed with `Ninja is required to load C++ extensions` —
`ninja` is present in `.venv/bin` and the engine's boot environment puts it on
PATH, but the probe did not. Adding it made Humming the winner. Conch remains
excluded only because it is not pip-installed. Neither is evidence about kernel
quality, and one of them already overturned the recommendation once.

## Measurement notes

The repo's `benchmark_machete.py` cannot answer this. Its `bench_fns` times a
Python loop, so every call carries dispatch; fitting time against weight bytes
gives per-call overheads of 12.58 µs (machete), 24.34 (marlin), 4.41
(torch.matmul) — larger than the entire bandwidth bound for three of four
shapes, with Marlin's fitted slope implying 5x HBM, which is impossible.

X10 therefore drives each kernel through the real `CompressedTensorsWNA16`
path — `create_weights` with `choose_mp_linear_kernel` patched per kernel, then
that kernel's own `process_weights_after_loading` repack — and times
`apply_weights` under CUDA-graph capture. A hand-rolled `nn.Module` does not
work: the kernels need vLLM's parameter wrappers (`input_dim`/`output_dim`),
`output_partition_sizes`, `params_dtype`, `has_bias`, and an initialised TP
group under `set_current_vllm_config`. Graph capture matters because the draft
chain is captured in production (X6) and because per-call dispatch would
otherwise dominate.

## Correction to an earlier figure

This phase previously reported the GEMM at "65% of bandwidth-bound, ~3.0 ms/step
headroom". That divided the whole 6.1 GB checkpoint by layer count, but the
checkpoint includes the bf16 `lm_head` and embeddings, which are neither
quantized nor touched per layer. The quantized projections are 99.5 MB/layer →
**2.79 GB active**, 11.14 GB/step of traffic, ideal 3.33 ms against 8.69
measured = **38%, headroom 5.36 ms/step**.

## At M<=32 activation precision is irrelevant

A 4096-wide bf16 activation row is 8 KB against tens of MB of weights — 0.02% of
traffic. **W4A8 and W4A16 have identical decode cost.** A-precision pays only at
prefill or large-batch verify. W4A8 is therefore not a route to this saving: it
would change the checkpoint and the draft's acceptance rate while buying nothing
where the draft is bottlenecked. For decode cost the quantization lever is
purely **weight bits**.

## Recommendation

1. Select **Humming** for the draft's W4A16 layers (priority change, gated to
   the draft so serving defaults are untouched). ~3.2 ms/step at M<=8, ~2.1 at
   M=32. Ensure `ninja` is on PATH in every boot path, since Humming JIT-compiles
   and silently falls back without it.
2. Re-measure in-engine: the microbenchmark must be confirmed to transfer, and
   Humming's JIT warmup interacts with capture (`_prewarm_jit_linear_kernels`
   exists for exactly this reason).
3. Before calling the selection final, close the two gaps this survey exposed —
   install Conch and measure it, and sweep Marlin's `use_atomic_add` /
   `use_fp32_reduce`, which target small-M behaviour and were left at defaults.
4. Only then consider a specialised decode kernel, against a measured Humming
   baseline, for the remaining **~2.9 ms/step**.

---

## X11 — the two gaps closed (Conch installed, Marlin flags swept)

Per-step W4A16 GEMM (28 layers x 4 forwards), ms:

| M | machete | marlin | +atomic | −fp32 | both | humming | conch | triton | ideal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 9.56 | 6.62 | 6.58 | 6.60 | 6.63 | **6.33** | 329.58 | 35.25 | 3.33 |
| 4 | 9.62 | 6.74 | 6.71 | 6.72 | 6.72 | **6.26** | 331.92 | 35.18 | 3.33 |
| 8 | 9.52 | 6.76 | 6.75 | 6.77 | 6.80 | **6.23** | 334.52 | 35.17 | 3.33 |
| 32 | 9.18 | 8.43 | 8.42 | 8.41 | 8.40 | **7.08** | 376.32 | 35.32 | 3.33 |

**Marlin's flags do nothing.** All four combinations land within 0.15 ms (~2%)
of each other and of the default.
`use_atomic_add` is gated off by the `n >= 2048` rule for all four projections
(n = 4096..24576); forcing it on changes nothing, so that heuristic is correct
here. `use_fp32_reduce` is free. There is no hidden headroom inside Marlin.

The first run of this sweep had a bug worth recording: the `finally` restoring
`ops.marlin_gemm` was missing, so forced flags LEAKED into every later variant —
`marlin_nofp32` silently also carried `atomic`, and the last variant layered all
of them. The conclusion survived (every variant was within 1% of the unpatched
default, which runs first), but the labels did not mean what they said. Rerun
with the restore in place; the table above is the clean version.

**Conch is a non-starter** at 330 ms/step, 50x slower than Humming. It is a
portability path for hardware without tuned kernels.

## X12 — in-engine, and the finding that decides it

**Humming wedges with whole-chain capture.** It selected correctly and its JIT
guard fired on all 112 layers (`pre-warmed JIT linear kernels on 112 Humming
layers x 18 M-tiles (eager, pre-capture)`), then hung immediately after
`torch.compile took 17.10 s` — GPU pinned at 100%, no output for 2h10m, killed.

So this is **not** the first-use dispatcher deadlock `_prewarm_jit_linear_kernels`
exists to prevent; that guard worked. It wedges later, at CUDA-graph capture of
the draft chain. Humming and the whole-chain graph do not compose, and the
existing mitigation does not cover it.

Verified in-engine, whole-chain captured, kernel resolution read back from the
log rather than assumed:

| kernel | in-engine median | resolved | wholechain |
| --- | --- | --- | --- |
| machete (today) | 25.06 ms | MacheteLinearKernel | captured |
| **marlin** | **22.36 ms** | MarlinLinearKernel | captured |

machete − marlin = **2.70 ms** in-engine vs **2.94 ms** in the microbenchmark —
within 8%, so the microbenchmark transfers for kernels that work.

## Decision: Marlin

Humming is 0.3 ms/step faster on isolated GEMMs and **unusable** with the
runtime adopted in X6. Whole-chain capture is worth 10-13 ms/step; trading it
for 0.3 ms would be a large net loss. Marlin is verified end to end.

This is why the in-engine check was not a formality: the microbenchmark winner
does not boot.

Superseded: the earlier recommendation to select Humming (X10), which rested on
isolated-GEMM timings only.
