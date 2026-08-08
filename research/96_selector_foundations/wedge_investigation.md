# The draft-boot wedge — investigation record (2026-08-08)

A ~45% fraction of speculative boots hang during engine startup and
never emit a measured cell. This document records what was established,
what was eliminated, and where the investigation is blocked. It exists
so the next person does not repeat the eliminations.

## Established facts

| fact | evidence |
| --- | --- |
| Location: `_initialize_kv_caches → determine_available_memory → profile_run → _dummy_run` (`gpu_model_runner.py:6290`) | `faulthandler` stack via SIGUSR1 |
| The process **spins**: main thread state `R`, `wchan=0` | `/proc/<pid>/task/*/status` |
| It burns a **full core**: 4058 CPU ticks in 40 s wall = 40.6 s CPU | `/proc/<pid>/stat` utime+stime delta |
| The **GPU is idle** (0% util, 22 GB allocated) | `nvidia-smi` during the wedge |
| It is a genuine **hang**, not slowness: no progress after 28 min | extended-wait probe |
| Rate ~45%, independent of configuration, K, batch and GPU | 17 wedges / 38 attempts in W14/D, spread evenly over GPUs 0/2/3 |
| **AR-only boots never wedge** | no draft model, so a much smaller `profile_run` |

Interpretation: a CPU thread spinning at full speed on a CUDA sync
primitive waiting for a GPU event that is never signalled.

## Hypotheses eliminated (each tested, each wrong)

1. **K-dependence** (K4 worse than K2) — `w-off_K2` wedged; K4 passes
   elsewhere. D's retry budget was raised on this reasoning; the budget
   is useful but the justification was wrong.
2. **Lazy Humming JIT during CUDA graph capture** — code-plausible
   (K4 adds capture shapes `[1,3,4]` vs K2's `[1,2]`) but the same
   shapes succeed on another lane; and the hang is *before* capture.
3. **Stale Humming JIT file locks** — 0 lock files in a 115-entry cache.
4. **Leaked shared memory accumulating across boots** — `/dev/shm`
   holds 20 KB of 945 GB, no leftovers.
5. **Concurrent multi-GPU load** ("co-tenant-load-sensitive" ops note) —
   refuted by timing: the first wedges preceded any concurrent run.
6. **A bad GPU lane (GPU1)** — GPU0 wedged 4× in one rerun and D wedged
   evenly across three lanes. GPU1's 76× higher SW-power-capping counter
   is a real anomaly but is not this.
7. **A failing kernel** — `CUDA_LAUNCH_BLOCKING=1` + `TORCH_USE_CUDA_DSA=1`
   raise nothing and change neither rate nor signature.
8. **EPLB collective** (the call immediately preceding the hung line) —
   returns immediately unless `parallel_config.enable_eplb`, which is
   off for dense TP1.
9. **Pathological slowness** — 28 min at 100% CPU with no progress.

## Where it is blocked

The Python frame is known; the **native** frame under it is not.
`faulthandler` reports Python frames only, and both `py-spy --native`
and `gdb` require ptrace, which this box forbids between siblings:

```text
/proc/sys/kernel/yama/ptrace_scope = 1
```

Two ways to finish the diagnosis, both needing elevated access:

- `sudo sysctl -w kernel.yama.ptrace_scope=0` (temporarily), then
  `py-spy dump --native --pid <EngineCore>` on a caught wedge; or
- run one probe under `sudo py-spy`.

The reproduction harness is ready:
`scratchpad/catchwedge{2,3,5}.sh` loop until a wedge occurs and then
dump, so a single privileged run would capture it.

## Why the measured results are still sound

A wedge produces **no observation**. The process dies during engine
startup, before any measured cell is emitted, so no scored quantity can
be biased by it — the cost is wall-clock time, not validity. W14/D's
21/21 registered boots are complete and were re-checked by Gate D0
(closure, exact scheduled-KV, stratum membership, monotonicity,
provenance) after collection.

The mitigation in force is the 400 s progress watchdog plus an
8-attempt budget: measured to lose no registered slot across 17 wedges,
at ~7 min per wedge instead of 25.

## Recommended disclosure if it stays unresolved

State it as an environment limitation with the retry protocol as the
disclosed mitigation, and note explicitly that a wedge yields no
observation and therefore cannot bias a result. Do not describe the
retry protocol as a fix.
