# Current Conclusion and Next Step

Date: 2026-06-23

## Current Run Conclusion

Phase 02 established four points:

1. **Default single-node EP is weak for Self-MoE-spec.** On GPUs 6 and 7,
   DP2+EP allgather/reducescatter with dummy `Qwen/Qwen1.5-MoE-A2.7B` barely
   reaches break-even under optimistic local-draft estimates and cannot reach
   1.3x. This matches the Phase 01 expectation.
2. **Forced same-host IB/GDRDMA is unstable here.** NCCL can select `NET/IB`
   over `mlx5_6` and `mlx5_7`, but both vLLM and a minimal torch all-reduce hit
   `IBV_WC_RETRY_EXC_ERR`. This is below vLLM.
3. **Forced socket networking is stable but not representative.** It can disable
   NVLink/P2P and complete the vLLM DP2+EP workflow, but it is not a practical
   multi-node IB/DeepEP proxy.
4. **A research-only communication-delay hook is now implemented.** The hook is
   controlled by:

   ```bash
   VLLM_SELF_SPEC_EMULATE_A2A_DELAY_MS=<delay_ms>
   ```

   It injects delay after the `allgather_reducescatter` MoE EP all-gather and
   reduce-scatter path. Default is `0`, so default vLLM behavior is unchanged.

## Hook Validation

A minimal GPU6/7 DP2+EP run with:

```bash
VLLM_SELF_SPEC_EMULATE_A2A_DELAY_MS=4
```

completed successfully under default NCCL transport. The p50 latency for global
batch size 2 increased from about `1536 ms` to about `11525 ms`.

This large increase is expected because the hook is **per MoE collective call**,
not per decode step. It is useful for controlled stress testing, but the delay
value must be calibrated carefully.

## Current Scientific Conclusion

The research direction is still viable only under the qualified condition:

```text
low-batch multi-node EP decode
+ high exposed all-to-all after DeepEP/DBO
+ sufficiently high local-only draft acceptance
```

The current local experiments do not prove multi-node effectiveness. They show
that single-node/default communication has too little room, and that local
emulation should use either analytical delay injection or the new vLLM delay hook
rather than forced IB.

## Next Step

Use the new delay hook with default NCCL settings on GPUs 6 and 7 to produce a
calibrated delay sweep:

```text
VLLM_SELF_SPEC_EMULATE_A2A_DELAY_MS = 0, 0.025, 0.05, 0.1, 0.2
B = 2, 4, 6, 10
```

These values are intentionally small because the delay is applied per MoE
collective. The target is to estimate how much per-collective exposed delay
corresponds to useful per-step speedup.

This sweep has now been run and is summarized in
`results_calibrated_delay_sweep.md`. The follow-up count probe in
`results_a2a_count_calibration.md` shows the current hook fires about `1536`
times per worker per measured iteration in the DP2+EP path, so the next delay
sweep should use much smaller values:

```text
0, 0.0005, 0.001, 0.0025, 0.005 ms
```

This count instrumentation has been added behind:

```bash
VLLM_SELF_SPEC_LOG_A2A_COUNTS=1
```

It logs the `AgRsAll2AllManager` allgather/reducescatter call count on worker
shutdown.

Use `VLLM_SELF_SPEC_A2A_COUNT_ACTIVE_FILE=<path>` with the phase-local DP sweep
helper to count only the active benchmark window after engine initialization.

After that, move to Phase 03:

```text
measure local-only draft acceptance beta and local expert mass gamma
```

because the remaining unknown is whether local-only routing can reach the
acceptance thresholds identified by the Phase 01/02 envelopes.

Phase 03 has been initialized at:

```text
research/03_local_routing_acceptance
```
