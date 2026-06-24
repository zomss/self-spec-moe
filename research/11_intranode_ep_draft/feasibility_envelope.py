#!/usr/bin/env python3
"""Combine measured NVLink all-to-all + compute floor + modeled IB into a
hierarchical-EP draft speedup envelope.

Draft step (intra-node EP): compute + intra-node NVLink all-to-all.
Verify step (full EP, multi-node): compute + inter-node IB all-to-all.
Both pay 2 collectives per MoE layer (dispatch + combine).
"""

import json

# measured NVLink per-collective latency at decode payloads (small msgs), us
NVLINK_US = 30.0  # from data/a2a_nvlink.json (small-message regime)

MODELS = {
    "Qwen3-30B-A3B": {"layers": 48, "compute_eager_ms": 94.9, "compute_opt_ms": 12.0},
    "GPT-OSS-20B": {"layers": 24, "compute_eager_ms": 39.4, "compute_opt_ms": 6.0},
}
IB_US = [50.0, 100.0, 200.0]          # modeled inter-node per-collective latency
BETAS = [0.5, 0.74, 0.9]               # acceptance (0.74 = 2-4 node + per-node cache)
NUM_SPEC = 3                            # draft tokens per cycle


def E_tokens(beta, n):
    return sum(beta**i for i in range(n + 1))


def speedup(compute_ms, layers, nvlink_us, ib_us, beta, n):
    coll = 2 * layers
    intra = coll * nvlink_us / 1000.0
    inter = coll * ib_us / 1000.0
    t_draft = compute_ms + intra
    t_verify = compute_ms + inter
    f_inter = 1 - t_draft / t_verify
    sp = E_tokens(beta, n) * t_verify / (n * t_draft + t_verify)
    return f_inter, sp, intra, inter


for name, m in MODELS.items():
    print(f"\n===== {name}  (layers={m['layers']}, "
          f"{2*m['layers']} collectives/step, NVLink {NVLINK_US}us/coll) =====")
    for compute_label, compute_ms in [
        ("optimized~%.0fms" % m["compute_opt_ms"], m["compute_opt_ms"]),
        ("HF-eager %.0fms" % m["compute_eager_ms"], m["compute_eager_ms"]),
    ]:
        for ib in IB_US:
            f_inter, _, intra, inter = speedup(compute_ms, m["layers"], NVLINK_US, ib, 0.74, NUM_SPEC)
            row = (f"  compute={compute_label:>16} | IB={ib:>5.0f}us/coll "
                   f"(intra {intra:.1f}ms / inter {inter:.1f}ms) | f_inter={f_inter:5.2f} | speedup@n=3: ")
            sps = []
            for b in BETAS:
                _, sp, _, _ = speedup(compute_ms, m["layers"], NVLINK_US, ib, b, NUM_SPEC)
                sps.append(f"b={b}:{sp:.2f}")
            print(row + "  ".join(sps))


if __name__ == "__main__":
    pass
