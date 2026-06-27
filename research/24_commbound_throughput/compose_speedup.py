#!/usr/bin/env python3
"""Compose lossless spec-decode speedup from measured comm-bound step times.

Stage A. Inputs are REAL measured decode-step times (no injection):
  S_nvlink (P2P on)  ~= compute-only           -> a comm-free local-routing draft
  S_socket (forced sockets) = compute + comm    -> the comm-bound full-EP verify
The PCIe-exact point is not measurable here (NCCL_P2P_DISABLE hangs this box), so we
bracket: socket is a pessimistic upper bound on comm; a real PCIe-P2P link is faster.
We report the socket-measured speedup and a PCIe estimate (socket comm deflated by a
factor) -- the true point lies between NVLink (no win) and socket.

speedup = T(k,beta) * S_verify / (k * S_draft + S_verify),  baseline = 1 tok / S_verify
T(k,beta) = 1 + beta*(1-beta**k)/(1-beta)   (expected accepted + 1 verify token)
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
DEFLATE = [1.0, 2.0, 4.0]   # socket comm -> PCIe: assume PCIe is 1x..4x faster
BETAS = {"local_partial(0.82)": 0.82, "local_full / actq(0.92)": 0.92}
KS = [1, 2, 4, 6, 8]


def load(tag):
    d = json.loads((HERE / "data" / f"qwen3_{tag}.json").read_text())
    return {r["batch_global"]: r["step_ms"] for r in d["rows"]}


def tokens_per_cycle(k, b):
    return 1.0 + b * (1 - b ** k) / (1 - b)


def best_speedup(s_draft, s_verify, beta):
    best = (0.0, 0)
    for k in KS:
        sp = tokens_per_cycle(k, beta) * s_verify / (k * s_draft + s_verify)
        if sp > best[0]:
            best = (sp, k)
    return best


def main():
    nvl, soc = load("p2p_on"), load("socket")
    shared = sorted(set(nvl) & set(soc))
    print("=== regime: comm fraction f = (S_socket - S_nvlink)/S_socket ===")
    print(f"{'global B':>8} {'S_nvlink':>9} {'S_socket':>9} {'f':>6}")
    for B in shared:
        f = (soc[B] - nvl[B]) / soc[B]
        print(f"{B:>8} {nvl[B]:>9.2f} {soc[B]:>9.2f} {f:>6.3f}")

    print("\n=== composed lossless speedup (local-routing draft, comm eliminated) ===")
    print("S_draft = S_nvlink (compute-only); S_verify = comm-bound full-EP step")
    for B in shared:
        s_draft = nvl[B]
        comm = soc[B] - nvl[B]
        print(f"\n-- global batch {B} (S_nvlink={nvl[B]:.1f}ms, socket comm={comm:.1f}ms) --")
        for label, beta in BETAS.items():
            cells = []
            for defl in DEFLATE:
                s_verify = nvl[B] + comm / defl
                sp, k = best_speedup(s_draft, s_verify, beta)
                tag = "socket" if defl == 1.0 else f"pcie/{int(defl)}x"
                cells.append(f"{tag}={sp:.2f}x(k{k})")
            print(f"  beta {label:<24}: " + "  ".join(cells))

    print("\nNote: 'socket' = measured (pessimistic comm upper bound); 'pcie/Nx' = "
          "socket comm deflated Nx as a PCIe-P2P estimate. Real PCIe point is between "
          "NVLink (1x, no win) and socket.")

    # --- Stage B2a: use the MEASURED comm-free draft step (skip-A2A) ---
    skip_path = HERE / "data" / "qwen3_socket_skip.json"
    if skip_path.exists():
        skip = {r["batch_global"]: r["step_ms"]
                for r in json.loads(skip_path.read_text())["rows"]}
        print("\n=== B2a: speedup with MEASURED comm-free draft (socket-skip) ===")
        print("S_draft = measured skip-A2A step on the comm-bound engine; "
              "S_verify = measured socket full-EP step")
        for B in sorted(set(skip) & set(soc)):
            sd, sv = skip[B], soc[B]
            print(f"\n-- global batch {B} (S_draft_skip={sd:.1f}ms = "
                  f"{sd / nvl[B]:.2f}x NVLink; S_verify={sv:.1f}ms) --")
            for label, beta in BETAS.items():
                sp, k = best_speedup(sd, sv, beta)
                print(f"  beta {label:<24}: {sp:.2f}x (k={k})")


if __name__ == "__main__":
    raise SystemExit(main())
