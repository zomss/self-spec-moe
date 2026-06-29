#!/usr/bin/env python3
"""Overlapped (continuous) spec: hide the comm-free draft behind the verify's comm.

Sequential spec pays draft + verify. If the comm-free draft compute overlaps the verify's
all-to-all (comm), the draft can be HIDDEN when comm > compute. Resources run concurrently:
  verify: compute C_v + comm M_v=R*C_v   (R = comm/compute; rises with batch & inter-node)
  draft (comm-free): C_d  (World A full forward C_d=1; EAGLE small head C_d~0.1)
  overlapped cycle = max(C_v + C_d, M_v) ;  baseline (DBO, no spec) = max(C_v, M_v)
  speedup = (beta+1) * baseline / cycle   (k=1, the comm-bound optimum)
The draft-cost-vs-quality tradeoff FLIPS at R=2 (comm>2x compute, f>~0.67): below it the
draft cost dominates (cheap EAGLE wins); above it the draft hides for free and only beta
matters (World A's full-model draft has higher beta -> wins, training-free).
"""
def cycle(Cd, R): return max(1.0 + Cd, R)
def sp(beta, Cd, R): return (beta + 1) * max(1.0, R) / cycle(Cd, R)

if __name__ == "__main__":
    import json
    rows = []
    for R in [0.7, 1.2, 1.9, 2.0, 4.0, 9.0]:
        f = R / (1 + R)
        rows.append({"R": R, "f": round(f, 2),
                     "eagle_b80": round(sp(0.80, 0.10, R), 3),
                     "worldA_local_b85": round(sp(0.85, 1.00, R), 3),
                     "worldA_fp4_b92": round(sp(0.92, 1.00, R), 3)})
        print(rows[-1])
    json.dump({"rows": rows, "crossover_R": 2.0, "crossover_f": 0.67},
              open(__file__.rsplit("/", 1)[0] + "/data/overlap_model.json", "w"), indent=2)
