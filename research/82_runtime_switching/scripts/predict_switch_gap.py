#!/usr/bin/env python3
"""Where does switching pay? Map-predicted policy-vs-best-static gap per
workload family (no GPU: measured cells only).

A workload = weights over regimes; each regime r has measured per-arm
speedup S_a(r) (vs that regime's AR). Static value = time-weighted
harmonic aggregate; policy value = per-regime max with a REGRET haircut
(measured tracking regret 2%). Gap = policy - best static, maximized
over the weight simplex (random search) -- i.e. the most switching-
favorable workload composition inside the family.
"""
import random

random.seed(7)
REGRET = 0.98  # measured runtime tracking: within 1-3% of per-regime max

# ---- measured regimes: name -> {arm: speedup vs regime AR} -------------
# 8B, wholechain stack + FIXED skip condition (traces, 2026-07-19)
Q38B = {
    "b1_2k_math":   {"off": 1.0, "k4": 1.155, "k6": 1.156},
    "b1_14k_rag":   {"off": 1.0, "k4": 1.233, "k6": 1.194},
    "b1_16k_docs":  {"off": 1.0, "k4": 0.809, "k6": 0.675},
    "b8_6k_docs":   {"off": 1.0, "k4": 0.895, "k6": 0.793},
    "b8_14k_rag":   {"off": 1.0, "k4": 1.403, "k6": 1.397},
    "b8_16k_docs":  {"off": 1.0, "k4": 1.248, "k6": 1.112},
    "b16_14k_rag":  {"off": 1.0, "k4": 1.519, "k6": 1.526},
    "b16_16k_docs": {"off": 1.0, "k4": 1.183, "k6": 1.079},
    "b32_2k_math":  {"off": 1.0, "k4": 0.862, "k6": 0.742},
}
# 32B column (h2h decode-only, fixed chain -- cross-stack transfer FLAGGED)
Q332B = {
    "b1_16k_math":  {"off": 1.0, "k4": 1.54, "k5": 1.63},
    "b1_16k_prose": {"off": 1.0, "k4": 1.41, "k5": 1.35},
    "b8_16k_prose": {"off": 1.0, "k4": 1.22, "k5": 1.28},
    "b8_2k_burst":  {"off": 1.0, "k4": 0.87, "k5": 0.82},  # transfer est.
}
# Fleet (per-model best lever vs one-global-config): measured e2e headline
# cells -- dense w4win, MoE win-K3, MLA (self-spec loses).
FLEET = {
    "dense_16k_b8": {"off": 1.0, "w4win": 1.81, "moewin": 1.30, "selfd": 1.20},
    "moe_16k_b8":   {"off": 1.0, "w4win": 0.63, "moewin": 1.15, "selfd": 0.80},
    "mla_16k_b32":  {"off": 1.0, "w4win": 0.56, "moewin": 0.56, "selfd": 0.56},
}


def value_static(regimes, weights, arm):
    # time-weighted: equal AR volume per regime scaled by weight; time for
    # arm a in regime r ~ w_r / S_a(r); value = 1 / sum(w/S) (harmonic).
    t = sum(w / regimes[r][arm] for r, w in weights.items())
    return 1.0 / t


def value_policy(regimes, weights):
    t = sum(w / (max(regimes[r].values()) * REGRET)
            for r, w in weights.items())
    return 1.0 / t


def max_gap(regimes, n=20000):
    arms = list(next(iter(regimes.values())).keys())
    names = list(regimes)
    best = (0.0, None)
    for _ in range(n):
        raw = [random.random() for _ in names]
        s = sum(raw)
        w = {r: x / s for r, x in zip(names, raw)}
        pol = value_policy(regimes, w)
        stat = max(value_static(regimes, w, a) for a in arms)
        gap = pol / stat - 1
        if gap > best[0]:
            best = (gap, w)
    gap, w = best
    top = sorted(w.items(), key=lambda kv: -kv[1])[:4]
    return gap, top


def rl_drift():
    """RL rollout: accept drifts over training (on-policy f .92 -> .70),
    batch drains within each step (b16 gen -> b4 tail). S from the
    compiled (K,R) cells at b16/8k and b4-ish (b1/14k proxy)."""
    def S(f, k, r):
        return (1 + f * k) / (k * r + 1)
    phases = []
    for i in range(10):
        f = 0.92 - 0.022 * i * 10 / 9  # -> 0.70 late in training
        phases.append({
            "off": 1.0,
            "k4": S(f, 4, 0.70),   # b16/8k R
            "k6": S(f, 6, 0.666),
        })
    w = {i: 1 / len(phases) for i in range(len(phases))}
    regs = {i: p for i, p in enumerate(phases)}
    pol = value_policy(regs, w)
    stat = max(value_static(regs, w, a) for a in ["off", "k4", "k6"])
    return pol / stat - 1, [round(max(p.values()), 2) for p in phases]


for name, regs in [("Qwen3-8B intra-column", Q38B),
                   ("Qwen3-32B column (transfer)", Q332B),
                   ("fleet (dense+MoE+MLA, one global lever)", FLEET)]:
    gap, top = max_gap(regs)
    print(f"{name}: max predicted gap = {gap*100:+.1f}%")
    print(f"   at weights {[(r, round(w,2)) for r, w in top]}")
rl_gap, curve = rl_drift()
print(f"RL rollout drift model: gap = {rl_gap*100:+.1f}% "
      f"(per-phase best S: {curve})")
