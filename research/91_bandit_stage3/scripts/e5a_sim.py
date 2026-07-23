#!/usr/bin/env python3
"""Phase 91 E5a: calibrated bandit simulator.

World: decode steps in a cell (b16 drift trace shape). Per step the
engine holds arm a in {OFF, K2, K3, K4}; drafted positions accept
i.i.d. Bernoulli(f_t) (f_t = phase-scheduled drift, calibrated to the
89-E3b measured accept curve); step time modeled from the compiled
(K, R): T_step(a) proportional to K*R_a + 1 (+ arming cost 2% of a
step when switching OFF->on; kmax padding tax applied to ALL arms
from the boot kmax). Goodput per step = accepted+1 tokens / T_step.

Policies:
  argmax-EMA   the deployed scheduler: EMA(f), argmax S, asymmetric
               hysteresis (-0.02 arming penalty), probe burst 8/128
  thompson     discounted Beta posterior per arm, sample f, argmax S
               with the same switch-cost awareness
  swucb        sliding-window UCB on goodput per arm
Variants: with/without switch-cost awareness (SC).

Drift: f multiplies a drift factor per phase [1.0, .97, .92, .48, .37]
(matching the measured accept curve 3.3->1.2 shape); a refresh event
resets drift to 1.0 when the policy's chosen estimate crosses the
gate (client-side, 10s poll modeled as 100-step delay).

Regret vs the clairvoyant per-step best arm. 20 seeds x 5000 steps.
"""
import json
import math
import random
from pathlib import Path

CELLS = {  # (K -> R) at b16/8k from the compiled Hum table
    "R": {2: 0.469, 3: 0.512, 4: 0.490},
    "f0": {2: 0.913, 3: 0.881, 4: 0.854},
}
DRIFT = [1.0, 0.97, 0.92, 0.48, 0.37]
PHASE_STEPS = 1000
ARM_COST = 0.02      # measured arming tax (fraction of a step)
POLL_DELAY = 100     # client refresh detection, steps
GATE = 0.65          # accept-fraction gate for refresh


def step_time(K, R, kmax):
    pad = 1 + 0.012 * max(0, kmax - K)   # measured kmax padding tax
    return (K * R + 1) * pad if K else 1.0 * pad


def goodput(K, f, R, kmax, rng):
    acc = 0
    for _ in range(K):
        if rng.random() < f:
            acc += 1
        else:
            break
    return (acc + 1) / step_time(K, CELLS["R"].get(K, 0), kmax), acc


class ArgmaxEMA:
    def __init__(self, kmax, sc=True):
        self.f = {k: 0.9 for k in (2, 3, 4) if k <= kmax}
        self.cur, self.kmax, self.sc = 0, kmax, sc
        self.n = 0

    def pick(self, rng):
        self.n += 1
        if self.cur == 0 and self.n % 128 < 8:
            return max(self.f)          # probe burst
        best_k, best_s = 0, 1.0
        for k, f in self.f.items():
            s = (1 + f * k) / (k * CELLS["R"][k] + 1)
            if self.sc and k != self.cur:
                s -= 0.02
            if s > best_s:
                best_k, best_s = k, s
        return best_k

    def update(self, k, acc):
        if k:
            w = k / (k + 24)
            self.f[k] = (1 - w) * self.f[k] + w * (acc / k)
            self.cur = k
        else:
            self.cur = 0


class Thompson:
    def __init__(self, kmax, sc=True, half=24.0):
        self.ab = {k: [9.0, 1.0] for k in (2, 3, 4) if k <= kmax}
        self.cur, self.sc = 0, sc
        self.decay = 0.5 ** (1.0 / half)

    def pick(self, rng):
        best_k, best_s = 0, 1.0
        for k, (a, b) in self.ab.items():
            f = rng.betavariate(a, b)
            s = (1 + f * k) / (k * CELLS["R"][k] + 1)
            if self.sc and k != self.cur and self.cur == 0:
                s -= ARM_COST
            if s > best_s:
                best_k, best_s = k, s
        return best_k

    def update(self, k, acc):
        for kk in self.ab:               # discount ALL posteriors
            self.ab[kk][0] = 1 + (self.ab[kk][0] - 1) * self.decay
            self.ab[kk][1] = 1 + (self.ab[kk][1] - 1) * self.decay
        if k:
            self.ab[k][0] += acc
            self.ab[k][1] += (k - acc) if acc < k else 0
        self.cur = k


class SWUCB:
    def __init__(self, kmax, sc=True, win=200):
        self.hist = {k: [] for k in (0, 2, 3, 4) if k == 0 or k <= kmax}
        self.cur, self.sc, self.win = 0, sc, win
        self.t = 0

    def pick(self, rng):
        self.t += 1
        for k, h in self.hist.items():
            if len(h) < 3:
                return k
        best_k, best_v = 0, -1
        for k, h in self.hist.items():
            m = sum(h) / len(h)
            u = m + math.sqrt(2 * math.log(min(self.t, self.win))
                              / len(h))
            if self.sc and k != self.cur and k and self.cur == 0:
                u -= ARM_COST
            if u > best_v:
                best_k, best_v = k, u
        return best_k

    def update_gp(self, k, gp):
        h = self.hist[k]
        h.append(gp)
        if len(h) > self.win:
            h.pop(0)
        self.cur = k


def run(policy_cls, kmax, sc, seed, refresh=True, **kw):
    rng = random.Random(seed)
    pol = policy_cls(kmax, sc, **kw)
    total_gp = 0.0
    oracle_gp = 0.0
    drift_idx = 0
    steps_in_phase = 0
    detect_timer = None
    drift = DRIFT[0]
    for t in range(PHASE_STEPS * len(DRIFT)):
        if steps_in_phase >= PHASE_STEPS:
            drift_idx += 1
            steps_in_phase = 0
            drift = DRIFT[min(drift_idx, len(DRIFT) - 1)]
        steps_in_phase += 1
        k = pol.pick(rng)
        f = {kk: CELLS["f0"][kk] * drift for kk in CELLS["f0"]}
        gp, acc = (goodput(k, f[k], CELLS["R"][k], kmax, rng)
                   if k else (1.0 / step_time(0, 0, kmax), 0))
        total_gp += gp
        # oracle: best arm at true f
        og = max([1.0 / step_time(0, 0, kmax)]
                 + [(1 + f[kk] * kk)
                    / step_time(kk, CELLS["R"][kk], kmax)
                    for kk in f if kk <= kmax])
        oracle_gp += og
        if isinstance(pol, SWUCB):
            pol.update_gp(k, gp)
        else:
            pol.update(k, acc)
        # client-side refresh: detect drifted accept, reset drift
        if refresh and drift < GATE and detect_timer is None:
            detect_timer = t + POLL_DELAY
        if detect_timer is not None and t >= detect_timer:
            drift = 1.0
            detect_timer = None
    return total_gp / (PHASE_STEPS * len(DRIFT)), \
        oracle_gp / (PHASE_STEPS * len(DRIFT))


def main():
    out = {}
    for name, cls, kw in [("argmaxEMA", ArgmaxEMA, {}),
                          ("thompson", Thompson, {}),
                          ("swucb", SWUCB, {})]:
        for kmax in (3, 4):
            for sc in (True, False):
                gps, ors = [], []
                for seed in range(20):
                    g, o = run(cls, kmax, sc, seed)
                    gps.append(g)
                    ors.append(o)
                key = f"{name}_k{kmax}_{'sc' if sc else 'nosc'}"
                mean_gp = sum(gps) / len(gps)
                mean_or = sum(ors) / len(ors)
                out[key] = {"goodput": round(mean_gp, 4),
                            "oracle": round(mean_or, 4),
                            "regret_pct":
                            round(100 * (1 - mean_gp / mean_or), 2)}
                print(f"[e5a] {key}: goodput={mean_gp:.4f} "
                      f"oracle={mean_or:.4f} "
                      f"regret={out[key]['regret_pct']:.2f}%",
                      flush=True)
    Path("research/91_bandit_stage3/data/e5a_sim.json").write_text(
        json.dumps(out, indent=1))
    print("[e5a] saved", flush=True)


if __name__ == "__main__":
    main()
