#!/usr/bin/env python3
"""Map v6: LCB selection under per-source uncertainty, execution-extended.

Per cell: admissible configs (residency-feasible realizations) are priced
by Monte-Carlo over (beta, R, phi, psi) with source-dependent sigmas;
winner = argmax of the 5th-percentile speedup (LCB); P(win) = argmax
frequency. OFF = exactly 1.0. New Phase-84 levers priced: vres16k (dense
composed add-on), ngram (mla; gamma=4 block semantics approximated as
geometric beta_eff -- flagged), topc4 (moe), mlpskip6 (dense).

phi/psi tiers (85 fit): dense_fixed (0.00, 0.00) s=(0.02,0.05);
moe_mitigated (0.03, 0.15) s=(0.05,0.10) [one-equation split, flagged];
ngram: no chain -> phi=0, psi (0.25, 0.10). All chains assumed at their
BEST measured implementation tier.
"""

import json
import random
import sys
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
R80 = PHASE.parent / "80_lever_composition"
for p in (R80, PHASE.parent / "76_lever_latency_sweep",
          PHASE.parent / "78_cost_model", PHASE.parent / "79_paper"):
    sys.path.insert(0, str(p / "scripts"))
import search_v4 as SV  # noqa: E402

random.seed(7)
N_MC = 600
CTX_OF = {2: 2048, 16: 16384, 32: 32768}

# (name, beta, beta_sigma, R_spec, R_sigma, tier, gcap, note)
# R_spec: None = measured single per cell; "combo" = per-cell combo table;
# ("mult", base_spec, factor) = derived edit; ("flr", ...) = realization table.
W4WIN_R = {(8, 16): (0.435, 0.05), (32, 16): (0.311, 0.05)}  # measured
WINFAC = {2: (1.00, 0.05), 16: (0.62, 0.15), 32: (0.40, 0.25)}  # w4->combo
FLR_R2K = {4: (0.78, 0.15), 8: (1.40, 0.25), 32: (1.20, 0.22)}  # E2c implied
FLR_CTXFAC = {2: 1.0, 16: 1.0, 32: 1.0}  # MEASURED (v6q): full-attention draft -> R ctx-invariant (b4/16k parity)
CONFIGS = {
    "dense": [
        ("q_int4", 0.9427, 0.010, None, 0.04, "fixed", 8, "GPTQ; measR/cell"),
        ("q_int4+win512", 0.917 * (0.9427 / 0.924), 0.012, "combo", None, "fixed", 8, "measR combo / w4xwinfac"),
        ("q_int4+win512+vres32kC4", 0.917 * (0.9427 / 0.924) * 0.88, 0.04, ("vres", "combo"), None, "fixed", 8, "HONEST re-gate: C4 keep, live coverage 0.90 -- loses at deep gamma"),
        ("mlpskip6+q_int4", 0.9427 * 0.8464, 0.025, ("mult", None, 0.85, 0.12), None, "fixed", 8, "84 sublayer x quant"),
        ("win512", 0.978, 0.010, None, 0.04, "fixed", 8, "measR/cell"),
    ],
    "moe": [
        ("win512", 0.971, 0.010, None, 0.04, "moe_mit", 8, "measR/cell"),
        ("win128", 0.969, 0.010, None, 0.04, "moe_mit", 8, "measR/cell"),
        ("flr50+q_fp8", 0.9531 * 0.993, 0.012, "flr", None, "moe_mit", 8, "83 realization R per cell (batch-dep)"),
        ("topc4+flr50", 0.8915 * 0.9531, 0.02, ("flrmult", 0.9, 0.2), None, "moe_mit", 8, "84 topc x freq, modelR"),
        ("q_fp8", 0.993, 0.010, None, 0.04, "moe_mit", 8, "measR/cell"),
    ],
    "mla": [
        ("ngram", 0.835, 0.030, ("const", 0.0, 0.0), None, "ngram", 4, "e2e MEASURED 0.77-0.78x: psi=2.4 (CPU lookup) kills it on this stack"),
        ("q_fp8", 0.995, 0.010, None, 0.04, "mla_eager", 8, "measR/cell; chain eager"),
        ("win512", 0.922, 0.015, None, 0.04, "mla_eager", 8, "measR/cell"),
    ],
}


def resolve_R(group, name, spec, b, ck, Rs):
    """Return (R, sigma_frac) for a config at a cell, or None."""
    if spec is None:
        arm = SINGLE_ARM.get((group, name))
        R = Rs[group].get((arm, (b, ck))) if arm else None
        return (R, 0.04) if R is not None else None
    if spec == "combo":
        if (b, ck) in W4WIN_R:
            return W4WIN_R[(b, ck)]
        w4 = Rs["dense"].get(("d_w4marlin", (b, ck)))
        if w4 is None:
            return None
        f, fs = WINFAC[ck]
        return (w4 * f, 0.05 + fs)
    if spec == "flr":
        base, sig = FLR_R2K[b]
        return (base * FLR_CTXFAC[ck], sig)
    if isinstance(spec, tuple) and spec[0] == "vres":
        base = resolve_R(group, "q_int4+win512", "combo", b, ck, Rs)
        return (base[0] * 0.84, base[1] + 0.04) if base else None  # 32k keep: smaller lm_head cut
    if isinstance(spec, tuple) and spec[0] == "mult":
        w4 = Rs["dense"].get(("d_w4marlin", (b, ck)))
        return (w4 * spec[2], spec[3]) if w4 is not None else None
    if isinstance(spec, tuple) and spec[0] == "flrmult":
        base, sig = FLR_R2K[b]
        return (base * FLR_CTXFAC[ck] * spec[1], sig + spec[2])
    if isinstance(spec, tuple) and spec[0] == "const":
        return (spec[1], spec[2])
    return None
SINGLE_ARM = {("dense", "q_int4"): "d_w4marlin", ("dense", "win512"): "d_win",
              ("moe", "win512"): "m_win", ("moe", "win128"): "m_win128",
              ("moe", "q_fp8"): "m_fp8marlin",
              ("mla", "q_fp8"): "ds_fp8marlin", ("mla", "win512"): "ds_win"}
TIERS = {"fixed": (0.00, 0.02, 0.00, 0.05),
         "moe_mit": (0.03, 0.05, 0.15, 0.10),
         "mla_eager": (0.90, 0.20, 0.20, 0.10),
         "ngram": (0.00, 0.00, 2.40, 0.30)}  # MEASURED (v6q): CPU O(ctx) lookup/req/cycle + verify width
# residency-infeasible (85 residency.py): dense b32/32k
INFEASIBLE = {("dense", (32, 32))}
GAMMAS = range(1, 9)


def tau(beta, g):
    return sum(beta ** k for k in range(1, g + 1)) + beta ** g * 0 + 1 \
        if False else (1 - beta ** (g + 1)) / (1 - beta) if beta < 1 else g + 1


def draw(mu, sigma):
    return max(0.0, random.gauss(mu, sigma * mu if mu > 0 else sigma))


def price_cell(group, b, ck, Rs):
    cands = []
    for name, beta, bs, spec, _, tier, gcap, note in CONFIGS[group]:
        got = resolve_R(group, name, spec, b, ck, Rs)
        if got is None:
            continue
        R, rs = got
        cands.append((name, beta, bs, R, rs, tier, gcap, note))
    # Monte-Carlo
    stats = {n: [] for n, *_ in cands}
    wins = {n: 0 for n, *_ in cands}
    wins["OFF"] = 0
    for _ in range(N_MC):
        best_v, best_n = 1.0, "OFF"
        for name, beta, bs, R, rs, tier, gcap, note in cands:
            bsamp = min(0.999, max(0.01, random.gauss(beta, bs)))
            Rsamp = max(0.0, random.gauss(R, rs * max(R, 0.05)))
            p0, ps0, q0, qs0 = TIERS[tier]
            phi = max(0.0, random.gauss(p0, ps0))
            psi = max(0.0, random.gauss(q0, qs0))
            v = max(tau(bsamp, g) / (g * (Rsamp + phi) + 1 + psi)
                    for g in range(1, gcap + 1))
            stats[name].append(v)
            if v > best_v:
                best_v, best_n = v, name
        wins[best_n] += 1
    out = []
    for name, beta, bs, R, rs, tier, gcap, note in cands:
        vs = sorted(stats[name])
        out.append((vs[int(0.05 * N_MC)], vs[N_MC // 2], wins[name] / N_MC,
                    name, note))
    out.sort(reverse=True)
    return out, wins["OFF"] / N_MC


def main() -> int:
    _, _, Rs, _, _ = SV.load_all()
    lines = ["# Strategy map v6 — LCB selection, execution-extended, "
             "feasibility-filtered", "",
             "winner = argmax LCB05; speedup = tau/(gamma(R+phi)+1+psi) at "
             "the chain tier's fitted constants; P(win) from 600 MC draws.",
             "ngram priced as geometric beta_eff at gamma<=8 (block-proposal "
             "semantics approximated; e2e unvalidated -- wide sigma).", ""]
    groups = {"dense": [1, 8, 32], "moe": [4, 8, 32], "mla": [4, 8, 32]}
    for group, bs in groups.items():
        lines.append(f"\n## {group}\n")
        lines.append("| cell | winner (LCB05 / median / P(win)) | runner-up | note |")
        lines.append("|---|---|---|---|")
        for b in bs:
            for ck in (2, 16, 32):
                if (group, (b, ck)) in INFEASIBLE:
                    lines.append(f"| b{b}/{ck}k | INFEASIBLE (residency) | | |")
                    continue
                ranked, p_off = price_cell(group, b, ck, Rs)
                if not ranked or ranked[0][0] <= 1.0:
                    top = ranked[0] if ranked else None
                    lines.append(
                        f"| b{b}/{ck}k | **OFF** (best LCB "
                        f"{top[0]:.2f} {top[3]} | P(OFF-ish)={p_off:.2f} | "
                        f"{top[4] if top else ''} |")
                    continue
                w = ranked[0]
                ru = ranked[1] if len(ranked) > 1 else None
                lines.append(
                    f"| b{b}/{ck}k | **{w[3]}** {w[0]:.2f}/{w[1]:.2f}/"
                    f"P{w[2]:.2f} | "
                    f"{ru[3] + f' {ru[0]:.2f}' if ru else '—'} | {w[4]} |")
    text = "\n".join(lines)
    (PHASE / "data/strategy_map_v6.md").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
