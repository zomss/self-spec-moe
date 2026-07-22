#!/usr/bin/env python3
"""Phase 90 E1f: LEARNED config->beta predictor on the committed record.

Features per skip-set (NO per-config forward needed):
  structural: size, adjacent-pair count, max run, position stats,
              early/mid/late thirds
  score aggregates: sum/mean/max over the set of per-layer scores
              (angular importance; +taylor/micro/on-policy at 8B)
Models: ridge + RBF kernel ridge (numpy; no sklearn dep).
Eval (leakage-strict): leave-one-ROUND-out — the model never sees any
arm of the round it ranks. Baselines: additive singles-beta product;
raw sum-of-score.  Also 8B->32B transfer on shared features.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

D = Path("research/90_hier_search/data")
LOGS = Path("research/86_scale_headtohead/logs")


def parse_pairs(profile_log, greedy_log):
    loo = defaultdict(list)
    for m in re.finditer(r"\[B\] ig_([0-9]+) p\d+: beta=(0\.\d+)",
                         open(profile_log, errors="ignore").read()):
        loo[int(m.group(1))].append(float(m.group(2)))
    loo = {k: float(np.mean(v)) for k, v in loo.items()}
    arms = defaultdict(list)
    for m in re.finditer(r"\[B\] igs_([0-9-]+) p\d+: beta=(0\.\d+)",
                         open(greedy_log, errors="ignore").read()):
        arms[tuple(sorted(int(x) for x in m.group(1).split("-")))].append(
            float(m.group(2)))
    arms = {k: float(np.mean(v)) for k, v in arms.items()}
    return loo, arms


def load_scores_8b():
    s = {}
    ang = {}
    for line in open(D / "q38b_c4_importance.csv").read().splitlines()[1:]:
        p = line.split(",")
        ang[int(p[2])] = float(p[3])
    s["angular"] = ang
    tay, mic = {}, {}
    for line in open(D / "q38b_acceptproxy.csv").read().splitlines()[1:]:
        p = line.split(",")
        tay[int(p[1])] = float(p[2])
        mic[int(p[1])] = float(p[3])
    s["taylor"] = tay
    s["micro"] = mic
    onp = {int(k): 1 - v for k, v in json.load(
        open(D / "e1d_onpolicy.json")).items()}
    s["onpolicy"] = onp
    return s


def load_scores_32b():
    ang = {}
    for line in open(D / "q332b_c4_importance.csv").read().splitlines()[1:]:
        p = line.split(",")
        ang[int(p[2])] = float(p[3])
    return {"angular": ang}


def featurize(sets, L, scores, shared_only=False):
    names = ["size", "adjpairs", "maxrun", "meanpos", "minpos",
             "maxpos", "stdpos", "early", "mid", "late"]
    use = ["angular"] if shared_only else sorted(scores)
    for sc in use:
        names += [f"{sc}_sum", f"{sc}_mean", f"{sc}_max"]
    X = []
    for S in sets:
        S = sorted(S)
        adj = sum(1 for a, b in zip(S, S[1:]) if b == a + 1)
        runs, cur = [], 1
        for a, b in zip(S, S[1:]):
            cur = cur + 1 if b == a + 1 else 1
            runs.append(cur)
        maxrun = max(runs) if runs else 1
        pos = np.array(S) / L
        row = [len(S), adj, maxrun, pos.mean(), pos.min(), pos.max(),
               pos.std(), sum(p < 1/3 for p in pos),
               sum(1/3 <= p < 2/3 for p in pos),
               sum(p >= 2/3 for p in pos)]
        for sc in use:
            vals = [scores[sc].get(l, np.median(list(scores[sc].values())))
                    for l in S]
            row += [float(np.sum(vals)), float(np.mean(vals)),
                    float(np.max(vals))]
        X.append(row)
    return np.array(X, dtype=float), names


def fit_predict(Xtr, ytr, Xte, kind="krr"):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    if kind == "ridge":
        A = Xtr.T @ Xtr + 1.0 * np.eye(Xtr.shape[1])
        w = np.linalg.solve(A, Xtr.T @ (ytr - ytr.mean()))
        return Xte @ w + ytr.mean()
    # RBF kernel ridge
    g = 1.0 / Xtr.shape[1]
    def K(A, B):
        d = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-g * d)
    a = np.linalg.solve(K(Xtr, Xtr) + 0.1 * np.eye(len(Xtr)),
                        ytr - ytr.mean())
    return K(Xte, Xtr) @ a + ytr.mean()


def eval_model(tag, loo, arms, scores, L, shared_only=False):
    sets = [(l,) for l in sorted(loo)] + sorted(arms)
    y = np.array([loo[s[0]] for s in sets if len(s) == 1]
                 + [arms[s] for s in sets if len(s) > 1])
    X, names = featurize(sets, L, scores, shared_only)
    sizes = np.array([len(s) for s in sets])

    by_size = defaultdict(list)
    for i, s in enumerate(sets):
        if len(s) > 1:
            by_size[len(s)].append(i)
    print(f"--- {tag} (n={len(sets)}, feats={len(names)}) ---")
    for kind in ["ridge", "krr"]:
        rhos, prod_rhos = [], []
        for size, idx in sorted(by_size.items()):
            if len(idx) < 4:
                continue
            te = np.array(idx)
            tr = np.array([i for i in range(len(sets))
                           if i not in set(idx)])
            pred = fit_predict(X[tr], y[tr], X[te], kind)
            r, _ = spearmanr(pred, y[te])
            rhos.append(r)
            # additive baseline: product of single-layer betas
            prod = [np.prod([loo.get(l, np.median(list(loo.values())))
                             for l in sets[i]]) for i in idx]
            pr, _ = spearmanr(prod, y[te])
            prod_rhos.append(pr)
        print(f"[{tag} {kind}] leave-round-out within-size rhos: "
              + ", ".join(f"{r:.2f}" for r in rhos)
              + f" | mean={np.mean(rhos):.3f}")
        if kind == "ridge":
            print(f"[{tag} baseline product-of-singles] rhos: "
                  + ", ".join(f"{r:.2f}" for r in prod_rhos)
                  + f" | mean={np.mean(prod_rhos):.3f}")
    return X, y, sets, names


def main():
    loo8, arms8 = parse_pairs(LOGS / "beta_profile.log",
                              LOGS / "beta_greedy.log")
    s8 = load_scores_8b()
    eval_model("8B full-feats", loo8, arms8, s8, 36)

    loo32, arms32 = parse_pairs(LOGS / "beta32_profile.log",
                                LOGS / "beta32_greedy.log")
    s32 = load_scores_32b()
    eval_model("32B shared-feats", loo32, arms32, s32, 64,
               shared_only=True)

    # ---- 8B -> 32B transfer (shared features) ----
    X8, y8, sets8, _ = (lambda t: t)(None) or (None, None, None, None)
    X8, names = featurize([(l,) for l in sorted(loo8)] + sorted(arms8),
                          36, s8, shared_only=True)
    y8 = np.array([loo8[l] for l in sorted(loo8)]
                  + [arms8[s] for s in sorted(arms8)])
    sets32 = [(l,) for l in sorted(loo32)] + sorted(arms32)
    X32, _ = featurize(sets32, 64, s32, shared_only=True)
    y32 = np.array([loo32[l] for l in sorted(loo32)]
                   + [arms32[s] for s in sorted(arms32)])
    pred = fit_predict(X8, y8, X32, "krr")
    by_size = defaultdict(list)
    for i, s in enumerate(sets32):
        if len(s) > 1:
            by_size[len(s)].append(i)
    rhos = []
    for size, idx in sorted(by_size.items()):
        if len(idx) < 4:
            continue
        r, _ = spearmanr(pred[np.array(idx)], y32[np.array(idx)])
        rhos.append(r)
    print(f"[8B->32B transfer krr] within-size rhos: "
          + ", ".join(f"{r:.2f}" for r in rhos)
          + f" | mean={np.mean(rhos):.3f}")


if __name__ == "__main__":
    main()
