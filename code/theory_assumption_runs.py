"""
Synthetic controller experiments run under the assumptions of Theorems 2 and 3.

Thompson sampling (Theorem 3): disjoint linear model with K actions, contexts
drawn uniformly from the unit sphere in R^m (so ||z_t|| = 1), parameters
theta*_a ~ N(0, lambda^{-1} I_m) with lambda = 1, Gaussian reward noise with
sigma = 1, and the controller of Algorithm 2 with prior precision lambda,
noise sigma = 1 and identity projection, so that it samples from the exact
posterior. Regret uses the expected rewards (Eq. 9).

Policy gradient (Theorem 2): linear-softmax policy pi_theta(a|x) proportional
to exp(<theta_a, x>) over unit-sphere contexts, so the score function is
bounded and J has a Lipschitz gradient. Rewards are Bernoulli with mean
mu(x, a) = 0.5 + 0.5 tanh(<w_a, x>), actions are sampled on-policy (no
epsilon-greedy), updates are unclipped, the baseline is
b_t = 0.9 b_{t-1} + 0.1 r_t, and eta_t = eta_0 / (t + 1) with t = 1 first.
||grad J(theta_t)||^2 is computed with the exact expectation over actions and
a Monte Carlo average over 20,000 fixed evaluation contexts.

Writes revision_results/theory_v2_ts_regret.csv, theory_v2_ts_curves.csv,
theory_v2_pg.csv and theory_v2_pg_variance.csv.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from hmag_core import LinearThompsonSamplingController

OUT = Path("./revision_results")
K = 6


def unit_sphere(rng, n, m):
    x = rng.normal(size=(n, m))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def run_ts(seeds=range(20), T=4000, dims=(4, 8, 16, 32), lam=1.0, sigma=1.0):
    reg_rows, curve_rows = [], []
    for m in dims:
        for seed in seeds:
            rng = np.random.default_rng(10_000 + 97 * m + seed)
            np.random.seed(seed)
            theta = rng.normal(0.0, 1.0 / math.sqrt(lam), size=(K, m))
            ctl = LinearThompsonSamplingController(context_dim=m, num_actions=K, sigma=sigma, lam=lam, feature_dim=m)
            ctl.projection = np.eye(m)
            cum = 0.0
            for t in range(1, T + 1):
                z = unit_sphere(rng, 1, m)[0]
                means = theta @ z
                a = ctl.select_action(z)
                ctl.update(z, a, float(means[a] + rng.normal(0.0, sigma)))
                cum += float(means.max() - means[a])
                if t % 50 == 0:
                    curve_rows.append({"m": m, "seed": seed, "t": t, "cum_regret": cum})
            reg_rows.append({"m": m, "seed": seed, "T": T, "regret_T": cum})
    curves = pd.DataFrame(curve_rows)
    reg = pd.DataFrame(reg_rows)
    mean_curve = curves.groupby(["m", "t"]).cum_regret.mean().reset_index()
    fits = []
    for m, g in mean_curve.groupby("m"):
        g = g[g.t >= T // 10]
        fits.append({"m": m, "growth_exponent": np.polyfit(np.log(g.t), np.log(g.cum_regret), 1)[0]})
    reg.to_csv(OUT / "theory_v2_ts_regret.csv", index=False)
    curves.to_csv(OUT / "theory_v2_ts_curves.csv", index=False)
    summary = reg.groupby("m").regret_T.agg(["mean", "std", "count"]).reset_index().merge(pd.DataFrame(fits), on="m")
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])
    summary["ratio"] = summary["mean"] / (summary.m * K * math.sqrt(T) * math.log(T))
    return summary, mean_curve


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def grad_J(theta, w, X):
    """Exact-over-actions gradient of J at theta, averaged over contexts X."""
    P = softmax(X @ theta.T)                      # n x K
    MU = 0.5 + 0.5 * np.tanh(X @ w.T)             # n x K
    J = float((P * MU).sum(axis=1).mean())
    # d/dtheta_b sum_a pi_a mu_a = pi_b (mu_b - sum_a pi_a mu_a) x
    coef = P * (MU - (P * MU).sum(axis=1, keepdims=True))
    G = coef.T @ X / len(X)                       # K x m
    return G, J


def run_pg(seeds=range(5), T=100000, m=16, etas=(1e-3, 1.0, 10.0, 100.0), checkpoints=(100, 1000, 10000, 100000)):
    rows, var_rows = [], []
    for eta0 in etas:
        for seed in seeds:
            rng = np.random.default_rng(20_000 + seed)
            w = rng.normal(0.0, 3.0, size=(K, m))
            Xeval = unit_sphere(np.random.default_rng(30_000 + seed), 20_000, m)
            j_star = float((0.5 + 0.5 * np.tanh(Xeval @ w.T)).max(axis=1).mean())
            theta = np.zeros((K, m))
            b = 0.0
            G, J = grad_J(theta, w, Xeval)
            rows.append({"eta0": eta0, "seed": seed, "t": 0, "grad_sq": float((G ** 2).sum()), "J": J, "J_star": j_star})
            for t in range(1, T + 1):
                x = unit_sphere(rng, 1, m)[0]
                p = softmax(theta @ x)
                a = int(rng.choice(K, p=p))
                r = float(rng.random() < 0.5 + 0.5 * math.tanh(float(w[a] @ x)))
                b = 0.9 * b + 0.1 * r
                score = -np.outer(p, x)
                score[a] += x
                theta = theta + eta0 / (t + 1) * score * (r - b)
                if t in checkpoints:
                    G, J = grad_J(theta, w, Xeval)
                    rows.append({"eta0": eta0, "seed": seed, "t": t, "grad_sq": float((G ** 2).sum()), "J": J, "J_star": j_star})
                    est = {"baseline": [], "none": []}
                    for _ in range(256):
                        xs = unit_sphere(rng, 1, m)[0]
                        ps = softmax(theta @ xs)
                        s = int(rng.choice(K, p=ps))
                        rs = float(rng.random() < 0.5 + 0.5 * math.tanh(float(w[s] @ xs)))
                        sc = -np.outer(ps, xs)
                        sc[s] += xs
                        est["baseline"].append((sc * (rs - b)).ravel())
                        est["none"].append((sc * rs).ravel())
                    vb = float(np.var(np.array(est["baseline"]), axis=0).sum())
                    vn = float(np.var(np.array(est["none"]), axis=0).sum())
                    var_rows.append({"eta0": eta0, "seed": seed, "t": t, "reduction": 1 - vb / vn})
    pg = pd.DataFrame(rows)
    var = pd.DataFrame(var_rows)
    pg.to_csv(OUT / "theory_v2_pg.csv", index=False)
    var.to_csv(OUT / "theory_v2_pg_variance.csv", index=False)
    return pg, var


if __name__ == "__main__":
    import sys
    if "--pg-only" in sys.argv:
        pg, var = run_pg()
        print(pg.groupby(["eta0", "t"])[["grad_sq", "J", "J_star"]].mean().round(5).to_string())
        print(var.groupby(["eta0", "t"]).reduction.mean().round(3).to_string())
        sys.exit()
    ts, curve = run_ts()
    print("== Thompson sampling (Theorem 3 assumptions)\n", ts.round(4).to_string())
    print(curve[curve.t.isin([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000])]
          .pivot(index="t", columns="m", values="cum_regret").round(1).to_string())
    pg, var = run_pg()
    print("== Policy gradient (Theorem 2 assumptions)")
    print(pg.groupby(["eta0", "t"])[["grad_sq", "J", "J_star"]].mean().round(5).to_string())
    print(var.groupby(["eta0", "t"]).reduction.mean().round(3).to_string())
    for eta0 in sorted(pg.eta0.unique()):
        print(f"sum eta_t up to 100000 for eta0={eta0}: {eta0 * (sum(1 / (t + 1) for t in range(1, 100001))):.4f}")
