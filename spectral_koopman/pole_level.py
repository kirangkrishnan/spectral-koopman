"""Bounds on the errors of the learned slowest rate and damping.

Population: uniform on R_kappa = {theta in R : r_hat <= 0, kappa <= kappa_max} (kappa_max = 99th percentile of kappa
on a separate sample). Labels are measured poles (16 runs x 20 steps); a failed identification gives r = inf,
zeta = -1. Split 50/30/20 into fit/calibration/audit, each score calibrated at alpha/2.
"""
import numpy as np

from .conformal import ConformalBound, clopper_pearson_lower, clopper_pearson_upper
from .identification import measured_poles
from .model import kappa, rate_damping, slowest_rate, smallest_damping
from .systems import DT

C, FIT_CAP = 1e-3, 10.0


def _stable_part(model, region, count, rng, dt, batch=20000, max_batches=500):
    # uniform on {theta in R : r_hat <= 0}
    kept = []
    for _ in range(max_batches):
        th = region.sample(batch, rng, 50 * batch)
        kept.append(th[rate_damping(model, th, dt)[0] <= 0])
        if sum(len(k) for k in kept) >= count:
            break
    pool = np.concatenate(kept)
    if len(pool) < count:
        raise RuntimeError(f"only {len(pool)}/{count} values of R with a stable learned linearization")
    return pool[:count]


def sample_population(model, region, count=20000, q_kappa=0.99, n_edge=4000, seed=1234, dt=DT):
    rng = np.random.default_rng(seed)
    edge = _stable_part(model, region, n_edge, rng, dt)
    kappa_max = float(np.quantile(kappa(model, edge), q_kappa))   # fixed before any label is used
    th = _stable_part(model, region, int(np.ceil(count / max(q_kappa, 0.5))), rng, dt)
    return th[kappa(model, th) <= kappa_max], kappa_max


def summarize(lam):
    bad = ~np.isfinite(lam).all(1)
    lam0 = np.nan_to_num(lam, nan=0.0)
    return (np.where(bad, np.inf, slowest_rate(lam0)),
            np.where(bad, -1.0, smallest_damping(lam0, stable_only=True)))


def label(system, model, thetas, budget, exact_labels=False, dt=DT):
    r_hat, z_hat = rate_damping(model, thetas, dt)
    exact = np.array([system.exact_poles(t) for t in thetas])
    meas = exact if exact_labels else measured_poles(system, thetas, budget, dt)
    r_meas, z_meas = summarize(meas)
    r_exact, z_exact = summarize(exact)
    return {"theta": thetas, "r_hat": r_hat, "z_hat": z_hat, "r_measured": r_meas, "z_measured": z_meas,
            "r_exact": r_exact, "z_exact": z_exact}


def scores(labels):
    # name -> (error vs measured pole, inf where it failed; error vs exact pole)
    out = {}
    for name, key in (("rate", "r"), ("damping", "z")):
        s = np.abs(labels[f"{key}_measured"] - labels[f"{key}_hat"])
        out[name] = (np.where(np.isfinite(s), s, np.inf), np.abs(labels[f"{key}_exact"] - labels[f"{key}_hat"]))
    return out


def split(n, fractions=(0.5, 0.3), seed=2026):
    idx = np.random.default_rng(seed).permutation(n)
    a, b = int(fractions[0] * n), int(fractions[1] * n)
    return {"fit": idx[:a], "calibration": idx[a:a + b], "audit": idx[a + b:]}


def fit_bounds(labels, bounds, alpha_each=0.025, seed=2026):
    th, parts = labels["theta"], split(len(labels["theta"]), seed=seed)
    i_fit, i_cal = parts["fit"], parts["calibration"]
    fitted = {}
    for name, (s, _) in scores(labels).items():
        ok = np.isfinite(s[i_fit])
        b = ConformalBound(bounds, C, FIT_CAP).fit(th[i_fit][ok], s[i_fit][ok])
        fitted[name] = b.calibrate(th[i_cal], s[i_cal], alpha_each)
    return fitted, parts


def pole_bounds(labels, bounds, alpha=0.05, seed=2026, audit_limit=0.05):
    fitted, parts = fit_bounds(labels, bounds, alpha / 2, seed)
    th, i_aud = labels["theta"], parts["audit"]
    n = len(i_aud)
    summary = {}
    for name, (s, s_exact) in scores(labels).items():
        u = fitted[name].upper(th[i_aud])
        miss = int(np.sum(s[i_aud] > u))
        summary[name] = {"median_bound": float(np.median(u)), "coverage": 1 - miss / n,
                         "coverage_of_exact_poles": 1 - int(np.sum(s_exact[i_aud] > u)) / n,
                         "miss_rate_ci95": [clopper_pearson_lower(miss, n), clopper_pearson_upper(miss, n)],
                         "audit_passed": clopper_pearson_upper(miss, n) <= audit_limit}
    return fitted, summary
