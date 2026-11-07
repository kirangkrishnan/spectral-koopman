"""Trajectory error e(theta), domain D, accuracy region R and its audit.

e(theta) = max over the grid X0 and k = 1..H_e of 100 ||x_k - x_hat_k|| / max(||x_k||, 0.1)  (inf if a run fails).
D = {U_scout <= tau} comes from a first model trained on the whole admissible set. The final model's bound U is
calibrated on 1000 values (-> R_1) and then on 4000, and R = {theta in R_1 : U <= 12%}.
"""
import copy

import numpy as np
import torch

from .conformal import ConformalBound, FailureGate, RFFModel, clopper_pearson_upper, conformal_quantile
from .systems import DT

CAP = 1000.0   # errors above 1000% and failed runs are censored here


def traj_errors(system, model, thetas, horizon, dt=DT, grid=None, floor=0.1):
    grid = system.grid() if grid is None else np.asarray(grid)
    dtype = next(model.parameters()).dtype
    errs = np.full(len(thetas), np.inf)
    with torch.no_grad():
        for i, theta in enumerate(thetas):
            th = torch.tensor(np.repeat(np.asarray(theta)[None], len(grid), axis=0), dtype=dtype)
            pred = model.predict(torch.tensor(grid, dtype=dtype), th, horizon).cpu().numpy()
            if not np.isfinite(pred).all():
                continue
            worst = 0.0
            for j, x0 in enumerate(grid):
                true, ok = system.simulate(x0, theta, horizon, dt)
                if not ok:
                    worst = np.inf
                    break
                rel = (100 * np.linalg.norm(true[1:] - pred[j], axis=1)
                       / np.maximum(np.linalg.norm(true[1:], axis=1), floor))
                worst = max(worst, float(np.max(rel)))
            errs[i] = worst
    return errs


def sample_uniform(contains, bounds, count, rng, max_proposals=200000, batch=2048):
    # rejection sampling from the admissible set
    bounds = np.asarray(bounds, dtype=float)
    kept, n_acc, n_prop = [], 0, 0
    while n_acc < count and n_prop < max_proposals:
        draws = rng.uniform(bounds[:, 0], bounds[:, 1], size=(min(batch, max_proposals - n_prop), len(bounds)))
        sel = draws[contains(draws)]
        kept.append(sel[:count - n_acc])
        n_acc += len(kept[-1])
        n_prop += len(draws)
    if n_acc < count:
        raise RuntimeError(f"only {n_acc}/{count} accepted after {n_prop} proposals")
    return np.concatenate(kept)


class TrajectoryBound:
    """U(theta) of e(theta), set to inf outside the domain and where the failure gate rejects."""

    def __init__(self, bounds, domain=None, seed=8):
        self.bounds = np.asarray(bounds, dtype=float)
        self.domain = domain
        self.bound = ConformalBound(bounds, offset=1.0, fit_cap=CAP, cal_cap=CAP, seed=seed)
        self.gate = None

    def fit(self, thetas, errs, gate_min_failures=None):
        failed = ~np.isfinite(errs)
        if gate_min_failures is not None and min(failed.sum(), (~failed).sum()) >= gate_min_failures:
            self.gate = FailureGate(self.bounds).fit(thetas, failed)
            thetas, errs = thetas[~failed], errs[~failed]
        self.bound.fit(thetas, errs)
        return self

    def calibrate(self, thetas, errs, alpha=0.05):
        if self.gate is not None:
            failed = ~np.isfinite(errs)
            self.gate.calibrate(thetas[failed])
            thetas, errs = thetas[~failed], errs[~failed]
        self.bound.calibrate(thetas, errs, alpha)
        return self

    @property
    def q(self):
        return self.bound.q

    def upper(self, thetas):
        u = self.bound.upper(thetas)
        keep = np.ones(len(u), bool)
        if self.domain is not None:
            keep &= self.domain.contains(thetas)
        if self.gate is not None:
            keep &= self.gate.passes(thetas)
        return np.where(keep, u, np.inf)

    def with_q(self, q):
        other = copy.copy(self)
        other.bound = copy.copy(self.bound)
        other.bound.q = q
        return other


class Domain:
    def __init__(self, scout_bound, tau):
        self.scout_bound, self.tau, self.bounds = scout_bound, float(tau), scout_bound.bounds

    def contains(self, thetas):
        return self.scout_bound.upper(thetas) <= self.tau

    def sample(self, count, rng):
        return sample_uniform(self.contains, self.bounds, count, rng, max(200000, 100 * count))


def select_domain(scout_bound, fraction=0.25, floor=12.0, probes=200000, seed=0):
    # tau = max(12%, 25% quantile of U_scout over the admissible set)
    cloud = np.random.default_rng(seed).uniform(scout_bound.bounds[:, 0], scout_bound.bounds[:, 1],
                                                (probes, len(scout_bound.bounds)))
    u = scout_bound.upper(cloud)
    qu = float(np.quantile(u, fraction, method="higher"))
    tau = max(floor, qu) if np.isfinite(qu) else np.inf
    if not np.isfinite(tau) or tau >= CAP or np.mean(u <= tau) == 0:
        raise RuntimeError("domain selection failed, the scout bound is too large almost everywhere")
    return Domain(scout_bound, tau)


class AccuracyRegion:
    # R = {theta in D : U_1 <= 12% and U <= 12%}. Uniform samples on R_1 restricted to R are uniform on R.
    def __init__(self, bound, q_first, acceptance=12.0):
        self.bound, self.first, self.acceptance = bound, bound.with_q(q_first), acceptance
        self.bounds = bound.bounds

    def upper(self, thetas):
        return self.bound.upper(thetas)

    def contains(self, thetas):
        return (self.bound.upper(thetas) <= self.acceptance) & (self.first.upper(thetas) <= self.acceptance)

    def sample(self, count, rng, max_proposals=200000):
        return sample_uniform(self.contains, self.bounds, count, rng, max_proposals)


def audit(errs, tol=15.0, limit=0.01, conf=0.95):
    fails = int(np.sum(np.asarray(errs) > tol))
    ub = clopper_pearson_upper(fails, len(errs), conf)
    return {"failures": fails, "count": len(errs), "upper_bound": ub, "passed": ub <= limit}


def quantile_on(bound, thetas, errs, alpha=0.05):
    # conformal quantile of an already fitted bound on another calibration set
    return conformal_quantile(bound.bound.residuals(thetas, errs), alpha)


def save_region(region, path):
    dom = region.bound.domain
    scout = dom.scout_bound
    a = {"bounds": region.bounds, "acceptance": np.asarray(region.acceptance), "tau": np.asarray(dom.tau),
         "q": np.asarray(region.bound.q), "q_first": np.asarray(region.first.q), "scout_q": np.asarray(scout.q),
         **region.bound.bound.g.arrays("final"), **scout.bound.g.arrays("scout")}
    if scout.gate is not None:
        a.update(scout.gate.clf.arrays("gate"), gate_threshold=np.asarray(scout.gate.threshold))
    np.savez_compressed(path, **a)


def load_region(path):
    a = dict(np.load(path))
    bounds = a["bounds"]

    def make(prefix, q, domain=None):
        tb = TrajectoryBound(bounds, domain)
        tb.bound.g = RFFModel.from_arrays(bounds, a, prefix)
        tb.bound.q = float(q)
        return tb

    scout = make("scout", a["scout_q"])
    if "gate_threshold" in a:
        scout.gate = FailureGate(bounds)
        scout.gate.clf = RFFModel.from_arrays(bounds, a, "gate")
        scout.gate.threshold = float(a["gate_threshold"])
    return AccuracyRegion(make("final", a["q"], Domain(scout, a["tau"])), float(a["q_first"]), float(a["acceptance"]))
