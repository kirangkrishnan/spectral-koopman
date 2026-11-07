"""One-shot parameter selection.

min over theta of sum_{k<H} ||x_hat_{k+1}||^2 + beta ||u(x_hat_k)||^2  (H = 40 steps)
s.t. theta in R, kappa(theta) <= kappa_max, r_hat(theta) + U_r(theta) <= -0.2.
Best value of a bank of 3000 from R, then SLSQP (20 iterations). decay=False drops the last two constraints.
"""
import time

import numpy as np
import torch
from scipy.optimize import minimize

from .model import kappa, rate_damping
from .systems import DT

SIGMA, HORIZON, BANK = 0.2, 40, 3000
BETA = {"vdp": 0.1, "arm": 0.1, "cartpole": 0.1, "quadrotor": 0.1, "double_pendulum": 1e-5}


class Selector:
    def __init__(self, system, model, region, rate_bound, kappa_max, dt=DT):
        self.system, self.model, self.region, self.dt = system, model, region, dt
        self.rate_bound, self.kappa_max = rate_bound, kappa_max
        self.beta = BETA[system.name]
        self.bounds = np.asarray(system.bounds, dtype=float)

    def cost(self, thetas, x0s):
        # predicted cost of each theta, summed over the initial states
        th = torch.tensor(np.atleast_2d(thetas))
        total = torch.zeros(len(th), dtype=th.dtype)
        with torch.no_grad():
            for x0 in np.atleast_2d(x0s):
                x = torch.tensor(np.repeat(x0[None], len(th), 0))
                pred = self.model.predict(x, th, HORIZON)
                prev = torch.cat((x[:, None], pred[:, :-1]), 1)
                u = self.system.feedback(prev, th[:, None].expand(-1, HORIZON, -1))
                total += pred.square().sum((1, 2)) + self.beta * u.square().reshape(len(th), -1).sum(1)
        return total.numpy()

    def calibrated_rate(self, thetas):
        thetas = np.atleast_2d(thetas)
        return rate_damping(self.model, thetas, self.dt)[0] + self.rate_bound.upper(thetas)

    def feasible(self, thetas, decay=True):
        thetas = np.atleast_2d(thetas)
        ok = self.region.contains(thetas)
        if decay:
            ok &= (kappa(self.model, thetas) <= self.kappa_max) & (self.calibrated_rate(thetas) <= -SIGMA)
        return ok

    def select(self, x0s, bank, decay=True):
        t0 = time.perf_counter()
        ok = self.feasible(bank, decay)
        costs = np.where(ok, self.cost(bank, x0s), np.inf)
        best = bank[int(np.argmin(costs))].copy()
        best_cost = float(costs.min())
        cons = [{"type": "ineq", "fun": lambda th: np.log1p(12.0) - np.log1p(min(self.region.upper(th[None])[0], 1e6))}]
        if decay:
            cons.append({"type": "ineq",
                         "fun": lambda th: -SIGMA - float(np.nan_to_num(self.calibrated_rate(th[None])[0], nan=1.0))})
        res = minimize(lambda th: float(self.cost(th, x0s)[0]), best, method="SLSQP",
                       bounds=list(zip(self.bounds[:, 0], self.bounds[:, 1])), constraints=cons,
                       options={"maxiter": 20, "ftol": 1e-9})
        if np.isfinite(res.x).all() and self.feasible(res.x, decay)[0]:
            c = float(self.cost(res.x, x0s)[0])
            if c < best_cost:
                best, best_cost = res.x.copy(), c
        return best, {"predicted_cost": best_cost, "seconds": time.perf_counter() - t0,
                      "feasible_share": float(ok.mean())}


def initial_states(system, count=10, seed=7):
    # corner of the grid with the largest scaled norm + 9 random grid points (not the origin)
    grid = system.grid()
    corner = int(np.argmax(np.linalg.norm(grid / np.asarray(system.initial_ranges), axis=1)))
    others = [i for i in range(len(grid)) if i != corner and np.linalg.norm(grid[i]) > 0]
    rest = np.random.default_rng(seed).choice(others, count - 1, replace=False)
    return grid[np.r_[corner, rest]]


def summarize_run(traj, u, ok, beta, dt=DT):
    norm = np.linalg.norm(traj, axis=1)
    fin = np.isfinite(norm)
    return {"valid": bool(ok), "settled": bool(ok and np.all(norm[-int(1 / dt):] <= 0.05)),
            "total_cost": float((np.nansum(norm ** 2) + beta * np.nansum(u ** 2)) * dt) if ok else None,
            "state_cost": float(np.nansum(norm[fin] ** 2) * dt) if ok else None}


def run_fixed(system, x0, theta, duration=20.0, dt=DT):
    traj, ok = system.simulate(x0, theta, int(round(duration / dt)), dt)
    u = system.feedback(np.nan_to_num(traj), np.asarray(theta)).reshape(len(traj), -1)
    return traj, np.where(np.isfinite(traj[:, :1]), u, np.nan), ok
