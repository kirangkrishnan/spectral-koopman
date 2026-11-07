"""Least-squares identification of the sampled closed loop's linearization from short runs.

J runs of L steps from x0 = a * r * w (w ~ U[-1, 1]^n), measured with noise of std noise * a * r, then
F = argmin sum ||y_{k+1} - F y_k||^2 (solved in coordinates scaled by a * r). F estimates exp(A dt).
Each identification is seeded by theta itself so it can be repeated exactly.
"""
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from .systems import DT


@dataclass(frozen=True)
class Budget:
    runs: int
    steps: int
    amplitude: float = 0.05
    noise: float = 0.01
    seed: int = 8
    purpose: str = "train"


TRAIN_BUDGET = Budget(runs=4, steps=10, seed=8, purpose="train")           # training targets
LABEL_BUDGET = Budget(runs=16, steps=20, seed=8 + 7919, purpose="label")   # labels of the pole bounds


def _rng(theta, budget):
    key = sha256(np.ascontiguousarray(theta, dtype=float).tobytes() + budget.purpose.encode()).digest()
    return np.random.default_rng([budget.seed, int.from_bytes(key[:8], "little")])


def identify(system, theta, budget, dt=DT):
    """F_hat (n x n), or None if the runs don't determine it."""
    theta = np.asarray(theta, dtype=float)
    n = system.n
    rng = _rng(theta, budget)
    scale = budget.amplitude * np.asarray(system.initial_ranges, dtype=float)
    z0, z1 = [], []
    for _ in range(budget.runs):
        x0 = scale * rng.uniform(-1, 1, n)
        traj, _ = system.simulate(x0, theta, budget.steps, dt)
        y = traj + budget.noise * scale * rng.standard_normal(traj.shape)
        z = y / scale
        ok = np.isfinite(z).all(1)
        pairs = ok[:-1] & ok[1:]          # a failed run still gives its valid prefix
        z0.append(z[:-1][pairs])
        z1.append(z[1:][pairs])
    z0, z1 = np.concatenate(z0), np.concatenate(z1)
    if len(z0) < 2 * n:
        return None
    sv = np.linalg.svd(z0, compute_uv=False)
    if sv[-1] <= 1e-8 * sv[0]:
        return None
    fz = np.linalg.lstsq(z0, z1, rcond=None)[0].T
    return scale[:, None] * fz / scale[None, :]


def map_poles(f, dt=DT):
    return np.log(np.linalg.eigvals(f).astype(complex)) / dt


def training_targets(system, thetas, budget=TRAIN_BUDGET, dt=DT):
    # identity where identification failed; mask says where it worked
    targets = np.repeat(np.eye(system.n)[None], len(thetas), 0)
    mask = np.zeros(len(thetas), bool)
    for i, theta in enumerate(thetas):
        f = identify(system, theta, budget, dt)
        if f is not None and np.isfinite(f).all():
            targets[i], mask[i] = f, True
    return targets, mask


def measured_poles(system, thetas, budget=LABEL_BUDGET, dt=DT):
    # NaN rows where identification failed
    thetas = np.atleast_2d(thetas)
    out = np.full((len(thetas), system.n), np.nan, dtype=complex)
    for i, theta in enumerate(thetas):
        f = identify(system, theta, budget, dt)
        if f is not None:
            out[i] = map_poles(f, dt)
    return out
