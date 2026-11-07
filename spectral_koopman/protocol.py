"""Full pipeline for one system and one variant.

1. scout model on the whole admissible set -> bound U_scout -> domain D
2. final model trained on parameter values from D
3. its trajectory bound, R_1 (1000 calibration values) and R (4000), audit on R, check on fresh values from D
4. labeled values from R_kappa -> pole bounds and their audit
"""
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from .identification import Budget, training_targets
from .model import save_model
from .pole_level import label, pole_bounds, sample_population
from .systems import DT
from .trajectory_level import (AccuracyRegion, TrajectoryBound, audit, quantile_on, save_region, select_domain,
                               traj_errors)
from .training import train


def make_data(system, count, rng, steps, scale=1.0, thetas=None, retries=20):
    # without thetas: theta ~ U(admissible set), runs that fail are dropped.
    # with thetas (drawn from D): up to `retries` initial states per value, values that always fail are skipped.
    bounds = np.asarray(system.bounds)
    trajs, kept, tried = [], [], 0
    limit = 30 * count if thetas is None else len(thetas)
    while len(trajs) < count and tried < limit:
        theta = rng.uniform(bounds[:, 0], bounds[:, 1]) if thetas is None else np.asarray(thetas[tried])
        tried += 1
        for _ in range(1 if thetas is None else retries):
            x0 = system.sample_x0(1, rng, scale)[0]
            traj, ok = system.simulate(x0, theta, steps)
            if ok:
                break
        if ok:
            trajs.append(traj)
            kept.append(theta)
    if len(trajs) != count:
        raise RuntimeError(f"only {len(trajs)}/{count} valid training trajectories")
    return np.asarray(trajs), np.asarray(kept)


def budgets(cfg):
    # training targets: 4 runs x 10 steps, labels: 16 runs x 20 steps
    return (Budget(4, 10, noise=cfg.noise, seed=cfg.seed, purpose="train"),
            Budget(16, 20, noise=cfg.noise, seed=cfg.seed + 7919, purpose="label"))


def train_model(system, cfg, x, thetas, seed):
    targets = mask = None
    if cfg.spectral and cfg.exact_targets:
        targets = np.array([expm(system.jacobian(t) * DT) for t in thetas])
    elif cfg.spectral:
        targets, mask = training_targets(system, thetas, budgets(cfg)[0])
    return train(system, x, thetas, cfg.epochs, seed, cfg.latent_dim, cfg.hidden, targets=targets, mask=mask,
                 horizons=cfg.horizons, direct=cfg.direct_readout_loss)


def choose_domain(system, cfg):
    x, th = make_data(system, cfg.trajectories, np.random.default_rng(_data_seed(cfg, 0)), cfg.training_steps,
                      cfg.initial_state_scale)
    scout, _ = train_model(system, cfg, x, th, seed=cfg.seed)
    errs = _errors(system, scout, cfg)
    rngs = _rngs(cfg.seed + 100)
    box = np.asarray(system.bounds)
    th_fit = rngs[0].uniform(box[:, 0], box[:, 1], (cfg.fit, system.p))
    th_cal = rngs[1].uniform(box[:, 0], box[:, 1], (cfg.calibration, system.p))
    bound = TrajectoryBound(box).fit(th_fit, errs(th_fit), cfg.gate_min_failures)
    bound.calibrate(th_cal, errs(th_cal), cfg.alpha)
    return select_domain(bound, cfg.domain_fraction, cfg.acceptance, seed=cfg.seed + 100)


def final_model(system, cfg, domain):
    pool = domain.sample(int(np.ceil(1.25 * cfg.trajectories)), np.random.default_rng([cfg.seed + 100, 3]))
    x, th = make_data(system, cfg.trajectories, np.random.default_rng(_data_seed(cfg, 1)), cfg.training_steps,
                      cfg.initial_state_scale, thetas=pool)
    model, history = train_model(system, cfg, x, th, seed=cfg.seed + 1)
    return model, history, th


def accuracy_region(system, cfg, model, domain):
    errs = _errors(system, model, cfg)
    rngs = _rngs(cfg.seed + 200)
    th_fit = domain.sample(cfg.fit, rngs[0])
    e_fit = errs(th_fit)
    bound = TrajectoryBound(system.bounds, domain).fit(th_fit, e_fit)
    th_cal = np.vstack([domain.sample(cfg.calibration, rngs[1]),
                        domain.sample(cfg.extra_calibration, np.random.default_rng(cfg.seed + 300))])
    e_cal = errs(th_cal)
    n1 = cfg.calibration
    q1 = quantile_on(bound, th_cal[:n1], e_cal[:n1], cfg.alpha)
    bound.bound.q = quantile_on(bound, th_cal, e_cal, cfg.alpha)
    region = AccuracyRegion(bound, q1, cfg.acceptance)

    th_audit = region.sample(cfg.audit, rngs[2])
    e_audit = errs(th_audit)
    th_check = domain.sample(cfg.final_check, np.random.default_rng(cfg.seed + 400))
    e_check = errs(th_check)
    inside = region.contains(th_check)
    report = {"q_first": q1, "q": bound.q, "audit": audit(e_audit, cfg.tolerance, cfg.audit_limit),
              "check": {"coverage": float(np.mean(e_check <= region.upper(th_check))),
                        "in_region": int(inside.sum()),
                        "above_tol_in_region": int(np.sum(e_check[inside] > cfg.tolerance))}}
    data = {"fit_theta": th_fit, "fit_error": e_fit, "calibration_theta": th_cal, "calibration_error": e_cal,
            "audit_theta": th_audit, "audit_error": e_audit, "check_theta": th_check, "check_error": e_check}
    return region, report, data


def spectral_level(system, cfg, model, region):
    model = model.double()
    thetas, kappa_max = sample_population(model, region, cfg.pole_population, cfg.kappa_quantile)
    labels = label(system, model, thetas, budgets(cfg)[1], exact_labels=cfg.exact_targets)
    _, report = pole_bounds(labels, system.bounds, cfg.alpha)
    return {**labels, "kappa_max": np.asarray(kappa_max)}, report


def run(system, cfg, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    domain = choose_domain(system, cfg)
    model, history, th_train = final_model(system, cfg, domain)
    save_model(model, out / "model.pth", system, settings=cfg.__dict__)
    np.save(out / "training_theta.npy", th_train)
    (out / "training_history.json").write_text(json.dumps(history))
    region, traj_report, traj_data = accuracy_region(system, cfg, model, domain)
    save_region(region, out / "region.npz")
    np.savez_compressed(out / "trajectory_level.npz", **traj_data)
    labels, pole_report = spectral_level(system, cfg, model, region)
    np.savez_compressed(out / "pole_labels.npz", **labels)
    summary = {"system": system.name, "settings": cfg.__dict__, "tau": domain.tau,
               "trajectory_level": traj_report, "pole_level": pole_report}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


def _data_seed(cfg, stage):
    return np.random.SeedSequence(cfg.seed).spawn(4)[stage]


def _rngs(seed):
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]


def _errors(system, model, cfg):
    return lambda thetas: traj_errors(system, model, thetas, cfg.evaluation_steps, floor=cfg.error_floor)
