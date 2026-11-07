"""One-shot parameter selection vs NMPC from ten initial states (Tables S8, S9).

ours: with the decay constraint, plain: without it, single: one value for all ten states, mean: mean of the bank.
Each controller runs 20 s on the true plant.

    python scripts/one_shot.py arm [--no-nmpc]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spectral_koopman import get_system
from spectral_koopman.model import load_model
from spectral_koopman.pole_level import fit_bounds
from spectral_koopman.selection import BANK, BETA, Selector, initial_states, run_fixed, summarize_run
from spectral_koopman.trajectory_level import load_region


def make_selector(results, system):
    d = results / system.name / "spectral"
    labels = dict(np.load(d / "pole_labels.npz"))
    rate_bound = fit_bounds(labels, system.bounds)[0]["rate"]
    return Selector(system, load_model(d / "model.pth", system), load_region(d / "region.npz"), rate_bound,
                    float(labels["kappa_max"]))


def evaluate_theta(system, sel, x0, theta, info=None):
    return {"theta": np.asarray(theta).tolist(), "selection": info,
            "true_rate": float(system.exact_poles(np.asarray(theta)).real.max()),
            "calibrated_rate": float(sel.calibrated_rate(theta)[0]),
            **summarize_run(*run_fixed(system, x0, theta), BETA[system.name])}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("system")
    parser.add_argument("--results", type=Path, default=root / "results")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-nmpc", action="store_true")
    args = parser.parse_args()

    system = get_system(args.system)
    sel = make_selector(args.results, system)
    bank = sel.region.sample(BANK, np.random.default_rng(11))
    x0s = initial_states(system)
    single, single_info = sel.select(x0s, bank, decay=True)
    out = {"x0": x0s.tolist(), "single_theta": single.tolist(), "runs": []}
    for i, x0 in enumerate(x0s):
        rec = {"x0": x0.tolist()}
        for key, decay in (("ours", True), ("plain", False)):
            th, info = sel.select(x0[None], bank, decay)
            rec[key] = evaluate_theta(system, sel, x0, th, info)
        rec["single"] = evaluate_theta(system, sel, x0, single, single_info)
        rec["mean"] = evaluate_theta(system, sel, x0, bank.mean(0))
        if not args.no_nmpc:
            from spectral_koopman.nmpc import run_nmpc
            traj, u, ok, times = run_nmpc(system, x0, warm_theta=single)
            rec["nmpc"] = {"seconds": float(np.sum(times)), **summarize_run(traj, u, ok, BETA[system.name])}
        out["runs"].append(rec)
        print(system.name, i, {k: (rec[k]["total_cost"], rec[k]["settled"]) for k in rec if k != "x0"}, flush=True)
    (args.out or args.results / f"one_shot_{system.name}.json").write_text(json.dumps(out, indent=1))
