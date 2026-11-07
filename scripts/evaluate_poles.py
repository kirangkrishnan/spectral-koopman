"""Slowest-pole error |r_hat - r| at 500 values drawn from the spectral model's region (Table 1, Table S5)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spectral_koopman import SYSTEMS
from spectral_koopman.identification import Budget, identify, map_poles
from spectral_koopman.model import load_model, poles
from spectral_koopman.trajectory_level import load_region

MODELS = ["spectral", "no_loss", "regressor", "exact", "seed2", "seed3", "direct", "one_step", "noise05"]
# identification at the parameter value itself, with the training and the label budget
ID_BUDGETS = {"identified_4x10": Budget(4, 10, seed=8, purpose="evaluation"),
              "identified_16x20": Budget(16, 20, seed=8 + 7919, purpose="evaluation")}


def rate_err(lam, lam_true):
    return abs(lam.real.max() - lam_true.real.max()) if np.isfinite(lam).all() else np.inf


def stats(errs):
    errs = np.asarray(errs, float)
    fin = np.isfinite(errs)
    return {"median": float(np.median(errs[fin])), "p90": float(np.quantile(errs[fin], 0.9)),
            "failed": int((~fin).sum())}


def evaluate(results, name, count=500, seed=2718):
    system = SYSTEMS[name]
    thetas = load_region(results / name / "spectral" / "region.npz").sample(count, np.random.default_rng(seed))
    true = [system.exact_poles(t) for t in thetas]
    out = {}
    for v in MODELS:
        path = results / name / v / "model.pth"
        if path.exists():
            lam = poles(load_model(path, system), thetas, 0.05)
            out[v] = stats([rate_err(a, b) for a, b in zip(lam, true)])
    for key, budget in ID_BUDGETS.items():
        errs = []
        for th, lt in zip(thetas, true):
            f = identify(system, th, budget)
            errs.append(rate_err(map_poles(f), lt) if f is not None else np.inf)
        out[key] = stats(errs)
    return out


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=root / "results")
    parser.add_argument("--out", type=Path, default=root / "results" / "pole_errors.json")
    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    args = parser.parse_args()
    out = json.loads(args.out.read_text()) if args.out.exists() else {}
    for name in args.systems:
        out[name] = evaluate(args.results, name)
        print(name, {k: (round(v["median"], 3), round(v["p90"], 3)) for k, v in out[name].items()}, flush=True)
        args.out.write_text(json.dumps(out, indent=1))
