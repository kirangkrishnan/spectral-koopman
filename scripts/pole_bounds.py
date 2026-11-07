"""Pole bounds and their held-out coverage for every model with labels (Tables 2, S3-S5).

For the spectral models also the coverage over 20 random splits and by quintile of the learned rate.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spectral_koopman import SYSTEMS
from spectral_koopman.pole_level import fit_bounds, pole_bounds, scores


def coverage(labels, bounds, seed):
    fitted, parts = fit_bounds(labels, bounds, seed=seed)
    i, th = parts["audit"], labels["theta"]
    return {k: float(np.mean(s[i] <= fitted[k].upper(th[i]))) for k, (s, _) in scores(labels).items()}


def coverage_by_quintile(labels, bounds):
    fitted, parts = fit_bounds(labels, bounds)
    i, th = parts["audit"], labels["theta"]
    r_hat = labels["r_hat"][i]
    edges = np.quantile(r_hat, np.linspace(0, 1, 6))
    q = np.clip(np.searchsorted(edges, r_hat, side="right") - 1, 0, 4)
    hit = scores(labels)["rate"][0][i] <= fitted["rate"].upper(th[i])
    return [float(hit[q == k].mean()) for k in range(5)]


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=root / "results")
    parser.add_argument("--out", type=Path, default=root / "results" / "pole_bounds.json")
    args = parser.parse_args()
    out = {}
    for path in sorted(args.results.glob("*/*/pole_labels.npz")):
        name, variant = path.parent.parent.name, path.parent.name
        labels = dict(np.load(path))
        bounds = SYSTEMS[name].bounds
        _, rec = pole_bounds(labels, bounds)
        rec["labels"] = len(labels["theta"])
        if variant == "spectral":
            cov = [coverage(labels, bounds, s) for s in range(20)]
            rec["splits"] = {k: [c[k] for c in cov] for k in ("rate", "damping")}
            rec["rate_coverage_by_quintile"] = coverage_by_quintile(labels, bounds)
        out[f"{name}/{variant}"] = rec
        print(name, variant, {k: (round(rec[k]["median_bound"], 3), round(rec[k]["coverage"], 4))
                              for k in ("rate", "damping")}, flush=True)
    args.out.write_text(json.dumps(out, indent=1))
