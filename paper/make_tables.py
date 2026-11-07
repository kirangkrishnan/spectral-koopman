"""Writes the rows of the paper's tables (paper/tables/*.tex) from the results in paper/data."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SYSTEMS = [("vdp", "Van der Pol"), ("arm", "Two-link arm"), ("cartpole", "Cart-pole"), ("quadrotor", "Quadrotor"),
           ("double_pendulum", "Double pendulum")]
# exact-linearization variant: first calibration only, pole error measured in its own region
EXACT_REGION, EXACT_POLE_ERROR = ("0.25", "0$^\\ddagger$"), "0.014 / 0.036"
ABLATIONS = [  # label, arm variant, double pendulum variant
    ("Spectral model (method)", "spectral", "spectral"),
    ("\\quad seed 2", "seed2", "seed2"),
    ("\\quad seed 3", "seed3", "seed3"),
    ("Direct supervision of $M_\\theta$", "direct", "direct"),
    ("One-step loss $\\mathcal N=\\{1\\}$", "one_step", "one_step"),
    ("No spectral loss", "no_loss", "no_loss"),
    (None, None, None),
    ("Spectral model, noise 5\\%", "noise05", "noise05"),
    ("No spectral loss, noise 5\\%", "no_loss_noise05", "no_loss_noise05"),
]
NO_POLE_BOUND = {"arm/direct": "no accuracy region", "arm/noise05": "no accuracy region",
                 "double_pendulum/seed2": "audit failed (7/500)"}
# without the loss the noise only changes the labels, not the model
SAME_MODEL = {"no_loss_noise05": "no_loss"}


def load(name):
    return json.loads((HERE / "data" / f"{name}.json").read_text())


def signed(value, digits):
    return f"{value:+.{digits}f}".replace("-", "$-$").replace("+", "$+$")


def median_p90(rec):
    return f"{rec['median']:.3f} / {rec['p90']:.3f}"


def audit_cell(region):
    failures = region["audit_failures"]
    return f"{failures}" if region["audit_passed"] else f"\\textbf{{{failures}}}$^\\S$"


def both_levels_row(label, region_cells, bounds):
    r, z = bounds["rate"], bounds["damping"]
    return (f"{label} & {region_cells[0]} & {region_cells[1]} & {r['median_bound']:.3f} & {z['median_bound']:.3f} & "
            f"{100 * r['coverage']:.1f}\\% & {100 * z['coverage']:.1f}\\%\\\\")


def tables():
    regions, bounds, errors = load("regions"), load("pole_bounds"), load("pole_errors")
    regressor, long_horizon, one_shot = load("regressor"), load("long_horizon"), load("one_shot")
    out = {}

    # tables 2 and S4
    for variant in ("spectral", "no_loss"):
        rows = [both_levels_row(label, (f"{regions[f'{s}/{variant}']['share']:.2f}",
                                        audit_cell(regions[f"{s}/{variant}"])), bounds[f"{s}/{variant}"])
                for s, label in SYSTEMS]
        if variant == "spectral":
            rows.append(both_levels_row("\\quad exact linearization", EXACT_REGION, bounds["double_pendulum/exact"]))
        out[f"tab_uq_{variant}"] = rows
    out["tab_uq_pairs"] = []
    for spectral_row, no_loss_row in zip(out["tab_uq_spectral"], out["tab_uq_no_loss"]):
        a = [c.strip() for c in spectral_row.rstrip("\\\\").split("&")]
        b = [c.strip() for c in no_loss_row.rstrip("\\\\").split("&")]
        out["tab_uq_pairs"].append(" & ".join([a[0]] + [f"{x} / {y}" for x, y in zip(a[1:5], b[1:5])] + a[5:]) + "\\\\")
    exact = [c.strip() for c in out["tab_uq_spectral"][-1].rstrip("\\\\").split("&")]
    out["tab_uq_pairs"].append(" & ".join(exact[:5]) + " & " + " & ".join(exact[5:]) + "\\\\")

    # table 1
    rows = [f"{label} & " + " & ".join(median_p90(errors[s][k]) for k in
                                         ("spectral", "identified_4x10", "identified_16x20", "regressor", "no_loss"))
            + "\\\\" for s, label in SYSTEMS]
    rows.append(f"\\quad exact linearization$^\\ddagger$ & {EXACT_POLE_ERROR} & --- & --- & --- & ---\\\\")
    out["tab_eig"] = rows

    # table S5 (ablations)
    rows = []
    for label, *variants in ABLATIONS:
        if label is None:
            rows.append("\\midrule")
            continue
        cells = []
        for s, variant in zip(("arm", "double_pendulum"), variants):
            cells.append(median_p90(errors[s][SAME_MODEL.get(variant, variant)]))
            key = f"{s}/{variant}"
            if key in NO_POLE_BOUND:
                cells.append(NO_POLE_BOUND[key])
            else:
                mark = "$^\\ast$" if key == "arm/no_loss_noise05" else ""
                cells.append(f"{bounds[key]['rate']['median_bound']:.3f}{mark}")
        rows.append(f"{label} & " + " & ".join(cells) + "\\\\")
    out["tab_ablation"] = rows

    # table S3
    rows = []
    for s, label in SYSTEMS:
        b = bounds[f"{s}/spectral"]
        r, z, q = np.array(b["splits"]["rate"]), np.array(b["splits"]["damping"]), b["rate_coverage_by_quintile"]
        rows.append(f"{label} & {100 * r.mean():.2f} & {100 * r.min():.1f}--{100 * r.max():.1f}"
                    f" & {100 * z.mean():.2f} & {100 * z.min():.1f}--{100 * z.max():.1f}"
                    f" & {100 * min(q):.1f}--{100 * max(q):.1f}\\\\")
    out["tab_cp_splits"] = rows

    # table S6 (regressor)
    rows = []
    for s, label in SYSTEMS:
        r = regressor[s]
        cells = [median_p90(r["pole_errors"][k]) for k in ("spectral", "regressor")]
        cells += [f"{r['pole_bounds'][k]['rate']['bound_median']:.3f} / {r['pole_bounds'][k]['damping']['bound_median']:.3f}"
                  for k in ("spectral", "regressor")]
        cells += [f"{r['trajectory'][k]['median']:.1f} / {100 * r['trajectory'][k]['share_le_15']:.1f}\\%"
                  for k in ("spectral", "regressor")]
        rows.append(f"{label} & " + " & ".join(cells) + "\\\\")
    out["tab_jacobian"] = rows

    # table S7 (long horizon)
    rows = []
    for s, label in SYSTEMS:
        d = long_horizon[s]
        horizons = [str(h) for h in d["horizons"]]
        cells = [f"{d['spectral'][h]['median']:.1f} / {d['no_loss'][h]['median']:.1f}" for h in horizons]
        cells.append(" / ".join(f"{100 * d['spectral_better_share'][h]:.0f}" for h in horizons))
        rows.append(f"{label} ({horizons[0]}) & " + " & ".join(cells) + "\\\\")
    out["tab_long_horizon"] = rows

    # table S8 (one-shot)
    rows = []
    for s, label in SYSTEMS:
        runs, timing = one_shot[s]["runs"], one_shot[s]["timing"]
        ratio = lambda k: np.median([r[k]["total_cost"] / r["nmpc"]["total_cost"] if r[k]["valid"] else np.inf
                                     for r in runs])
        cells = [f"{ratio(k):.2f}" if np.isfinite(ratio(k)) else "--" for k in ("ours", "plain", "single")]
        cells += [f"{sum(r[k]['settled'] for r in runs)}" for k in ("ours", "plain", "single", "centroid")]
        cells += [signed(max(r[k]["true_rate"] for r in runs), 2) for k in ("ours", "plain")]
        cells += [f"{timing[k]:.1f}" if timing[k] < 10 else f"{timing[k]:.0f}"
                  for k in ("selection_seconds", "nmpc_seconds")]
        rows.append(f"{label} & " + " & ".join(cells) + "\\\\")
    out["tab_one_shot"] = rows

    # table S9
    rows = []
    variants = (("ours", "spectral", "$\\hat r+U_r$", "nmpc"), ("rhat_only", "spectral", "$\\hat r$", "nmpc_variants"),
                ("no_loss_bound", "no spectral loss", "$\\hat r+U_r$", "nmpc_variants"),
                ("no_loss_rhat", "no spectral loss", "$\\hat r$", "nmpc_variants_no_loss"))
    for s, label in SYSTEMS:
        runs = one_shot[s]["runs"]
        for i, (key, model, constraint, reference) in enumerate(variants):
            nmpc = (lambda r: r["nmpc"]["total_cost"]) if reference == "nmpc" else (lambda r: r[reference])
            cost = np.median([r[key]["total_cost"] / nmpc(r) if r[key].get("valid") else np.inf for r in runs])
            cells = [label if i == 0 else "", model, constraint, f"{cost:.2f}",
                     f"{sum(bool(r[key].get('settled')) for r in runs)}",
                     signed(max(r[key]["true_rate"] for r in runs), 3),
                     f"{100 * runs[0][key]['feasible_share_of_bank']:.1f}"]
            rows.append(" & ".join(cells) + "\\\\")
        rows.append("\\addlinespace[2pt]")
    out["tab_one_shot_variants"] = rows
    return out


def main():
    folder = HERE / "tables"
    folder.mkdir(exist_ok=True)
    for name, rows in tables().items():
        (folder / f"{name}.tex").write_text("\n".join(rows) + "\n")
        print(f"{name}.tex: {len(rows)} rows")


if __name__ == "__main__":
    main()
