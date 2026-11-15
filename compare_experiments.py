"""Compare saved holdout predictions, fixed-weight blends, and pressure snapping.

No fitting uses validation targets. All comparisons use the same validation rows.
Pressure levels for snapping come only from the training partition's targets.
This script records results; it does not change the selected model or submit.
"""

import argparse
import itertools
import json

import numpy as np
import pandas as pd

from experiments import ARTIFACTS, ROOT, TRACK, snap_to_grid, split_breaths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--select-best", action="store_true",
                        help="Save a reloadable descriptor for the best measured candidate")
    args = parser.parse_args()
    results = pd.read_csv(TRACK / "results.csv")
    cache = ARTIFACTS / "cache_base"
    y = np.load(cache / "y.npy")
    mask = np.load(cache / "mask.npy")
    ids = np.load(cache / "breaths.npy")
    tr, va = split_breaths(ids)
    pressure_grid = np.unique(y[tr][mask[tr]])
    predictions = {}
    configurations = {}
    for result in results.to_dict("records"):
        folder = ROOT / result["model"]
        folder = folder.parent
        np.testing.assert_array_equal(ids[va], np.load(folder / "validation_breath_ids.npy"))
        pred = np.load(folder / "validation_predictions.npy")
        measured = np.abs(pred - y[va])[mask[va]].mean()
        np.testing.assert_allclose(measured, result["validation_mae"], atol=1e-6)
        predictions[result["name"]] = pred
        configurations[result["name"]] = json.loads((folder / "result.json").read_text())["args"]
    (TRACK / "configurations.json").write_text(json.dumps(configurations, indent=2))

    rows = []

    def evaluate(name, pred, members, snapped=False):
        if snapped:
            pred = snap_to_grid(pred, pressure_grid)
        rows.append(dict(name=name, validation_mae=float(np.abs(pred - y[va])[mask[va]].mean()),
                         step0_mae=float(np.abs(pred[:, 0] - y[va, 0]).mean()),
                         snapped=snapped, members=json.dumps(members)))

    for name, pred in predictions.items():
        evaluate(name, pred, {name: 1.0})
        evaluate(name + "_snap", pred, {name: 1.0}, snapped=True)
    for a, b in itertools.combinations(predictions, 2):
        for weight in (0.1, 0.25, 0.5, 0.75, 0.9):
            pred = predictions[a] * weight + predictions[b] * (1 - weight)
            name = f"blend_{a}_{b}_w{weight:.2f}"
            evaluate(name, pred, {a: weight, b: 1 - weight})
            evaluate(name + "_snap", pred, {a: weight, b: 1 - weight}, snapped=True)
    comparison = pd.DataFrame(rows).sort_values("validation_mae")
    comparison.to_csv(TRACK / "comparisons.csv", index=False)
    print(comparison[["name", "validation_mae", "step0_mae"]].head(20).to_string(index=False))
    print(f"Training-only pressure grid: {len(pressure_grid)} levels")

    # Resample complete validation breaths, preserving within-breath correlation.
    best_name = min(predictions, key=lambda name: float(
        np.abs(predictions[name] - y[va])[mask[va]].mean()
    ))
    best_error = np.abs(predictions[best_name] - y[va])
    rng = np.random.default_rng(42)
    samples = rng.integers(0, len(va), size=(500, len(va)))
    counts = mask[va].sum(axis=1)
    denominators = counts[samples].sum(axis=1)
    uncertainty = []
    for name, pred in predictions.items():
        differences = ((np.abs(pred - y[va]) - best_error) * mask[va]).sum(axis=1)
        bootstrap = differences[samples].sum(axis=1) / denominators
        uncertainty.append(dict(model=name, reference=best_name,
                                mae_difference=float(differences.sum() / counts.sum()),
                                ci_low=float(np.quantile(bootstrap, 0.025)),
                                ci_high=float(np.quantile(bootstrap, 0.975))))
    pd.DataFrame(uncertainty).to_csv(TRACK / "paired_bootstrap.csv", index=False)
    ablations = []
    for changed, reference in (
        ("lstm_extended", "lstm_base"),
        ("tree_without_leads", "tree_capacity_127"),
        ("tree_without_physics", "tree_capacity_127"),
        ("tree_l2_loss", "tree_capacity_127"),
        ("tree_all_phases", "tree_capacity_127"),
    ):
        if changed not in predictions or reference not in predictions:
            continue
        difference = ((np.abs(predictions[changed] - y[va])
                       - np.abs(predictions[reference] - y[va])) * mask[va]).sum(axis=1)
        bootstrap = difference[samples].sum(axis=1) / denominators
        ablations.append(dict(changed=changed, reference=reference,
                              mae_difference=float(difference.sum() / counts.sum()),
                              ci_low=float(np.quantile(bootstrap, 0.025)),
                              ci_high=float(np.quantile(bootstrap, 0.975))))
    pd.DataFrame(ablations).to_csv(TRACK / "ablation_bootstrap.csv", index=False)

    if args.select_best:
        winner = comparison.iloc[0].to_dict()
        members = json.loads(winner["members"])
        selected = ARTIFACTS / "selected"
        selected.mkdir(exist_ok=True)
        by_name = {row["name"]: row for row in results.to_dict("records")}
        descriptor = dict(members=[dict(weight=weight, run=by_name[name])
                                   for name, weight in members.items()])
        pred = sum(predictions[name] * weight for name, weight in members.items())
        if winner["snapped"]:
            np.save(selected / "pressure_grid.npy", pressure_grid)
            descriptor["pressure_grid"] = (selected / "pressure_grid.npy").relative_to(ROOT).as_posix()
            pred = snap_to_grid(pred, pressure_grid)
        np.save(selected / "validation_predictions.npy", pred)
        np.save(selected / "validation_breath_ids.npy", ids[va])
        error = np.abs(pred - y[va]).astype("float64")
        counts_by_step = mask[va].sum(axis=0)
        pd.DataFrame(dict(step=np.arange(80), rows=counts_by_step, mae=np.divide(
            (error * mask[va]).sum(axis=0), counts_by_step,
            out=np.full(80, np.nan), where=counts_by_step > 0,
        ))).to_csv(selected / "mae_by_step.csv", index=False)
        columns = json.loads((cache / "columns.json").read_text())
        x = np.load(cache / "X.npy", mmap_mode="r")
        codes = x[va, 0, columns.index("R_C")].astype(int)
        lungs = []
        for code in np.unique(codes):
            chosen = codes == code
            lungs.append(dict(R=[5, 20, 50][code // 3], C=[10, 20, 50][code % 3],
                              mae=float(error[chosen][mask[va][chosen]].mean())))
        pd.DataFrame(lungs).sort_values("mae", ascending=False).to_csv(
            selected / "mae_by_lung.csv", index=False
        )
        (selected / "ensemble.json").write_text(json.dumps(descriptor, indent=2))
        (TRACK / "best.json").write_text(json.dumps(dict(
            name=winner["name"], kind="ensemble", validation_mae=winner["validation_mae"],
            step0_mae=winner["step0_mae"], model=(selected / "ensemble.json").relative_to(ROOT).as_posix(),
            split_sha256=results.iloc[0]["split_sha256"],
            note="Training-partition checkpoints; preprocessing and any pressure grid use training data only",
        ), indent=2))
        print("Selected and saved reloadable best model/ensemble; nothing submitted")


if __name__ == "__main__":
    main()
