"""Summarize measured resource use, learning curves, and held-out architecture results."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("outputs/architecture")


def main():
    comparison = []
    reference = None
    for path in sorted(ROOT.glob("*_assessment.json")):
        name = path.stem.removesuffix("_assessment")
        data = json.loads(path.read_text())
        for variant in ("raw", "fixed_no_cleanup", "fixed_cleanup", "tuned_cleanup"):
            row = data[variant]
            if name == "reference" and variant == "fixed_cleanup":
                reference = row
            comparison.append(dict(run=name, evaluation=variant, macro_dice=row["macro_dice"],
                                   hausdorff=row["mean_normalized_hausdorff"],
                                   validation_proxy=row["validation_proxy"],
                                   large_bowel=row["organs"]["large_bowel"]["dice"],
                                   small_bowel=row["organs"]["small_bowel"]["dice"],
                                   stomach=row["organs"]["stomach"]["dice"],
                                   thresholds=json.dumps([r["threshold"] for r in row["organs"].values()])))
    if reference:
        for row in comparison:
            row["submission_gate"] = (row["macro_dice"] > reference["macro_dice"]
                                      and row["hausdorff"] < reference["mean_normalized_hausdorff"])
    if comparison:
        with (ROOT / "comparison.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(comparison[0]))
            writer.writeheader()
            writer.writerows(comparison)
    rows = []
    suspension = ROOT / "suspension.json"
    sleep = json.loads(suspension.read_text(encoding="utf-8-sig")) if suspension.exists() else {}
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for path in sorted(ROOT.glob("*_epochs.jsonl")):
        name = path.stem.removesuffix("_epochs")
        data = [json.loads(line) for line in path.read_text().splitlines()]
        if not data:
            continue
        best = max(data, key=lambda row: row["macro_dice"])
        status_path = ROOT / f"{name}_status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        sleep_seconds = sleep.get("seconds", 0) if sleep.get("run") == name else 0
        reason = "suspended" if sleep_seconds else status.get("reason", "")
        rows.append(dict(run=name, epochs=len(data), best_epoch=best["epoch"],
                         best_macro_dice=best["macro_dice"],
                         seconds=sum(row["train_seconds"] for row in data) - sleep_seconds,
                         end_reason=reason,
                         last_macro_dice=data[-1]["macro_dice"]))
        for ax, key, label in zip(axes.flat, ("loss", "validation_loss", "macro_dice", "lr"),
                                  ("Training objective", "Validation BCE + Dice",
                                   "Validation macro Dice", "Learning rate")):
            ax.plot([r["epoch"] for r in data], [r[key] for r in data], ".-", label=name)
            ax.set(xlabel="Epoch", ylabel=label)
    if rows:
        axes[0, 0].legend()
        figure.savefig(ROOT / "learning_curves.png", dpi=160)
        with (ROOT / "runtimes.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
