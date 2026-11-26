"""Score and export architecture candidates using native masks and 3D distances."""
import argparse
import hashlib
import json

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from architecture_trials import ROOT, make_model
from atlas import OUTPUT, resize
from architecture_clock import awake_seconds
from dataset import CLASSES, SIZE, DatasetArchive, encode
from unet import Slices, scan_probabilities, smooth_slices
from volume_metrics import normalized_hausdorff

FIXED = (0.40, 0.45, 0.45)
MINIMUM = (1820, 1300, 1820)
GRID = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)


def load(spec):
    model, checkpoint = spec.split(":", 1)
    module = make_model(model).to("cuda", memory_format=torch.channels_last)
    path = OUTPUT / "packed256_scale_jitter.pt" if model == "current" else ROOT / f"{checkpoint}.pt"
    module.load_state_dict(torch.load(path, map_location="cuda", weights_only=True))
    return module.eval()


def volumes(models, split, batch, flip=True):
    data = Slices(split)
    scans = [scan_probabilities(m, data, torch.device("cuda"), batch, flip) for m in models]
    for group in zip(*scans, strict=True):
        keys = group[0][0]
        assert all(keys == item[0] for item in group)
        branch = 2 if flip else 1
        probabilities = group[0][branch]
        if len(group) > 1:
            probabilities = sum(item[branch] for item in group) / len(group)
        yield keys, smooth_slices(probabilities)


def clean(probabilities, threshold, minimum):
    if not minimum:
        return probabilities
    labels, _ = ndimage.label(probabilities >= threshold)
    sizes = np.bincount(labels.ravel())
    if minimum == -1:
        sizes[0] = 0
        keep = sizes.argmax()
        return np.where(labels == keep, probabilities, 0) if keep else probabilities
    small = sizes < minimum
    small[0] = False
    return np.where(small[labels], 0, probabilities)


def mask(probability, shape, threshold):
    return resize(probability, shape[::-1], Image.Resampling.BILINEAR) >= threshold


def organ_metrics(probabilities, targets, low_target, threshold, minimum):
    filtered = clean(probabilities, threshold, minimum)
    predicted_low = filtered[:, ::2, ::2] >= threshold
    distance = normalized_hausdorff(low_target, predicted_low)
    hits = predicted = actual = 0
    for probability, target in zip(filtered, targets):
        pred = mask(probability, target.shape, threshold)
        hits += np.count_nonzero(pred & target)
        predicted += np.count_nonzero(pred)
        actual += np.count_nonzero(target)
    return (hits, predicted, actual), distance


def evaluate(models, archive, batch, name, *, quick=False, flip=True, thresholds=FIXED, minimum=None, largest=False):
    start = awake_seconds()
    validation = Slices("validation")
    if minimum is None:
        minimum = tuple(round(n * (SIZE / 256) ** 2) for n in MINIMUM)
    minimums = tuple(minimum)
    settings = []
    for organ, (fixed, count) in enumerate(zip(thresholds, minimums)):
        trials = [(fixed, count)] if quick else [(0.5, 0), (fixed, 0), (fixed, count)] + [
            (threshold, count) for threshold in GRID
        ]
        if largest:
            trials.append((fixed, -1))
        settings.append(tuple(dict.fromkeys(trials)))
    aggregates = [[np.zeros(3, dtype=np.int64) for _ in trials] for trials in settings]
    distances = [[[] for _ in trials] for trials in settings]
    offset = 0
    for v, (keys, probabilities) in enumerate(volumes(models, "validation", batch, flip), 1):
        assert list(keys) == validation.keys[offset:offset + len(keys)]
        low = validation.masks[offset:offset + len(keys), :, ::2, ::2].astype(bool)
        offset += len(keys)
        for organ, cls in enumerate(CLASSES):
            targets = [archive.mask(k, cls).astype(bool) for k in keys]
            for i, (threshold, minimum) in enumerate(settings[organ]):
                counts, hd = organ_metrics(probabilities[:, organ], targets, low[:, organ],
                                            threshold, minimum)
                aggregates[organ][i] += counts
                distances[organ][i].append(hd)
        if v % 15 == 0:
            print(f"{name}: evaluated {v} validation scans", flush=True)
    per_organ = []
    for organ, trials in enumerate(settings):
        rows = {}
        for i, (threshold, minimum) in enumerate(trials):
            intersection, pred, target = aggregates[organ][i]
            dice = float(2 * intersection / max(pred + target, 1))
            hd = float(np.mean(distances[organ][i]))
            rows[threshold, minimum] = dict(threshold=threshold, minimum=minimum, dice=dice,
                                             hausdorff=hd, proxy=0.4*dice + 0.6*(1-hd))
        per_organ.append(rows)
    def summarize(rows):
        return dict(macro_dice=float(np.mean([r["dice"] for r in rows])),
                    mean_normalized_hausdorff=float(np.mean([r["hausdorff"] for r in rows])),
                    validation_proxy=float(np.mean([r["proxy"] for r in rows])),
                    organs=dict(zip(CLASSES, rows)))
    selected = [max((r for r in rows.values() if r["minimum"]),
                    key=lambda r: r["dice"] if quick else r["proxy"]) for rows in per_organ]
    report = dict(
        raw=None if quick else summarize([rows[0.5, 0] for rows in per_organ]),
        fixed_no_cleanup=None if quick else summarize([rows[fixed, 0] for rows, fixed in zip(per_organ, thresholds)]),
        fixed_cleanup=summarize([rows[fixed, minimum] for rows, fixed, minimum
                                 in zip(per_organ, thresholds, minimums)]),
        tuned_cleanup=summarize(selected),
        flip=flip,
        trials={organ: list(rows.values()) for organ, rows in zip(CLASSES, per_organ)},
        validation_seconds=awake_seconds() - start,
    )
    (ROOT / f"{name}_assessment.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "trials"}), flush=True)


def export(models, archive, batch, name, dice_only=False):
    report = json.loads((ROOT / f"{name}_assessment.json").read_text())
    reference = json.loads((ROOT / "reference_assessment.json").read_text())["fixed_cleanup"]
    winner = report["tuned_cleanup"]
    if (winner["macro_dice"] <= reference["macro_dice"] or
            (not dice_only and winner["mean_normalized_hausdorff"] >= reference["mean_normalized_hausdorff"])):
        raise ValueError("Candidate must improve Dice and Hausdorff against the frozen reference")
    predictions = {}
    for keys, probabilities in volumes(models, "test", batch, report.get("flip", True)):
        for j, organ in enumerate(CLASSES):
            config = winner["organs"][organ]
            threshold = config["threshold"]
            filtered = clean(probabilities[:, j], threshold, config["minimum"])
            for key, probability in zip(keys, filtered):
                predictions[key, organ] = encode(mask(probability, archive.shapes[key], threshold))
    temporary = ROOT / f"{name}.tmp"
    archive.save(temporary, predictions)
    path = ROOT / f"{name}_submission.csv"
    temporary.replace(path)
    print(f"Verified {len(predictions)} rows; SHA256 {hashlib.sha256(path.read_bytes()).hexdigest()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("evaluate", "export"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--largest", action="store_true")
    parser.add_argument("--thresholds", type=float, nargs=3, default=FIXED)
    parser.add_argument("--minimum", type=int, nargs=3)
    parser.add_argument("--dice-only", action="store_true")
    args = parser.parse_args()
    models = [load(spec) for spec in args.models]
    archive = DatasetArchive()
    if args.action == "evaluate":
        evaluate(models, archive, args.batch, args.name, quick=args.quick, flip=not args.no_flip,
                 thresholds=args.thresholds, minimum=args.minimum, largest=args.largest)
    else:
        export(models, archive, args.batch, args.name, dice_only=args.dice_only)


if __name__ == "__main__":
    main()

