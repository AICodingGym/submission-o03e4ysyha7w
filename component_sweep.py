"""Compare 3D component size cutoffs with a fixed segmentation model."""
import argparse
import json

import numpy as np
import torch
from scipy import ndimage

from atlas import OUTPUT, dice
from dataset import CLASSES, SIZE, DatasetArchive, encode
from unet import Slices, UNet, metrics_row, native_masks, scan_probabilities, smooth_slices
from volume_metrics import normalized_hausdorff

CUTOFFS = (0, 64, 256, 1024, 4096)
THRESHOLDS = (0.30, 0.35, 0.35)
CHECKPOINTS = ("packed_branch_start.pt", "packed_control.pt", "packed_scale_jitter.pt")
PREFIX = "component"
if SIZE == 256:
    THRESHOLDS = (0.4, 0.5, 0.45)
    CHECKPOINTS = ("packed256_scale_jitter.pt",)
    PREFIX = "component256"
    CUTOFFS = (0, 128, 448, 1820, 7280)

if SIZE == 320:
    THRESHOLDS = (0.4, 0.5, 0.45)
    CHECKPOINTS = ("packed320_scale_jitter.pt",)
    PREFIX = "component320"
    CUTOFFS = (0, 700, 2844, 6000)


def components(probabilities):
    result = []
    for i, cutoff in enumerate(THRESHOLDS):
        labels, _ = ndimage.label(probabilities[:, i] >= cutoff)
        counts = np.bincount(labels.ravel())
        counts[0] = np.iinfo(counts.dtype).max
        result.append((labels, counts))
    return result


def filter_components(probabilities, labeled, minimum):
    if minimum == 0:
        return probabilities
    result = probabilities.copy()
    for i, (labels, counts) in enumerate(labeled):
        result[:, i][counts[labels] < minimum] = 0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--deep-encoder", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--minimum", type=int, nargs="+", default=CUTOFFS)
    args = parser.parse_args()
    if min(args.minimum) < 0:
        parser.error("component minimums must be non-negative")
    checkpoints = (args.checkpoint,) if args.checkpoint else CHECKPOINTS
    cutoffs = args.minimum
    prefix = args.prefix
    archive = DatasetArchive()
    device = torch.device("cuda")
    model = torch.nn.ModuleList()
    for name in checkpoints:
        member = UNet(packed=True, deep_encoder=args.deep_encoder).to(device, memory_format=torch.channels_last)
        member.load_state_dict(torch.load(OUTPUT / name, map_location=device, weights_only=True))
        model.append(member)
    model.eval()
    validation = Slices("validation")
    tally = np.zeros((len(cutoffs), 3, 3), dtype=np.int64)
    distances = [[] for _ in cutoffs]
    start = 0
    for scan_index, (keys, _, flipped) in enumerate(scan_probabilities(model, validation, device, 24, True)):
        probabilities = smooth_slices(flipped)
        labeled = components(probabilities)
        target_low = validation.masks[start:start+len(keys), :, ::2, ::2].astype(bool)
        full_targets = [[archive.mask(key, organ).astype(bool) for organ in CLASSES] for key in keys]
        start += len(keys)
        for trial, minimum in enumerate(cutoffs):
            filtered = filter_components(probabilities, labeled, minimum)
            distances[trial].append([
                normalized_hausdorff(target_low[:, i], filtered[:, i, ::2, ::2] >= cutoff)
                for i, cutoff in enumerate(THRESHOLDS)
            ])
            for key, channels, targets in zip(keys, filtered, full_targets):
                for i, (pred, truth) in enumerate(zip(
                    native_masks(channels, archive.shapes[key][::-1], THRESHOLDS), targets
                )):
                    tally[trial, :, i] += np.count_nonzero(pred & truth), pred.sum(), truth.sum()
        if (scan_index + 1) % 15 == 0:
            print(f"Evaluated {scan_index + 1} scans", flush=True)
    results = []
    for cutoff, counts, geometry in zip(cutoffs, tally, distances):
        row = metrics_row(dice(*counts))
        mean_hd = np.mean(geometry, axis=0)
        proxy = 0.4 * row["macro_dice"] + 0.6 * (1 - mean_hd.mean())
        row.update(minimum_voxels=cutoff, normalized_hausdorff=dict(zip(CLASSES, mean_hd.tolist())),
                   validation_proxy=float(proxy))
        results.append(row)
    best = max(results, key=lambda row: row["validation_proxy"])
    (OUTPUT / f"{prefix}_sweep.json").write_text(json.dumps({"experiments": results, "selected": best}, indent=2))
    print(json.dumps({"experiments": results, "selected": best}), flush=True)
    if args.validation_only:
        return
    predictions = {}
    for keys, _, flipped in scan_probabilities(model, Slices("test"), device, 24, True):
        probabilities = smooth_slices(flipped)
        filtered = filter_components(probabilities, components(probabilities), best["minimum_voxels"])
        for key, channels in zip(keys, filtered):
            for organ, mask in zip(CLASSES, native_masks(channels, archive.shapes[key][::-1], THRESHOLDS)):
                predictions[key, organ] = encode(mask)
    temporary = OUTPUT / f"{prefix}_submission.tmp"
    archive.save(temporary, predictions)
    temporary.replace(OUTPUT / f"{prefix}_submission.csv")
    print(f"Validated {len(predictions)} cleanup prediction rows", flush=True)


if __name__ == "__main__":
    main()
