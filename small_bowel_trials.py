"""Small-bowel-only comparisons against the preserved 256-pixel submission."""
import argparse
import csv
import json
import time

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.nn import functional as F

import unet
from atlas import OUTPUT, resize
from dataset import DatasetArchive, encode
from unet import UNet, Slices, scan_probabilities, smooth_slices
from volume_metrics import normalized_hausdorff

ORGAN = "small_bowel"
ROOT = OUTPUT / "small_bowel_trials"
BASELINE = OUTPUT / "component256_submission.csv"


def load(names, device, context=unet.CONTEXT):
    models = torch.nn.ModuleList()
    for name in names:
        model = UNet(packed=True, context=context).to(device, memory_format=torch.channels_last)
        model.load_state_dict(torch.load(OUTPUT / name, map_location=device, weights_only=True))
        models.append(model)
    return models.eval()


def zoom(image, scale):
    matrix = image.new_zeros((len(image), 2, 3))
    matrix[:, 0, 0] = matrix[:, 1, 1] = scale
    return F.grid_sample(image, F.affine_grid(matrix, image.shape, align_corners=False),
                         align_corners=False)


def probabilities(model, split, device, mode):
    base = Slices(split)
    if mode == "ensemble":
        extra = Slices(split, root=OUTPUT / "cache")
        aux_model = load(["packed_branch_start.pt", "packed_control.pt", "packed_scale_jitter.pt"], device)
        extra_scans = scan_probabilities(aux_model, extra, device, 24, True)
    if mode == "context_ensemble":
        extra = Slices(split, context=5)
        aux_model = load(["packed256_context5_scale_jitter.pt"], device, context=5)
        extra_scans = scan_probabilities(aux_model, extra, device, 24, True)
    offset = 0
    scans = scan_probabilities(model, base, device, 24, True)
    for keys, _, flipped in scans:
        bowel = smooth_slices(flipped[:, 1:2])[:, 0]
        if mode in ("ensemble", "context_ensemble"):
            aux_keys, _, aux = next(extra_scans)
            assert keys == aux_keys
            aux = smooth_slices(aux[:, 1:2])[:, 0]
            if mode == "ensemble":
                aux = np.stack([
                    resize(p, (bowel.shape[-1], bowel.shape[-2]), Image.Resampling.BILINEAR)
                    for p in aux
                ])
            weight = 0.5 if mode == "context_ensemble" else 0.25
            bowel = (1 - weight) * bowel + weight * aux
        if mode == "zoom":
            chunks = []
            for i in range(0, len(keys), 24):
                x = base.inputs(np.arange(offset + i, offset + min(i + 24, len(keys))), device)
                votes = []
                for scale in (0.94, 1.06):
                    pred = unet.infer(model, zoom(x, scale))[:, 1:2]
                    votes.append(zoom(pred, 1 / scale)[:, 0].cpu().numpy())
                chunks.append((votes[0] + votes[1]) / 2)
            extra_votes = smooth_slices(np.concatenate(chunks)[:, None])[:, 0]
            bowel = 0.7 * bowel + 0.3 * extra_votes
        offset += len(keys)
        yield keys, bowel


def cleaned(volume, threshold, minimum):
    labels, _ = ndimage.label(volume >= threshold)
    small = np.bincount(labels.ravel()) < minimum
    small[0] = False
    return np.where(small[labels], 0, volume)


def native(image, shape, threshold):
    return resize(image, shape[::-1], Image.Resampling.BILINEAR) >= threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("threshold", "ensemble", "zoom", "components", "resolution", "context", "weighted_loss",
                                        "threshold_components", "translation", "context_ensemble"))
    args = parser.parse_args()
    device = torch.device("cuda")
    archive = DatasetArchive()
    ROOT.mkdir(exist_ok=True)
    checkpoint = {
        "resolution": "packed320_scale_jitter.pt",
        "context": "packed256_context5_scale_jitter.pt",
        "weighted_loss": "packed256_scale_jitter_bowel_weight1.5.pt",
        "translation": "packed256_scale_jitter_translation0.1.pt",
    }.get(args.mode, "packed256_scale_jitter.pt")
    model = load([checkpoint], device)
    settings = [(0.5, 1820)]
    if args.mode == "threshold":
        settings += [(t, 1820) for t in (0.35, 0.4, 0.45, 0.55, 0.6, 0.65)]
    if args.mode == "components":
        settings += [(0.5, cutoff) for cutoff in (0, 448, 900, 3000, 5000)]
    if args.mode == "threshold_components":
        settings = [(0.45, cutoff) for cutoff in (1820, 448, 900, 1300, 2400)]
    if args.mode == "resolution":
        settings = [(0.5, 2844)]
    counts = np.zeros((len(settings), 3), dtype=np.int64)
    distances = [[] for _ in settings]
    start = time.perf_counter()
    for j, (keys, probs) in enumerate(probabilities(model, "validation", device, args.mode), 1):
        targets = [archive.mask(k, ORGAN).astype(bool) for k in keys]
        low = np.stack([resize(t, (128, 128)) for t in targets])
        for i, (threshold, minimum) in enumerate(settings):
            filtered = cleaned(probs, threshold, minimum)
            sample = np.stack([resize(p, (128, 128)) for p in filtered]) >= threshold
            distances[i].append(normalized_hausdorff(low, sample))
            for k, p, t in zip(keys, filtered, targets):
                mask = native(p, archive.shapes[k], threshold)
                counts[i] += (np.count_nonzero(mask & t), mask.sum(), t.sum())
        if j % 15 == 0:
            print(f"{args.mode}: {j} validation scans", flush=True)
    trials = []
    for (threshold, minimum), (hits, pred, truth), values in zip(settings, counts, distances):
        dice = float(2 * hits / max(pred + truth, 1))
        hd = float(np.mean(values))
        trials.append(dict(threshold=threshold, minimum=minimum, dice=dice, hausdorff=hd,
                           proxy=0.4 * dice + 0.6 * (1 - hd)))
    winner = max(trials, key=lambda trial: trial["proxy"])
    logs = {
        "resolution": "packed320_scale_jitter_epochs.jsonl",
        "context": "packed256_context5_scale_jitter_epochs.jsonl",
        "weighted_loss": "packed256_scale_jitter_bowel_weight1.5_epochs.jsonl",
        "translation": "packed256_scale_jitter_translation0.1_epochs.jsonl",
    }
    training_seconds = 0
    if args.mode in logs:
        epochs = (OUTPUT / logs[args.mode]).read_text().splitlines()
        training_seconds = sum(json.loads(line)["train_seconds"] for line in epochs)
    report = dict(mode=args.mode, trials=trials, selected=winner,
                  evaluation_seconds=time.perf_counter() - start, training_seconds=training_seconds,
                  hausdorff_grid=[128, 128])
    (ROOT / f"{args.mode}.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)
    with BASELINE.open(newline="") as source:
        predictions = {(r["id"], r["class"]):r["predicted"] for r in csv.DictReader(source)}
    for keys, probs in probabilities(model, "test", device, args.mode):
        filtered = cleaned(probs, winner["threshold"], winner["minimum"])
        for key, p in zip(keys, filtered):
            predictions[key, ORGAN] = encode(native(p, archive.shapes[key], winner["threshold"]))
    temporary = ROOT / f"{args.mode}.tmp"
    archive.save(temporary, predictions)
    temporary.replace(ROOT / f"{args.mode}.csv")


if __name__ == "__main__":
    main()
