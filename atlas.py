"""Fit a position-only baseline and export checked predictions."""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from dataset import CLASSES, DatasetArchive, encode, volume

OUTPUT = Path("outputs")
ATLAS_SIZE = 96
BINS = 48


def resize(pixels, size, method=Image.Resampling.NEAREST):
    return np.asarray(Image.fromarray(pixels).resize(size, method))


def dice(inter, predicted, actual):
    return 2 * inter / np.maximum(predicted + actual, 1)


def position_keys(data, keys):
    length = {day: len(items) for day, items in data.volumes.items()}
    for key in keys:
        index = int(key.rsplit("_", 1)[1]) - 1
        position = min(BINS - 1, index * BINS // length[volume(key)])
        yield key, position


def make_review(archive, worst, path):
    from PIL import ImageDraw

    width, height = 224, 244
    sheet = Image.new("RGB", (width * 4, height * len(CLASSES)), "#111111")
    draw = ImageDraw.Draw(sheet)
    for organ_index, examples in enumerate(worst):
        for column, (score, key, pred, target) in enumerate(examples):
            raw = archive.image(key).astype(np.float32)
            cutoff = max(1, np.quantile(raw, 0.995))
            gray = np.uint8(np.clip(raw / cutoff, 0, 1) * 175)
            rgb = np.repeat(gray[:, :, None], 3, axis=2)
            rgb[target & ~pred] = (245, 88, 66)
            rgb[pred & ~target] = (65, 155, 255)
            rgb[pred & target] = (112, 224, 93)
            tile = Image.fromarray(rgb)
            tile.thumbnail((width, height - 22))
            x, y = column * width, organ_index * height
            sheet.paste(tile, (x, y + 20))
            draw.text((x + 2, y + 2), f"{CLASSES[organ_index]} {key.split('_')[0]} #{key.split('_')[-1]} Dice {score:.2f}")
    sheet.save(path)


def main():
    start = time.perf_counter()
    data = DatasetArchive()
    fit, validation = data.split()
    atlas = np.zeros((BINS, 3, ATLAS_SIZE, ATLAS_SIZE), dtype=np.float32)
    counts = np.zeros(BINS)
    for key, position in position_keys(data, fit):
        counts[position] += 1
        for index, organ in enumerate(CLASSES):
            if data.labels[key][organ]:
                atlas[position, index] += resize(data.mask(key, organ), (ATLAS_SIZE, ATLAS_SIZE))
    atlas /= np.maximum(counts, 1)[:, None, None, None]
    fit_seconds = time.perf_counter() - start
    thresholds = (0.055, 0.075, 0.10, 0.12, 0.20, 0.30)
    binary = atlas[None, ...] >= np.array(thresholds)[:, None, None, None, None]
    totals = np.zeros((3, len(thresholds), 3))
    for key, position in position_keys(data, validation):
        height, width = data.shapes[key]
        for index, organ in enumerate(CLASSES):
            target = data.mask(key, organ)
            for j, masks in enumerate(binary[:, position, index]):
                prediction = resize(masks, (width, height))
                totals[:, j, index] += (
                    np.count_nonzero(prediction & target), prediction.sum(), target.sum()
                )
    scores = dice(*totals)
    best = scores.argmax(axis=0)
    metrics = {
        "fit_seconds": fit_seconds,
        "evaluation_seconds": time.perf_counter() - start - fit_seconds,
        "experiments": {str(t): dict(zip(CLASSES, values.tolist()))
                        for t, values in zip(thresholds, scores)},
        "selected_thresholds": {c: thresholds[j] for c, j in zip(CLASSES, best)},
        "selected_dice": dict(zip(CLASSES, scores[best, np.arange(3)].tolist())),
        "macro_dice": float(scores[best, np.arange(3)].mean()),
    }
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "atlas_metrics.json").write_text(json.dumps(metrics, indent=2))
    examples = [[] for _ in CLASSES]
    for key, position in position_keys(data, validation):
        size = data.shapes[key][::-1]
        for index, organ in enumerate(CLASSES):
            if len(examples[index]) == 4 or not data.labels[key][organ]:
                continue
            if any(volume(candidate[1]) == volume(key) for candidate in examples[index]):
                continue
            target = data.mask(key, organ).astype(bool)
            if target.sum() < 300:
                continue
            pred = resize(binary[best[index], position, index], size)
            score = float(dice((pred & target).sum(), pred.sum(), target.sum()))
            examples[index].append((score, key, pred, target))
    make_review(data, examples, OUTPUT / "atlas_review.png")
    predictions = {}
    for key, position in position_keys(data, data.parts["test"]):
        height, width = data.shapes[key]
        for index, organ in enumerate(CLASSES):
            mask = resize(binary[best[index], position, index], (width, height))
            predictions[key, organ] = encode(mask)
    temporary = OUTPUT / "atlas_submission.tmp"
    data.save(temporary, predictions)
    temporary.replace(OUTPUT / "submission.csv")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Validated {len(predictions)} prediction rows", flush=True)


if __name__ == "__main__":
    main()
