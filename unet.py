"""Train and evaluate a residual 2.5D U-Net on a fixed patient split."""
import argparse
import json
import os
import time
from itertools import groupby
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from atlas import BINS, OUTPUT, dice, make_review, position_keys, resize
from dataset import CLASSES, SIZE, DatasetArchive, encode, patient, volume
from prepare import CACHE
from volume_metrics import normalized_hausdorff, remove_small_components


CONTEXT = int(os.environ.get("GI_SEGMENTATION_CONTEXT", "3"))
if CONTEXT not in (3, 5):
    raise ValueError("GI_SEGMENTATION_CONTEXT must be 3 or 5")


class Slices:
    def __init__(self, name, root=CACHE, context=CONTEXT):
        root = root / name
        self.keys = json.loads((root / "keys.json").read_text())
        self.images = np.load(root / "images.npy", mmap_mode="r")
        self.masks = None if name == "test" else np.load(root / "masks.npy", mmap_mode="r")
        index = {key: i for i, key in enumerate(self.keys)}
        self.context = np.asarray([
            [index.get(f"{volume(key)}_slice_{int(key.rsplit('_', 1)[1]) + offset:04d}", i)
             for offset in range(-(context // 2), context // 2 + 1)]
            for i, key in enumerate(self.keys)
        ])

    def inputs(self, indices, device):
        batch = torch.from_numpy(self.images[self.context[indices]]).to(device)
        return batch.float().div_(255).contiguous(memory_format=torch.channels_last)

    def targets(self, indices, device):
        return torch.from_numpy(self.masks[indices]).to(device).float()


class Residual(nn.Module):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outgoing), nn.SiLU(inplace=True),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.BatchNorm2d(outgoing),
        )
        self.skip = nn.Identity() if incoming == outgoing and stride == 1 else nn.Sequential(
            nn.Conv2d(incoming, outgoing, 1, stride, bias=False), nn.BatchNorm2d(outgoing)
        )

    def forward(self, x):
        return F.silu(self.body(x) + self.skip(x), inplace=True)


class UNet(nn.Module):
    def __init__(self, packed=False, context=CONTEXT, deep_encoder=False):
        super().__init__()
        self.packed = packed
        widths = (24, 40, 64, 96, 160)
        self.stem = Residual(context * (4 if packed else 1), widths[0])
        self.encoder = nn.ModuleList(
            nn.Sequential(Residual(a, b, 2), Residual(b, b))
            for a, b in zip(widths, widths[1:])
        )
        self.encoder_extra = nn.Sequential()
        if deep_encoder:
            block = Residual(widths[-1], widths[-1])
            nn.init.zeros_(block.body[-1].weight)
            self.encoder_extra = nn.Sequential(block)
        self.decoder = nn.ModuleList(
            Residual(a + b, b) for a, b in zip(widths[:0:-1], widths[-2::-1])
        )
        self.head = nn.Conv2d(widths[0], 12 if packed else 3, 1)

    def forward(self, x):
        if self.packed:
            x = F.pixel_unshuffle(x, 2)
        features = [self.stem(x)]
        for block in self.encoder:
            features.append(block(features[-1]))
        x = features.pop()
        for block in self.encoder_extra:
            x = x + block.body(x)
        for block, skip in zip(self.decoder, reversed(features)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat((x, skip), dim=1))
        logits = self.head(x)
        return F.pixel_shuffle(logits, 2) if self.packed else logits


def batches(size, batch, rng=None):
    indices = np.arange(size) if rng is None else rng.permutation(size)
    return (indices[i:i + batch] for i in range(0, size, batch))


@torch.inference_mode()
def infer(model, images):
    models = model if isinstance(model, nn.ModuleList) else (model,)
    with torch.autocast(images.device.type, enabled=images.is_cuda):
        probabilities = [member(images).float().sigmoid() for member in models]
    return probabilities[0] if len(probabilities) == 1 else torch.stack(probabilities).mean(0)


def metrics_row(scores):
    return {"macro_dice": float(np.mean(scores)), "dice": dict(zip(CLASSES, scores.tolist()))}


@torch.inference_mode()
def validate(model, data, device, batch):
    model.eval()
    totals = torch.zeros(3, 3, dtype=torch.float64, device=device)
    loss_sum = torch.zeros((), device=device)
    for indices in batches(len(data.keys), batch):
        inputs = data.inputs(indices, device)
        target = data.targets(indices, device)
        with torch.autocast(device.type, enabled=inputs.is_cuda):
            logits = model(inputs)
            probs = logits.float().sigmoid()
            overlap = (probs * target).sum((0, 2, 3))
            soft_dice = (2 * overlap + 1) / ((probs + target).sum((0, 2, 3)) + 1)
            loss_sum += len(indices) * (F.binary_cross_entropy_with_logits(logits, target)
                                        + 1 - soft_dice.mean())
        pred = probs >= 0.5
        target = target.bool()
        totals += torch.stack([v.sum((0, 2, 3)) for v in (pred & target, pred, target)])
    return dict(metrics_row(dice(*totals.cpu().numpy())),
                validation_loss=float(loss_sum / len(data.keys)))


def jitter_scale(images, masks, translation=0.0):
    scale = torch.empty(len(images), device=images.device).uniform_(0.85, 1.15)
    transform = torch.zeros((len(images), 2, 3), device=images.device)
    transform[:, 0, 0] = scale
    transform[:, 1, 1] = scale
    if translation:
        transform[:, :, 2].uniform_(-translation, translation)
    grid = F.affine_grid(transform, images.shape, align_corners=False)
    return (F.grid_sample(images, grid, mode="bilinear", align_corners=False),
            F.grid_sample(masks, grid, mode="nearest", align_corners=False))


def train(model, device, args, prefix):
    fitting, validation = Slices("fit"), Slices("validation")
    rng = np.random.default_rng(102)
    learning_rate = 0.0002 if args.fine_tune_from else 0.001
    if args.lr is not None:
        learning_rate = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, fused=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.00003)
    if args.schedule == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=0, threshold=0.002, threshold_mode="abs"
        )
    elif args.schedule == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    best = -1
    weights = torch.tensor([1.0, args.small_bowel_loss_weight, 1.0], device=device)
    weights /= weights.sum()
    for epoch in range(1, args.epochs + 1):
        then = time.perf_counter()
        model.train()
        running_loss = torch.zeros((), device=device)
        for i, indices in enumerate(batches(len(fitting.keys), args.batch, rng), start=1):
            images = fitting.inputs(indices, device)
            masks = fitting.targets(indices, device)
            if rng.random() < 0.5:
                images, masks = images.flip(-1), masks.flip(-1)
            if args.scale_jitter:
                images, masks = jitter_scale(images, masks, args.translation_jitter)
            images = (images * rng.uniform(0.85, 1.15)).clamp_(0, 1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(images)
                bce = F.binary_cross_entropy_with_logits(logits, masks)
                probabilities = logits.float().sigmoid()
                axes = (0, 2, 3)
                overlap = (probabilities * masks).sum(axes)
                soft_dice = (2 * overlap + 1) / ((probabilities + masks).sum(axes) + 1)
                dice_score = soft_dice.mean() if args.small_bowel_loss_weight == 1 else (soft_dice * weights).sum()
                if args.objective == "focal":
                    per_pixel = F.binary_cross_entropy_with_logits(logits, masks, reduction="none")
                    wrong = torch.where(masks.bool(), 1 - probabilities, probabilities)
                    bce = (wrong.square() * per_pixel).mean()
                loss = bce + 1 - dice_score
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.detach()
            if i == 50 or i % 300 == 0:
                print(f"Epoch {epoch}, batch {i}, {time.perf_counter() - then:.1f}s", flush=True)
        elapsed = time.perf_counter() - then
        result = validate(model, validation, device, args.batch)
        result.update(epoch=epoch, lr=optimizer.param_groups[0]["lr"], loss=float(running_loss / i), train_seconds=elapsed,
                      validation_seconds=time.perf_counter() - then - elapsed)
        with (OUTPUT / f"{prefix}_epochs.jsonl").open("a") as log:
            log.write(json.dumps(result) + "\n")
        print(result, flush=True)
        if result["macro_dice"] > best:
            best = result["macro_dice"]
            temporary = OUTPUT / f"{prefix}.tmp"
            torch.save(model.state_dict(), temporary)
            temporary.replace(OUTPUT / f"{prefix}.pt")
            if epoch == 1 and args.fine_tune_from is None and not args.train_only:
                assess(model, device, args.batch, prefix, compare=False)
        if args.schedule == "plateau":
            scheduler.step(result["macro_dice"])
        else:
            scheduler.step()


def scan_probabilities(model, data, device, batch, flip):
    grouped = groupby(enumerate(data.keys), key=lambda item: volume(item[1]))
    for _, scan in grouped:
        scan = list(scan)
        indices, keys = zip(*scan)
        indices = np.asarray(indices)
        chunks = []
        flipped_chunks = []
        for batch_indices in batches(len(keys), batch):
            inputs = data.inputs(indices[batch_indices], device)
            plain = infer(model, inputs)
            chunks.append(plain.cpu().numpy())
            if flip:
                augmented = infer(model, inputs.flip(-1)).flip(-1)
                flipped_chunks.append(((plain + augmented) * 0.5).cpu().numpy())
        yield keys, np.concatenate(chunks), np.concatenate(flipped_chunks) if flip else None


def smooth_slices(probabilities):
    previous = np.concatenate((probabilities[:1], probabilities[:-1]))
    following = np.concatenate((probabilities[1:], probabilities[-1:]))
    return 0.7 * probabilities + 0.15 * (previous + following)


def fit_axial_support(archive):
    fit, _ = archive.split()
    counts = np.zeros((BINS, len(CLASSES)), dtype=np.int64)
    for key, position in position_keys(archive, fit):
        counts[position] += [bool(archive.labels[key][organ]) for organ in CLASSES]
    cumulative = counts.cumsum(axis=0) / counts.sum(axis=0)
    bounds = np.array([
        (np.searchsorted(organ, 0.005) - 2, np.searchsorted(organ, 0.995) + 2)
        for organ in cumulative.T
    ])
    positions = np.arange(BINS)[:, None]
    return (positions >= bounds[:, 0]) & (positions <= bounds[:, 1])


def apply_axial_support(archive, keys, probabilities, support):
    positions = [position for _, position in position_keys(archive, keys)]
    return probabilities * support[positions, :, None, None]


def uses_flip(method):
    return method == "horizontal_flip" or method.startswith("flip_")


def refine(method, archive, keys, plain, flipped, support):
    probabilities = flipped if uses_flip(method) else plain
    if method.endswith("smoothing"):
        return smooth_slices(probabilities)
    if method.endswith("support"):
        return apply_axial_support(archive, keys, probabilities, support)
    if method == "flip_small_components":
        return remove_small_components(probabilities)
    return probabilities


def native_masks(probabilities, dimensions, threshold):
    thresholds = np.broadcast_to(threshold, (len(CLASSES),))
    return tuple(resize(channel, dimensions, Image.Resampling.BILINEAR) >= cutoff
                 for channel, cutoff in zip(probabilities, thresholds))


def track_example(examples, key, pred, target, score):
    existing = next((item for item in examples if patient(item[1]) == patient(key)), None)
    if existing is not None:
        if score >= existing[0]:
            return
        examples.remove(existing)
    examples.append((score, key, pred, target))
    examples.sort(key=lambda item: (item[0], item[1]))
    del examples[4:]


def assess(model, device, batch, prefix, compare=True):
    start = time.perf_counter()
    model.eval()
    archive = DatasetArchive()
    validation = Slices("validation")
    configurations = [("plain", 0.5)]
    if compare:
        support = fit_axial_support(archive)
        configurations.extend([
            ("horizontal_flip", 0.5), ("axial_smoothing", 0.5), ("axial_support", 0.5),
            ("flip_axial_smoothing", 0.5), ("flip_axial_support", 0.5),
            ("flip_small_components", 0.5),
        ])
        configurations.extend((method, threshold)
                              for method in ("plain", "horizontal_flip", "flip_axial_smoothing")
                              for threshold in (0.3, 0.35, 0.4, 0.45, 0.55, 0.6))
    else:
        support = None
    totals = np.zeros((len(configurations), 3, 3), dtype=np.int64)
    worst = [[[] for _ in CLASSES] for _ in configurations]
    geometry = {method: [] for method in
                ("horizontal_flip", "flip_axial_smoothing", "flip_small_components")} if compare else {}
    next_scan = 0
    for keys, plain, flipped in scan_probabilities(model, validation, device, batch, compare):
        variants = {
            method: refine(method, archive, keys, plain, flipped, support)
            for method in dict.fromkeys(method for method, _ in configurations)
        }
        if geometry:
            scan_masks = validation.masks[next_scan:next_scan + len(keys), :, ::2, ::2].astype(bool)
            for method, distances in geometry.items():
                predicted = variants[method][:, :, ::2, ::2] >= 0.5
                distances.append([
                    normalized_hausdorff(scan_masks[:, i], predicted[:, i])
                    for i in range(len(CLASSES))
                ])
            next_scan += len(keys)
        for offset, key in enumerate(keys):
            dimensions = archive.shapes[key][::-1]
            for organ_index, organ in enumerate(CLASSES):
                target = archive.mask(key, organ).astype(bool)
                actual = np.count_nonzero(target)
                native = {name: resize(p[offset, organ_index], dimensions, Image.Resampling.BILINEAR)
                          for name, p in variants.items()}
                for j, (method, threshold) in enumerate(configurations):
                    pred = native[method] >= threshold
                    hits = np.count_nonzero(pred & target)
                    predicted = np.count_nonzero(pred)
                    totals[j, :, organ_index] += hits, predicted, actual
                    if actual >= 150:
                        score = float(dice(hits, predicted, actual))
                        track_example(worst[j][organ_index], key, pred, target, score)
    results = [
        dict(metrics_row(dice(*counts)), method=method, threshold=threshold,
             predicted_pixels=dict(zip(CLASSES, counts[1].tolist())),
             labeled_pixels=dict(zip(CLASSES, counts[2].tolist())))
        for counts, (method, threshold) in zip(totals, configurations)
    ]
    if compare:
        for method in ("plain", "horizontal_flip", "flip_axial_smoothing"):
            choices = [i for i, result in enumerate(results) if result["method"] == method]
            by_organ = [max(choices, key=lambda i: results[i]["dice"][organ]) for organ in CLASSES]
            selected_rows = [results[i] for i in by_organ]
            scores = np.array([row["dice"][organ] for organ, row in zip(CLASSES, selected_rows)])
            results.append(dict(metrics_row(scores), method=method,
                                threshold=[row["threshold"] for row in selected_rows],
                                predicted_pixels={organ: row["predicted_pixels"][organ]
                                                  for organ, row in zip(CLASSES, selected_rows)},
                                labeled_pixels=selected_rows[0]["labeled_pixels"]))
            worst.append([worst[i][organ_index] for organ_index, i in enumerate(by_organ)])
    winner = max(range(len(results)), key=lambda i: results[i]["macro_dice"])
    best = results[winner]
    make_review(archive, worst[winner], OUTPUT / f"{prefix}_review.png")
    report = {
        "experiments": results,
        "selected": best,
        "mean_normalized_hausdorff_at_0_5": {
            method: dict(zip(CLASSES, np.mean(scans, axis=0).tolist()))
            for method, scans in geometry.items()
        },
        "validation_seconds": time.perf_counter() - start,
    }
    then = time.perf_counter()
    export(model, archive, device, batch, best["method"], best["threshold"], prefix)
    report["export_seconds"] = time.perf_counter() - then
    (OUTPUT / f"{prefix}_validation.json").write_text(json.dumps(report, indent=2))
    print(report["selected"], flush=True)


def export(model, archive, device, batch, method, threshold, prefix):
    model.eval()
    data = Slices("test")
    predictions = {}
    support = fit_axial_support(archive) if method.endswith("support") else None
    for keys, plain, flipped in scan_probabilities(model, data, device, batch, uses_flip(method)):
        probabilities = refine(method, archive, keys, plain, flipped, support)
        for key, channels in zip(keys, probabilities):
            for organ, mask in zip(CLASSES, native_masks(channels, archive.shapes[key][::-1], threshold)):
                predictions[key, organ] = encode(mask)
    temporary = OUTPUT / f"{prefix}_submission.tmp"
    archive.save(temporary, predictions)
    temporary.replace(OUTPUT / f"{prefix}_submission.csv")
    print(f"Validated {len(predictions)} U-Net prediction rows", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--schedule", choices=("cosine", "plateau", "step"), default="cosine")
    parser.add_argument("--objective", choices=("bce", "focal"), default="bce")
    parser.add_argument("--deep-encoder", action="store_true")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--packed", action="store_true",
                        help="Process 2x2 input neighborhoods as channels at half resolution")
    parser.add_argument("--fine-tune-from", type=Path)
    parser.add_argument("--scale-jitter", action="store_true")
    parser.add_argument("--small-bowel-loss-weight", type=float, default=1.0)
    parser.add_argument("--translation-jitter", type=float, default=0.0)
    parser.add_argument("--ensemble", type=Path, nargs="+", metavar="WEIGHTS",
                        help="Average probabilities from multiple checkpoints during evaluation")
    args = parser.parse_args()
    if not 0 <= args.translation_jitter <= 1 or (args.translation_jitter and not args.scale_jitter):
        parser.error("translation jitter requires scale jitter and a value from 0 to 1")
    if not np.isfinite(args.small_bowel_loss_weight) or args.small_bowel_loss_weight <= 0:
        parser.error("small bowel loss weight must be positive and finite")
    if args.batch < 1 or args.epochs < 1:
        parser.error("batch size and epochs must be positive")
    if args.ensemble and (not args.evaluate or args.fine_tune_from or len(args.ensemble) < 2):
        parser.error("ensemble evaluation needs --evaluate and at least two checkpoints")
    prefix = "packed" if args.packed else "unet"
    if SIZE != 192:
        prefix += str(SIZE)
    if CONTEXT != 3:
        prefix += f"_context{CONTEXT}"
    if args.fine_tune_from:
        prefix += "_scale_jitter" if args.scale_jitter else "_control"
    elif args.scale_jitter:
        prefix += "_scale_jitter"
    if args.small_bowel_loss_weight != 1:
        prefix += f"_bowel_weight{args.small_bowel_loss_weight:g}"
    if args.translation_jitter:
        prefix += f"_translation{args.translation_jitter:g}"
    if args.run_name:
        if not args.run_name.replace("_", "").isalnum():
            parser.error("run name must contain only letters, digits, and underscores")
        prefix = args.run_name
    if args.lr is not None and (not np.isfinite(args.lr) or args.lr <= 0):
        parser.error("learning rate must be positive and finite")
    if args.train_only and args.evaluate:
        parser.error("--train-only and --evaluate are mutually exclusive")
    torch.manual_seed(102)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = True
    print(device, torch.cuda.get_device_name() if device.type == "cuda" else "", flush=True)
    if args.ensemble:
        ensemble = nn.ModuleList()
        for path in args.ensemble:
            member = UNet(packed=args.packed, deep_encoder=args.deep_encoder).to(device, memory_format=torch.channels_last)
            member.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            ensemble.append(member)
        assess(ensemble, device, args.batch, f"{prefix}_ensemble")
        return
    model = UNet(packed=args.packed, deep_encoder=args.deep_encoder).to(device, memory_format=torch.channels_last)
    if not args.evaluate:
        if args.fine_tune_from:
            weights = torch.load(args.fine_tune_from, map_location=device, weights_only=True)
            destination = model.state_dict()
            for name in ("stem.body.0.weight", "stem.skip.0.weight"):
                extra = destination[name].shape[1] - weights[name].shape[1]
                if extra > 0:
                    padded = torch.zeros_like(destination[name])
                    padded[:, extra // 2:extra // 2 + weights[name].shape[1]] = weights[name]
                    weights[name] = padded
            missing, unexpected = model.load_state_dict(weights, strict=not args.deep_encoder)
            if unexpected or any(not key.startswith("encoder_extra.") for key in missing):
                raise ValueError(f"Incompatible checkpoint: missing={missing}, extra={unexpected}")
        train(model, device, args, prefix)
        if args.train_only:
            return
    model.load_state_dict(torch.load(OUTPUT / f"{prefix}.pt", map_location=device, weights_only=True))
    assess(model, device, args.batch, prefix)


if __name__ == "__main__":
    main()
