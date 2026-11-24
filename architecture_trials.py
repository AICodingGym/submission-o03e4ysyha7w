"""Architecture experiments using the established three-slice MRI representation."""
import argparse
import json
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet34

from architecture_clock import awake_seconds
from dataset import SIZE
from unet import OUTPUT, Slices, UNet, batches, jitter_scale, validate

ROOT = OUTPUT / "architecture"
WEIGHTS = ROOT / "resnet34-b627a593.pth"
NAMES = ("resnet34_unet", "resnet34_unetpp")


def up(x, skip):
    return F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)


class Decode(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                         nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True),
                         nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
                         nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True))

    def forward(self, skip, deeper, *same_level):
        x = torch.cat((skip, up(deeper, skip), *same_level), dim=1)
        return super().forward(x)


class ResNetDecoder(nn.Module):
    def __init__(self, nested, pretrained):
        super().__init__()
        self.nested = nested
        self.encoder = resnet34(weights=None)
        if pretrained:
            self.encoder.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
        self.encoder.fc = nn.Identity()
        widths = (64, 64, 128, 256, 512)
        out = (24, 32, 48, 80)
        self.nodes = nn.ModuleDict()
        for depth in range(1, 5):
            for level in range(5 - depth):
                if not nested and level != 4 - depth:
                    continue
                incoming = widths[level] + (widths[level + 1] if depth == 1 else out[level + 1])
                if nested:
                    incoming += (depth - 1) * out[level]
                self.nodes[f"{level}_{depth}"] = Decode(incoming, out[level])
        self.head = nn.Conv2d(out[0], 12, 1)

    def forward(self, images):
        # Every input channel is an MRI slice, with a shared normalization.
        e = self.encoder
        x = e.relu(e.bn1(e.conv1((images - 0.45) / 0.225)))
        features = {(0, 0): x}
        x = e.maxpool(x)
        for i, block in enumerate((e.layer1, e.layer2, e.layer3, e.layer4), 1):
            x = block(x)
            features[i, 0] = x
        for depth in range(1, 5):
            for level in range(5 - depth):
                name = f"{level}_{depth}"
                if name in self.nodes:
                    history = [features[level, h] for h in range(1, depth)] if self.nested else []
                    features[level, depth] = self.nodes[name](
                        features[level, 0], features[level + 1, depth - 1], *history
                    )
        return F.pixel_shuffle(self.head(features[0, 4]), 2)


def make_model(name, pretrained=False):
    if name == "current":
        return UNet(packed=True)
    if name not in NAMES:
        raise ValueError(f"Unknown architecture: {name}")
    return ResNetDecoder(name.endswith("unetpp"), pretrained)


def loss_value(logits, targets, objective="bce"):
    probabilities = logits.float().sigmoid()
    axes = (0, 2, 3)
    hits = (probabilities * targets).sum(axes)
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    if objective == "tversky":
        fp = (probabilities * (1 - targets)).sum(axes)
        fn = ((1 - probabilities) * targets).sum(axes)
        overlap = (hits + 1) / (hits + 0.35 * fp + 0.65 * fn + 1)
    else:
        overlap = (2 * hits + 1) / ((probabilities + targets).sum(axes) + 1)
    return bce + 1 - overlap.mean()


def update(model, images, targets, scaler, divisor, objective):
    with torch.autocast("cuda"):
        loss = loss_value(model(images), targets, objective)
    scaler.scale(loss / divisor).backward()
    return loss.detach()


def profile(name, batch):
    device = torch.device("cuda")
    model = make_model(name, pretrained=name != "current").to(device, memory_format=torch.channels_last)
    n = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), fused=True)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.rand(batch, 3, SIZE, SIZE, device=device).contiguous(memory_format=torch.channels_last)
    y = torch.rand_like(x).round_()
    for i in range(16):
        if i == 4:
            torch.cuda.synchronize()
            start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
        opt.zero_grad(set_to_none=True)
        update(model, x, y, scaler, 1, "bce")
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / 12
    return dict(model=name, parameters=n, batch=batch, seconds_per_batch=seconds,
                estimated_epoch_seconds=math.ceil(25216/batch)*seconds,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)


def train(args):
    torch.manual_seed(102)
    np_rng = np.random.default_rng(102)
    device = torch.device("cuda")
    start = awake_seconds()
    model = make_model(args.model, pretrained=not args.initialize).to(device, memory_format=torch.channels_last)
    if args.initialize:
        model.load_state_dict(torch.load(ROOT / f"{args.initialize}.pt", map_location=device, weights_only=True))
    fit, val = Slices("fit"), Slices("validation")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
    scaler = torch.amp.GradScaler("cuda")
    micro_batches = math.ceil(len(fit.keys) / args.batch)
    steps = math.ceil(micro_batches / args.accumulation)
    max_steps = args.epochs * steps
    warm_steps = steps
    def lr_multiplier(step):
        if step < warm_steps:
            return 0.1 + 0.9 * (step + 1) / warm_steps
        fraction = (step - warm_steps) / max(1, max_steps - warm_steps)
        return 0.05 + 0.95 * (1 + math.cos(math.pi * fraction)) / 2
    schedule = torch.optim.lr_scheduler.LambdaLR(opt, lr_multiplier)
    best, stagnant = -1.0, 0
    completed, reason = 0, "epoch_limit"
    for epoch in range(1, args.epochs + 1):
        if awake_seconds() - start >= args.budget * 60:
            print("Training budget reached", flush=True)
            reason = "time_budget"
            break
        epoch_start = awake_seconds()
        model.train()
        loss_sum = torch.zeros((), device=device)
        opt.zero_grad(set_to_none=True)
        for i, ix in enumerate(batches(len(fit.keys), args.batch, np_rng)):
            x, y = fit.inputs(ix, device), fit.targets(ix, device)
            if np_rng.random() < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            x, y = jitter_scale(x, y)
            x = (x * np_rng.uniform(0.85, 1.15)).clamp_(0, 1)
            group = (i // args.accumulation) * args.accumulation
            group_samples = min(len(fit.keys) - group * args.batch, args.batch * args.accumulation)
            loss_sum += len(ix) * update(model, x, y, scaler, group_samples / len(ix), args.objective)
            if (i + 1) % args.accumulation == 0 or i + 1 == micro_batches:
                scale = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if scaler.get_scale() >= scale:
                    schedule.step()
            if (i + 1) % 500 == 0:
                print(f"{args.run} epoch {epoch} batch {i+1}/{micro_batches}", flush=True)
        train_seconds = awake_seconds() - epoch_start
        row = validate(model, val, device, args.batch)
        row.update(epoch=epoch, loss=float(loss_sum / len(fit.keys)), train_seconds=train_seconds,
                   validation_seconds=awake_seconds() - epoch_start - train_seconds,
                   lr=opt.param_groups[0]["lr"])
        with (ROOT / f"{args.run}_epochs.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(row, flush=True)
        completed = epoch
        if row["macro_dice"] > best:
            best, stagnant = row["macro_dice"], 0
            temp = ROOT / f"{args.run}.tmp"
            torch.save(model.state_dict(), temp)
            temp.replace(ROOT / f"{args.run}.pt")
        else:
            stagnant += 1
        if stagnant >= 10:
            print("Early stopped after 10 epochs without higher validation Dice", flush=True)
            reason = "patience"
            break
    report = dict(run=args.run, reason=reason, completed_epochs=completed,
                  best_validation_dice=best, elapsed_seconds=awake_seconds() - start)
    temporary = ROOT / f"{args.run}_status.tmp"
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(ROOT / f"{args.run}_status.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("profile", "train"))
    parser.add_argument("--model", choices=(*NAMES, "current"))
    parser.add_argument("--run")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=42)
    parser.add_argument("--budget", type=float, default=200)
    parser.add_argument("--lr", type=float, default=0.00035)
    parser.add_argument("--initialize")
    parser.add_argument("--objective", choices=("bce", "tversky"), default="bce")
    args = parser.parse_args()
    if SIZE != 256 or args.batch < 1 or args.accumulation < 1:
        parser.error("Use the 256-pixel cache and positive batch and accumulation sizes")
    ROOT.mkdir(exist_ok=True, parents=True)
    if args.action == "profile":
        results = [profile(name, args.batch) for name in ("current", *NAMES)]
        (ROOT / "profiles.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2), flush=True)
    else:
        if args.model not in NAMES or not args.run or not args.run.replace("_", "").isalnum():
            parser.error("Training needs a new architecture and an alphanumeric run name")
        if (ROOT / f"{args.run}_epochs.jsonl").exists():
            parser.error("Run name already exists")
        (ROOT / f"{args.run}_config.json").write_text(json.dumps(vars(args), indent=2))
        train(args)


if __name__ == "__main__":
    main()
