"""Dataset inspection, patient split, and checked competition CSV export."""
import csv
import io
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

CLASSES = ("large_bowel", "small_bowel", "stomach")
ARCHIVE = Path("data/uw-madison-gi-tract-image-segmentation.zip")
SIZE = int(os.environ.get("GI_SEGMENTATION_SIZE", "192"))


def rows(archive, name):
    with archive.open(name) as source:
        return list(csv.DictReader(io.TextIOWrapper(source, encoding="utf-8")))


def volume(key):
    return key.rsplit("_slice_", 1)[0]


def patient(key):
    return key.split("_", 1)[0]


def decode(value, shape, order="C"):
    # Row-major decoding aligns the provided masks with rectangular PNG scans.
    flat = np.zeros(np.prod(shape), dtype=np.uint8)
    runs = np.array(value.split(), dtype=np.int64).reshape(-1, 2)
    end = 0
    for start, length in runs:
        if start < 1 or length < 1 or start - 1 < end or start - 1 + length > flat.size:
            raise ValueError(f"Invalid RLE run {(start, length)} for {shape}")
        end = start - 1 + length
        flat[start - 1:end] = 1
    return flat.reshape(shape, order=order)


def encode(mask, order="C"):
    pixels = mask.ravel(order=order).astype(np.int8)
    edges = np.flatnonzero(np.diff(pixels, prepend=0, append=0))
    edges[1::2] -= edges[::2]
    edges[::2] += 1
    return " ".join(map(str, edges))


class DatasetArchive:
    def __init__(self):
        self.archive = ZipFile(ARCHIVE)
        self.paths = {}
        self.shapes = {}
        self.parts = defaultdict(list)
        self.volumes = defaultdict(list)
        for path in sorted(self.archive.namelist()):
            match = re.fullmatch(
                r"(train|test)/[^/]+/([^/]+)/scans/(slice_\d+)_(\d+)_(\d+)_[\d.]+_[\d.]+\.png",
                path,
            )
            if match:
                part, day, number, width, height = match.groups()
                key = f"{day}_{number}"
                self.paths[key] = path
                self.shapes[key] = (int(height), int(width))
                self.parts[part].append(key)
                self.volumes[day].append(key)
        self.labels = defaultdict(dict)
        for row in rows(self.archive, "train.csv"):
            self.labels[row["id"]][row["class"]] = row["segmentation"]
        self.sample = rows(self.archive, "sample_submission.csv")
        test = rows(self.archive, "test.csv")
        assert [(r["id"], r["class"]) for r in test] == [
            (r["id"], r["class"]) for r in self.sample
        ]
        assert len(test) == len(self.parts["test"]) * len(CLASSES)
        assert {(r["id"], r["class"]) for r in test} == {
            (k, c) for k in self.parts["test"] for c in CLASSES
        }
        assert set(self.labels) == set(self.parts["train"])
        assert all(set(labels) == set(CLASSES) for labels in self.labels.values())

    def split(self):
        patients = sorted({patient(k) for k in self.parts["train"]})
        random.Random(102).shuffle(patients)
        held_out = set(patients[:round(len(patients) * 0.2)])
        return tuple(
            [k for k in self.parts["train"] if (patient(k) in held_out) == validation]
            for validation in (False, True)
        )

    def image(self, key):
        with Image.open(io.BytesIO(self.archive.read(self.paths[key]))) as image:
            values = np.asarray(image)
        if values.shape != self.shapes[key]:
            raise ValueError(f"Unexpected PNG dimensions for {key}: {values.shape}")
        return values

    def mask(self, key, organ):
        return decode(self.labels[key][organ], self.shapes[key])

    def save(self, path, predictions):
        expected = {(r["id"], r["class"]) for r in self.sample}
        if set(predictions) != expected:
            raise ValueError("Prediction keys differ from the submission template")
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(("id", "class", "predicted"))
            for row in self.sample:
                key, organ = row["id"], row["class"]
                rle = predictions[key, organ]
                decode(rle, self.shapes[key])
                writer.writerow((key, organ, rle))

    def inspect(self):
        for part, keys in self.parts.items():
            print(part, "slices", len(keys), "volumes", len({volume(k) for k in keys}),
                  "patients", len({patient(k) for k in keys}))
            print("dimensions (height, width)", Counter(self.shapes[k] for k in keys))
        print("train/test patient overlap", len(
            {patient(k) for k in self.parts["train"]} & {patient(k) for k in self.parts["test"]}
        ))
        print("positive annotations", {c: sum(bool(m[c]) for m in self.labels.values()) for c in CLASSES})
        for name, keys in zip(("fit", "validation"), self.split()):
            print(name, len(keys), "slices", len({patient(k) for k in keys}), "patients")
        print("sample rows", len(self.sample))
        key = self.parts["train"][0]
        print("PNG dtype and intensity range", self.image(key).dtype,
              self.image(key).min(), self.image(key).max())


if __name__ == "__main__":
    DatasetArchive().inspect()
