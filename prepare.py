"""Cache fixed patient splits and normalized 2.5D MRI inputs."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from atlas import resize
from dataset import CLASSES, SIZE, DatasetArchive, patient, volume

CACHE = Path("outputs/cache" if SIZE == 192 else f"outputs/cache_{SIZE}")
BILINEAR = Image.Resampling.BILINEAR


def prepare():
    data = DatasetArchive()
    fit, validation = data.split()
    train_patients = {patient(k) for k in fit}
    assert train_patients.isdisjoint(patient(k) for k in validation)
    for name, keys in (("fit", fit), ("validation", validation), ("test", data.parts["test"])):
        root = CACHE / name
        manifest = root / "keys.json"
        if manifest.exists():
            continue
        root.mkdir(parents=True, exist_ok=True)
        images = np.lib.format.open_memmap(root / "images.npy", mode="w+", dtype=np.uint8,
                                           shape=(len(keys), SIZE, SIZE))
        masks = None if name == "test" else np.lib.format.open_memmap(
            root / "masks.npy", mode="w+", dtype=np.uint8, shape=(len(keys), 3, SIZE, SIZE)
        )
        ceilings = {}
        for i, key in enumerate(keys):
            day = volume(key)
            if day not in ceilings:
                samples = np.concatenate([
                    data.image(k)[::4, ::4].ravel() for k in data.volumes[day][::8]
                ])
                foreground = samples[samples > 0]
                ceilings[day] = max(1, np.quantile(foreground, 0.995)) if foreground.size else 1
            image = data.image(key).astype(np.float32)
            image = resize(image, (SIZE, SIZE), BILINEAR)
            images[i] = np.rint(np.clip(image / ceilings[day], 0, 1) * 255).astype(np.uint8)
            if masks is not None:
                for j, organ in enumerate(CLASSES):
                    masks[i, j] = resize(data.mask(key, organ), (SIZE, SIZE))
            if i % 2000 == 0:
                print(f"{name}: {i}/{len(keys)}", flush=True)
        images.flush()
        if masks is not None:
            masks.flush()
        temporary = root / "keys.tmp"
        temporary.write_text(json.dumps(keys))
        temporary.replace(manifest)


if __name__ == "__main__":
    prepare()
