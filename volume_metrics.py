"""Volume postprocessing and a validation-only Hausdorff diagnostic."""

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def remove_small_components(probabilities, minimum_voxels=64):
    cleaned = probabilities.copy()
    for organ in np.moveaxis(cleaned, 1, 0):
        labels, _ = ndimage.label(organ >= 0.5)
        small = np.bincount(labels.ravel()) < minimum_voxels
        small[0] = False
        organ[small[labels]] = 0
    return cleaned


def normalized_hausdorff(target, prediction):
    """Measure maximal surface distance after scaling the scan axes to [0, 1]."""
    def surface(mask):
        boundary = mask & ~ndimage.binary_erosion(mask)
        return np.argwhere(boundary) / (np.maximum(np.asarray(mask.shape) - 1, 1) * np.sqrt(3))

    a, b = surface(target), surface(prediction)
    if not len(a) or not len(b):
        return float(bool(len(a)) != bool(len(b)))
    return max(cKDTree(a).query(b)[0].max(), cKDTree(b).query(a)[0].max())
