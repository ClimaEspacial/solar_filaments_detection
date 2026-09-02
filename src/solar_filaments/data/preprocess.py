"""Image-level preprocessing primitives - a leaf module (no dependency on
coco.py or augment.py) so coco.py::load_image can depend on it without a
circular import.

Solar full-disk H-alpha images have a real, substantial limb-darkening
gradient: mean intensity in this dataset falls from ~140 near disk center to
~76 near the limb (confirmed empirically) - nearly 2x. Left uncorrected, the
same filament looks very different in absolute intensity depending on where
it sits on the disk, purely from this large-scale effect. `preprocess_image`
flattens it (divide by the local radial-average intensity) and then applies
CLAHE for local contrast enhancement, so filament-vs-background contrast is
more uniform across the whole disk.
"""
from typing import Tuple

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d


def estimate_disk_mask(img: np.ndarray) -> np.ndarray:
    """Boolean mask of the on-disk region (bright) vs. off-disk background
    (near-black), via Otsu thresholding + largest connected component."""
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats((thresh > 0).astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return thresh > 0
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def flatten_limb_darkening(img: np.ndarray, disk_mask: np.ndarray) -> np.ndarray:
    """Divide each on-disk pixel by its expected radial-average intensity
    (smoothed), rescaled back to the central region's original brightness -
    flattens the limb-darkening gradient without needing a fitted physical
    model, since the radial profile is estimated directly from this image."""
    H, W = img.shape
    ys, xs = np.where(disk_mask)
    if len(ys) == 0:
        return img.copy()
    cy, cx = ys.mean(), xs.mean()

    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.int32)
    rmax = int(r[disk_mask].max())

    profile = np.zeros(rmax + 1, dtype=np.float64)
    counts = np.zeros(rmax + 1, dtype=np.float64)
    np.add.at(profile, r[disk_mask], img[disk_mask].astype(np.float64))
    np.add.at(counts, r[disk_mask], 1)
    counts[counts == 0] = 1
    profile /= counts
    profile = uniform_filter1d(profile, size=21)
    profile[profile < 1] = 1

    r_clipped = np.clip(r, 0, rmax)
    expected = profile[r_clipped]
    target_mean = profile[:max(1, int(rmax * 0.7))].mean()  # roughly the flat central brightness

    flattened = img.astype(np.float32) / expected * target_mean
    flattened = np.clip(flattened, 0, 255).astype(np.uint8)
    flattened[~disk_mask] = 0
    return flattened


def apply_clahe(img: np.ndarray, disk_mask: np.ndarray, clip_limit: float = 2.0,
                 tile_grid: Tuple[int, int] = (16, 16)) -> np.ndarray:
    """Local contrast enhancement, restricted to the on-disk region."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    enhanced = clahe.apply(img)
    enhanced[~disk_mask] = 0
    return enhanced


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Limb-darkening flattening + CLAHE, composed. See module docstring."""
    disk_mask = estimate_disk_mask(img)
    flattened = flatten_limb_darkening(img, disk_mask)
    return apply_clahe(flattened, disk_mask)
