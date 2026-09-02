"""Classical (non-learned) filament detector, reimplemented from Jean Carlo
Santos's "Filament detection using Python" presentation, itself based on
Shih & Kowalski (2003):

1. Local-median thresholding (asymmetric cutoff) to separate dark filament
   candidates from the chromosphere background, robust to limb darkening
   since the threshold is computed locally rather than globally.
2. A 3x3 closing to clean small dark noise specks.
3. Negative (filaments become the bright foreground).
4. Disjunctive multi-directional closing (8 orientations, 11x11 linear
   structuring elements) to bridge gaps along each filament's own local
   orientation, whatever that orientation is - a classical, rotation-covering
   analogue to the OBB detector's learned angle.
5. A final 3x3 closing to clean spurious directional-closing artifacts.
6. Connected-component labeling (8-connectivity) - each component is already
   a separate instance, no NMS needed.
7. Shape-based filtering: drop components that are too small, touch the
   image border, or are too round (active regions/sunspots, not filaments).

No training, no GPU. Useful as an independent baseline PQ and, later, as a
diverse candidate source for ensembling with the learned pipeline.
"""
from typing import List, Tuple

import cv2
import numpy as np
from scipy.ndimage import median_filter
from skimage.measure import label, regionprops
from skimage.morphology import closing

from solar_filaments.data.augment import estimate_disk_mask


def local_median_mask(img: np.ndarray, size: int = 20, cutoff_m: float = 0.96) -> np.ndarray:
    """1 = brighter than `cutoff_m` times the local median (normal
    chromosphere), 0 = darker (filament candidate, or noise)."""
    filtered = median_filter(img, size=size)
    mask = np.zeros_like(img, dtype=np.uint8)
    mask[img.astype(np.float32) > cutoff_m * filtered.astype(np.float32)] = 1
    return mask


def _directional_footprints(size: int = 11) -> List[np.ndarray]:
    """8 linear structuring elements at evenly-spaced orientations across
    0-157.5 degrees (a line has no direction, so 180 degrees covers every
    orientation) - a discrete-line rasterization of each angle through an
    size x size grid, approximating the fixed set of hand-drawn footprints
    in the original presentation."""
    center = size // 2
    footprints = []
    for angle_deg in np.arange(0, 180, 180 / 8):
        theta = np.radians(angle_deg)
        fp = np.zeros((size, size), dtype=np.uint8)
        for t in range(-center, center + 1):
            x = int(round(center + t * np.cos(theta)))
            y = int(round(center + t * np.sin(theta)))
            if 0 <= x < size and 0 <= y < size:
                fp[y, x] = 1
        footprints.append(fp)
    return footprints


def directional_closing(mask: np.ndarray, size: int = 11) -> np.ndarray:
    """Disjunctive (max-combined) closing with 8 rotated linear structuring
    elements - bridges gaps along a filament's own orientation regardless of
    its angle on the disk, without needing to know that angle in advance."""
    out = np.zeros_like(mask)
    for fp in _directional_footprints(size):
        out = np.maximum(out, closing(mask, footprint=fp))
    return out


def detect_filaments_classical(img: np.ndarray, working_size: int = 512, median_size: int = 20,
                                cutoff_m: float = 0.96, directional_size: int = 11, min_area: int = 100,
                                min_eccentricity: float = 0.9, border_margin: int = 5
                                ) -> Tuple[np.ndarray, List[float]]:
    """Full classical pipeline -> (label_map, scores), matching the return
    shape of inference.predict_instances/predict_instances_obb. The method
    has no native confidence score, so every accepted instance gets 1.0.

    The original presentation's code operates on an `image_resized` at every
    step, and its parameters (a 20px median window, 11px closings) are sized
    for that resolution, not our native 2048x2048 - so this downsamples to
    `working_size` first (matching what the reference implementation
    apparently did) and upsamples the final label map back to the input's
    original resolution with nearest-neighbor (label-preserving) resizing.

    "Touching the border" means the *solar disk's own limb*, not the image's
    rectangular edges - the disk is a circle inset within the frame, and the
    limb's brightness transition is itself a strong local-median outlier that
    would otherwise show up as one giant spurious "filament" tracing the
    whole rim (reuses the same disk-mask estimate built for copy-paste
    augmentation, see data/augment.py)."""
    H, W = img.shape
    scale = working_size / max(H, W)
    small = cv2.resize(img, (round(W * scale), round(H * scale)), interpolation=cv2.INTER_AREA)

    mask = local_median_mask(small, size=median_size, cutoff_m=cutoff_m)
    mask = closing(mask, footprint=np.ones((3, 3), dtype=np.uint8))
    mask = 1 - mask
    mask = directional_closing(mask, size=directional_size)
    mask = closing(mask, footprint=np.ones((3, 3), dtype=np.uint8))

    disk_mask = estimate_disk_mask(small)
    erode_size = 2 * border_margin + 1
    interior_disk = cv2.erode(disk_mask.astype(np.uint8), np.ones((erode_size, erode_size), np.uint8))

    labeled = label(mask, connectivity=2)
    small_label_map = np.zeros(small.shape, dtype=np.uint16)
    scores: List[float] = []
    next_id = 1
    for region in regionprops(labeled):
        if region.area < min_area:
            continue
        region_mask = labeled == region.label
        if not np.all(interior_disk[region_mask]):
            continue  # touches or is outside the disk's own limb
        if region.eccentricity < min_eccentricity:
            continue
        small_label_map[region_mask] = next_id
        scores.append(1.0)
        next_id += 1

    label_map = cv2.resize(small_label_map, (W, H), interpolation=cv2.INTER_NEAREST)
    return label_map, scores
