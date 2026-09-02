"""Spine-oriented crop geometry.

Filaments are long, thin structures at arbitrary angles on the solar disk.
An axis-aligned bbox crop wastes resolution on empty corner margin for a
diagonal filament, and gives the U-Net a different-looking input depending on
orientation even for otherwise-identical filaments. Rotating the crop so the
filament's spine (its long axis) lies horizontal before letterboxing gives a
canonical, maximally-zoomed view regardless of on-disk orientation.

At train time the GT `spine` polyline gives the angle directly. At inference
time there's no GT spine, so `predict_instances` does a cheap first pass with
the existing axis-aligned crop to get a rough mask, estimates the angle from
that mask's own principal axis, then re-crops+re-segments using this module.
"""
from typing import Tuple

import cv2
import numpy as np
from ultralytics.utils.ops import xyxyxyxy2xywhr


def pca_angle_deg(points: np.ndarray) -> float:
    """Angle (degrees) of the first principal axis of a set of 2D points."""
    angle, _ = pca_angle_and_elongation(points)
    return angle


def pca_angle_and_elongation(points: np.ndarray) -> Tuple[float, float]:
    """Angle (degrees) of the first principal axis, plus an elongation ratio
    (spread along the principal axis / spread along the secondary axis).
    Elongation near 1 means the point cloud is roughly round - its principal
    axis direction is essentially noise, not a meaningful orientation."""
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 2:
        return 0.0, 1.0
    centered = points - points.mean(axis=0)
    if np.allclose(centered, 0):
        return 0.0, 1.0
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    vx, vy = vt[0]
    angle = float(np.degrees(np.arctan2(vy, vx)))
    elongation = float(s[0] / max(s[1], 1e-6)) if len(s) > 1 else 1.0
    return angle, elongation


def minarearect_angle_deg(polygon: np.ndarray) -> float:
    """Angle (degrees) of the long axis of `polygon`'s minimum-area enclosing
    rectangle, canonicalized via ultralytics' own `xyxyxyxy2xywhr` (the exact
    function used to interpret an OBB detector's predictions at inference, in
    predict_instances_obb) - using the same canonicalization here for training
    crops (obb_oriented_sample) guarantees both agree on which of the two
    possible 180-degrees-apart directions is "the" long-axis angle, so
    training and inference never disagree by a flip."""
    corners = cv2.boxPoints(cv2.minAreaRect(polygon.astype(np.float32))).astype(np.float32)
    xywhr = xyxyxyxy2xywhr(corners.reshape(1, 8))[0]
    return float(np.degrees(xywhr[4]))


def crop_patch_with_padding(img: np.ndarray, cx: float, cy: float, half: int) -> Tuple[np.ndarray, int, int]:
    """Square patch of side 2*half centered at (cx,cy), zero-padded past the
    image edges. Returns (patch, off_x, off_y) such that
    img_coord = patch_coord + (off_x, off_y)."""
    H, W = img.shape[:2]
    x1, y1 = int(round(cx - half)), int(round(cy - half))
    size = 2 * half
    patch = np.zeros((size, size), dtype=img.dtype)
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(W, x1 + size), min(H, y1 + size)
    if sx2 > sx1 and sy2 > sy1:
        dx1, dy1 = sx1 - x1, sy1 - y1
        patch[dy1:dy1 + (sy2 - sy1), dx1:dx1 + (sx2 - sx1)] = img[sy1:sy2, sx1:sx2]
    return patch, x1, y1


def rotation_matrix(center: Tuple[float, float], angle_deg: float) -> np.ndarray:
    return cv2.getRotationMatrix2D(center, angle_deg, 1.0)


def apply_affine_points(points: np.ndarray, M: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homog = np.hstack([points, ones])
    return (M @ homog.T).T


def patch_half_size(extent: float, margin: int = 16) -> int:
    """Half-side of the local patch to rotate: big enough that the filament
    (of the given bbox diagonal `extent`) can't be clipped by any rotation."""
    return int(np.ceil(extent * 0.75)) + margin


def clip_patch_to_image(patch_array: np.ndarray, off_x: int, off_y: int, H: int, W: int):
    """Inverse of crop_patch_with_padding: given an array in the off_x/off_y
    local patch frame, return (sub_array, sx1, sy1, sx2, sy2) where
    label_map[sy1:sy2, sx1:sx2] is exactly the placement `sub_array`
    corresponds to in the full HxW image (empty sub_array if fully out of
    bounds)."""
    size = patch_array.shape[0]
    sx1, sy1 = max(0, off_x), max(0, off_y)
    sx2, sy2 = min(W, off_x + size), min(H, off_y + size)
    if sx2 <= sx1 or sy2 <= sy1:
        return patch_array[:0, :0], sx1, sy1, sx1, sy1
    dx1, dy1 = sx1 - off_x, sy1 - off_y
    sub_array = patch_array[dy1:dy1 + (sy2 - sy1), dx1:dx1 + (sx2 - sx1)]
    return sub_array, sx1, sy1, sx2, sy2
