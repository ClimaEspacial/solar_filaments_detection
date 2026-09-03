"""Aspect-preserving (letterboxed) crops around each filament's bbox, for the
U-Net stage.

Filaments are long and thin - a naive square crop around an elongated bbox
wastes most of its resolution on empty margin along the short axis once
resized to `crop_size`, which directly hurts the model's ability to resolve
barbs. Instead we pad width/height independently, then letterbox-resize
(scale to fit, pad with black) into a square canvas - this keeps the
filament's true aspect ratio undistorted.
"""
import random
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from solar_filaments.data.augment_ops import random_gamma
from solar_filaments.data.coco import Dataset, load_image, polygon_to_mask
from solar_filaments.data.oriented_crop import (apply_affine_points, crop_patch_with_padding,
                                                 minarearect_angle_deg, patch_half_size, pca_angle_deg,
                                                 rotation_matrix)


def padded_rect_crop(x: float, y: float, w: float, h: float, W: int, H: int,
                      pad_factor: float, min_side: int) -> Tuple[int, int, int, int]:
    """Independently-padded rectangle around a bbox (NOT forced to a square) -
    preserves the filament's true aspect ratio; see letterbox_resize below."""
    cx, cy = x + w / 2, y + h / 2
    pad_w, pad_h = max(w * pad_factor, min_side), max(h * pad_factor, min_side)
    x1, y1 = max(0, int(round(cx - pad_w / 2))), max(0, int(round(cy - pad_h / 2)))
    x2, y2 = min(W, int(round(cx + pad_w / 2))), min(H, int(round(cy + pad_h / 2)))
    return x1, y1, x2, y2


def letterbox_resize(img: np.ndarray, size: int, interp=cv2.INTER_LINEAR):
    """Scale `img` so its longer side == size, pad the rest with zeros to reach
    a size x size square. Returns (canvas, scale, left, top) - the last three
    let you invert the transform exactly (see undo_letterbox)."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    canvas = np.zeros((size, size), dtype=img.dtype)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, scale, left, top


def undo_letterbox(canvas: np.ndarray, scale: float, left: int, top: int,
                    orig_w: int, orig_h: int, interp=cv2.INTER_NEAREST) -> np.ndarray:
    new_w, new_h = max(1, round(orig_w * scale)), max(1, round(orig_h * scale))
    cropped = canvas[top:top + new_h, left:left + new_w]
    return cv2.resize(cropped, (orig_w, orig_h), interpolation=interp)


def _oriented_sample(img: np.ndarray, poly: np.ndarray, angle: float, crop_size: int,
                      pad_factor: float, min_crop_side: int):
    """Shared geometry for spine_oriented_sample/obb_oriented_sample: rotate a
    local patch around `poly`'s centroid by `angle` (degrees), then crop+mask+
    letterbox the rotated filament - see oriented_crop.py for the primitives."""
    centroid = poly.mean(axis=0)
    diag = float(np.linalg.norm(poly.max(axis=0) - poly.min(axis=0)))
    half = patch_half_size(diag)
    patch, off_x, off_y = crop_patch_with_padding(img, centroid[0], centroid[1], half)

    local_center = (centroid[0] - off_x, centroid[1] - off_y)
    M = rotation_matrix(local_center, angle)
    rotated = cv2.warpAffine(patch, M, (patch.shape[1], patch.shape[0]), flags=cv2.INTER_LINEAR)

    poly_rot = apply_affine_points(poly - np.array([off_x, off_y], dtype=np.float32), M)
    mask_rot = np.zeros(rotated.shape, dtype=np.uint8)
    cv2.fillPoly(mask_rot, [poly_rot.round().astype(np.int32)], 1)

    bx1, by1 = poly_rot.min(axis=0)
    bx2, by2 = poly_rot.max(axis=0)
    cx1, cy1, cx2, cy2 = padded_rect_crop(bx1, by1, bx2 - bx1, by2 - by1,
                                           rotated.shape[1], rotated.shape[0], pad_factor, min_crop_side)
    crop = rotated[cy1:cy2, cx1:cx2]
    mask = mask_rot[cy1:cy2, cx1:cx2]

    crop, _, _, _ = letterbox_resize(crop, crop_size, cv2.INTER_LINEAR)
    mask, _, _, _ = letterbox_resize(mask, crop_size, cv2.INTER_NEAREST)
    return crop, mask


def spine_oriented_sample(img: np.ndarray, segmentation, spine, crop_size: int,
                           pad_factor: float, min_crop_side: int):
    """Crop+letterbox a filament rotated so its GT spine lies horizontal,
    instead of a plain axis-aligned bbox crop. Uses the spine polyline's
    principal axis for the angle - see oriented_crop.py."""
    poly = np.asarray(segmentation, dtype=np.float32).reshape(-1, 2)
    spine_pts = np.asarray(spine, dtype=np.float32).reshape(-1, 2)
    angle = pca_angle_deg(spine_pts)
    return _oriented_sample(img, poly, angle, crop_size, pad_factor, min_crop_side)


def obb_oriented_sample(img: np.ndarray, segmentation, crop_size: int,
                         pad_factor: float, min_crop_side: int):
    """Crop+letterbox a filament rotated so the long edge of its minimum-area
    enclosing rectangle lies horizontal - the same geometry an OBB detector
    is trained to predict, so the U-Net sees crops matching what
    predict_instances_obb will produce at inference (see oriented_crop.py)."""
    poly = np.asarray(segmentation, dtype=np.float32).reshape(-1, 2)
    angle = minarearect_angle_deg(poly)
    return _oriented_sample(img, poly, angle, crop_size, pad_factor, min_crop_side)


class FilamentCropDataset(TorchDataset):
    def __init__(self, dataset: Dataset, train_img_dir, image_ids, crop_size: int,
                 pad_factor: float, min_crop_side: int, augment: bool = False,
                 use_spine_orientation: bool = False, use_obb_orientation: bool = False,
                 synthetic_img_dir=None, preprocess: bool = False):
        self.dataset = dataset
        self.train_img_dir = train_img_dir
        self.synthetic_img_dir = synthetic_img_dir
        self.crop_size = crop_size
        self.pad_factor = pad_factor
        self.min_crop_side = min_crop_side
        self.augment = augment
        self.use_spine_orientation = use_spine_orientation
        self.use_obb_orientation = use_obb_orientation
        self.preprocess = preprocess
        self.samples = [
            (img_id, ann)
            for img_id in image_ids
            for ann in dataset.anns_by_image[img_id]
        ]
        self._img_cache = {}

    def __len__(self):
        return len(self.samples)

    def _get_image(self, img_id):
        if img_id not in self._img_cache:
            path = self.dataset.resolve_image_path(img_id, self.train_img_dir, self.synthetic_img_dir)
            self._img_cache[img_id] = load_image(path, preprocess=self.preprocess)
        return self._img_cache[img_id]

    def __getitem__(self, idx):
        img_id, ann = self.samples[idx]
        img = self._get_image(img_id)
        H, W = img.shape

        if self.use_obb_orientation:
            crop, mask = obb_oriented_sample(img, ann["segmentation"],
                                              self.crop_size, self.pad_factor, self.min_crop_side)
        elif self.use_spine_orientation:
            crop, mask = spine_oriented_sample(img, ann["segmentation"], ann["spine"],
                                                self.crop_size, self.pad_factor, self.min_crop_side)
        else:
            full_mask = polygon_to_mask(ann["segmentation"], H, W)
            x, y, w, h = ann["bbox"]
            x1, y1, x2, y2 = padded_rect_crop(x, y, w, h, W, H, self.pad_factor, self.min_crop_side)
            crop = img[y1:y2, x1:x2]
            mask = full_mask[y1:y2, x1:x2]
            crop, _, _, _ = letterbox_resize(crop, self.crop_size, cv2.INTER_LINEAR)
            mask, _, _, _ = letterbox_resize(mask, self.crop_size, cv2.INTER_NEAREST)

        if self.augment:
            if random.random() < 0.5:
                crop, mask = crop[:, ::-1].copy(), mask[:, ::-1].copy()
            if random.random() < 0.5:
                crop, mask = crop[::-1, :].copy(), mask[::-1, :].copy()
            if not (self.use_spine_orientation or self.use_obb_orientation):
                # a 90-degree rotation would undo the whole point of a canonical
                # horizontal orientation, so skip it in either oriented mode.
                k = random.choice([0, 1, 2, 3])
                if k:
                    crop = np.rot90(crop, k).copy()
                    mask = np.rot90(mask, k).copy()
            crop = random_gamma(crop)  # see data/augment_ops.py for the rationale

        crop_t = torch.from_numpy(crop).float().unsqueeze(0) / 255.0
        mask_t = torch.from_numpy(mask).float().unsqueeze(0)
        return crop_t, mask_t
