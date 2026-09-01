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

from solar_filaments.data.coco import Dataset, load_image, polygon_to_mask


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


class FilamentCropDataset(TorchDataset):
    def __init__(self, dataset: Dataset, train_img_dir, image_ids, crop_size: int,
                 pad_factor: float, min_crop_side: int, augment: bool = False):
        self.dataset = dataset
        self.train_img_dir = train_img_dir
        self.crop_size = crop_size
        self.pad_factor = pad_factor
        self.min_crop_side = min_crop_side
        self.augment = augment
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
            info = self.dataset.images_by_id[img_id]
            self._img_cache[img_id] = load_image(self.train_img_dir / info["file_name"])
        return self._img_cache[img_id]

    def __getitem__(self, idx):
        img_id, ann = self.samples[idx]
        img = self._get_image(img_id)
        H, W = img.shape
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
            k = random.choice([0, 1, 2, 3])
            if k:
                crop = np.rot90(crop, k).copy()
                mask = np.rot90(mask, k).copy()
            if random.random() < 0.3:
                gain = random.uniform(0.85, 1.15)
                bias = random.uniform(-10, 10)
                crop = np.clip(crop.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)

        crop_t = torch.from_numpy(crop).float().unsqueeze(0) / 255.0
        mask_t = torch.from_numpy(mask).float().unsqueeze(0)
        return crop_t, mask_t
