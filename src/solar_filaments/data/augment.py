"""Copy-paste data augmentation: extract real filaments from the train split
and paste them onto other train images, to multiply effective training
examples from a small (637-image) dataset.

Only ever operates on `train_ids` - augmentation must never leak synthetic
data into a validation split. Pasted filaments are placed on-disk (away from
the black off-disk background) and away from existing GT boxes, composited
using the filament's own polygon mask (not its bounding rectangle) with a
feathered edge and a brightness match, to avoid an obvious cut-paste tell.
"""
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np

from solar_filaments.data.coco import Dataset, load_image
from solar_filaments.data.crop_dataset import padded_rect_crop
from solar_filaments.data.preprocess import estimate_disk_mask


@dataclass
class FilamentPatch:
    img: np.ndarray            # grayscale crop containing the filament
    mask: np.ndarray           # uint8 {0,1}, same shape as img
    polygon_local: np.ndarray  # Nx2, in img's local coordinates
    spine_local: Optional[np.ndarray]  # Mx2 or None
    bbox_local: Tuple[float, float, float, float]  # x, y, w, h in img's local coords
    category_id: int


def build_filament_bank(dataset: Dataset, train_img_dir: Path, train_ids: Set,
                         pad_factor: float = 1.15, min_side: int = 32) -> List[FilamentPatch]:
    """Extract a (crop, mask, polygon, spine) patch for every annotation in
    the train split, for later pasting elsewhere."""
    bank = []
    img_cache = {}
    for img_id in train_ids:
        info = dataset.images_by_id[img_id]
        if img_id not in img_cache:
            img_cache[img_id] = load_image(train_img_dir / info["file_name"])
        img = img_cache[img_id]
        H, W = img.shape

        for ann in dataset.anns_by_image[img_id]:
            x, y, w, h = ann["bbox"]
            x1, y1, x2, y2 = padded_rect_crop(x, y, w, h, W, H, pad_factor, min_side)
            if x2 <= x1 or y2 <= y1:
                continue

            crop_img = img[y1:y2, x1:x2].copy()
            offset = np.array([x1, y1], dtype=np.float32)

            poly = np.asarray(ann["segmentation"], dtype=np.float32).reshape(-1, 2) - offset
            mask = np.zeros(crop_img.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [poly.round().astype(np.int32)], 1)
            if mask.sum() < 4:
                continue

            spine = ann.get("spine")
            spine_local = np.asarray(spine, dtype=np.float32).reshape(-1, 2) - offset if spine else None

            bank.append(FilamentPatch(
                img=crop_img, mask=mask, polygon_local=poly, spine_local=spine_local,
                bbox_local=(x - x1, y - y1, w, h), category_id=ann.get("category_id", 1),
            ))
    return bank


def find_paste_location(disk_mask: np.ndarray, existing_boxes: List[Tuple[float, float, float, float]],
                         patch_shape: Tuple[int, int], rng: random.Random,
                         max_tries: int = 20) -> Optional[Tuple[float, float]]:
    """Rejection-sample a paste center that keeps the whole patch on-disk and
    away from existing GT boxes on the target image."""
    H, W = disk_mask.shape
    ph, pw = patch_shape
    half_h, half_w = ph / 2, pw / 2
    if pw >= W or ph >= H:
        return None

    for _ in range(max_tries):
        cx = rng.uniform(half_w, W - half_w)
        cy = rng.uniform(half_h, H - half_h)
        corners = [(cx - half_w, cy - half_h), (cx + half_w, cy - half_h),
                   (cx - half_w, cy + half_h), (cx + half_w, cy + half_h), (cx, cy)]
        if not all(disk_mask[int(py), int(px)] for px, py in corners):
            continue

        x1, y1, x2, y2 = cx - half_w, cy - half_h, cx + half_w, cy + half_h
        if any(x1 < bx + bw and x2 > bx and y1 < by + bh and y2 > by for bx, by, bw, bh in existing_boxes):
            continue
        return cx, cy
    return None


def paste_patch(target_img: np.ndarray, patch: FilamentPatch, cx: float, cy: float):
    """Composite `patch` into `target_img` centered at (cx, cy), using the
    patch's own mask (feathered) plus a brightness match. Returns
    (composited_img, polygon, spine, bbox) in target-image coordinates, or
    None if the patch would fall outside the target image."""
    H, W = target_img.shape
    ph, pw = patch.img.shape
    x1, y1 = int(round(cx - pw / 2)), int(round(cy - ph / 2))
    x2, y2 = x1 + pw, y1 + ph
    if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
        return None

    mask = patch.mask.astype(np.float32)
    soft_mask = np.clip(cv2.GaussianBlur(mask, (5, 5), 0), 0, 1)

    target_region = target_img[y1:y2, x1:x2].astype(np.float32)
    patch_pixels = patch.img.astype(np.float32)

    fg = mask > 0.5
    if fg.any():
        patch_mean = patch_pixels[fg].mean()
        if patch_mean > 1e-3:
            gain = np.clip(target_region[fg].mean() / patch_mean, 0.6, 1.6)
            patch_pixels = patch_pixels * gain

    composited = soft_mask * patch_pixels + (1 - soft_mask) * target_region
    out = target_img.copy()
    out[y1:y2, x1:x2] = np.clip(composited, 0, 255).astype(target_img.dtype)

    offset = np.array([x1, y1], dtype=np.float32)
    polygon = patch.polygon_local + offset
    spine = patch.spine_local + offset if patch.spine_local is not None else None
    bx, by, bw, bh = patch.bbox_local
    bbox = (bx + x1, by + y1, bw, bh)
    return out, polygon, spine, bbox


def generate_synthetic_images(dataset: Dataset, train_img_dir: Path, synthetic_dir: Path,
                               train_ids: Set, n_synthetic: int, max_pastes_per_image: int = 2,
                               seed: int = 0, id_prefix: str = "synthetic") -> Set:
    """Generate `n_synthetic` composite training images by pasting real
    filaments (from `train_ids` only) onto other `train_ids` backgrounds.
    Mutates `dataset` in place with new synthetic image/annotation entries.
    Returns the set of new synthetic image ids (add these to train_ids;
    never to val_ids). `id_prefix` must be unique per call when generating
    synthetic images for multiple folds against the same Dataset instance,
    so their ids don't collide."""
    rng = random.Random(seed)
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    bank = build_filament_bank(dataset, train_img_dir, train_ids)
    if not bank:
        return set()

    train_id_list = sorted(train_ids)
    img_cache = {}
    new_ids = set()
    next_ann_id = 0

    for i in range(n_synthetic):
        bg_id = rng.choice(train_id_list)
        info = dataset.images_by_id[bg_id]
        if bg_id not in img_cache:
            img_cache[bg_id] = load_image(train_img_dir / info["file_name"])
        img = img_cache[bg_id].copy()
        H, W = img.shape

        disk_mask = estimate_disk_mask(img)
        existing_boxes = [tuple(ann["bbox"]) for ann in dataset.anns_by_image[bg_id]]

        new_anns = []
        for _ in range(rng.randint(1, max_pastes_per_image)):
            patch = rng.choice(bank)
            loc = find_paste_location(disk_mask, existing_boxes, patch.img.shape, rng)
            if loc is None:
                continue
            result = paste_patch(img, patch, loc[0], loc[1])
            if result is None:
                continue
            img, polygon, spine, bbox = result
            existing_boxes.append(bbox)
            next_ann_id += 1
            new_anns.append({
                "segmentation": [polygon.flatten().tolist()],
                "area": float(bbox[2] * bbox[3]),
                "iscrowd": 0,
                "spine": spine.flatten().tolist() if spine is not None else [],
                "bbox": list(bbox),
                "category_id": patch.category_id,
                "id": f"{id_prefix}-{i:05d}-{next_ann_id}",
            })

        if not new_anns:
            continue

        file_name = f"{id_prefix}_{i:05d}.jpeg"
        cv2.imwrite(str(synthetic_dir / file_name), img)

        new_img_id = f"{id_prefix}-{i:05d}"
        dataset.images_by_id[new_img_id] = {
            "file_name": file_name, "height": H, "width": W, "synthetic": True, "id": new_img_id,
        }
        dataset.file_name_to_image_ids[file_name].append(new_img_id)
        for ann in new_anns:
            ann["image_id"] = new_img_id
        dataset.anns_by_image[new_img_id] = new_anns
        new_ids.add(new_img_id)

    return new_ids
