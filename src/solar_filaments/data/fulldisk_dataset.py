"""Full-disk semantic segmentation samples (one filaments-vs-background mask per
image) for the Flat U-Net branch.

Unlike the crop dataset (one sample per filament) this yields one sample per
*physical image*, with the mask being the union of every annotator's filaments.
The union - not the majority consensus - is deliberate: this branch exists to
provide an independent high-recall "is there a filament here?" signal used to
revalidate the detector's low-confidence boxes, so it should know about every
filament anyone marked.

Preprocessing is min-max normalization only. The Flat U-Net paper is explicit
that full-disk H-alpha images "require only min-max normalization without the
removal of limb darkening and unevenness correction" - which independently
matches this project's own negative result on limb-darkening flattening + CLAHE.
"""
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from solar_filaments.data.augment_ops import random_gamma
from solar_filaments.data.coco import Dataset, load_image, polygon_to_mask


def minmax_normalize(img: np.ndarray) -> np.ndarray:
    """Per-image min-max to [0, 1] (the paper's only preprocessing step)."""
    img = img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    return (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)


def union_mask(dataset: Dataset, file_name: str, height: int, width: int) -> np.ndarray:
    """Union of every filament from every annotator record of this image."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for img_id in dataset.file_name_to_image_ids[file_name]:
        for ann in dataset.anns_by_image[img_id]:
            mask |= polygon_to_mask(ann["segmentation"], height, width)
    return mask


class FullDiskDataset(TorchDataset):
    def __init__(self, dataset: Dataset, train_img_dir: Path, image_ids, imgsz: int = 512,
                 augment: bool = False, synthetic_img_dir: Optional[Path] = None):
        self.dataset, self.train_img_dir, self.synthetic_img_dir = dataset, train_img_dir, synthetic_img_dir
        self.imgsz, self.augment = imgsz, augment
        ids = set(image_ids)
        self.file_names = [fn for fn, rec_ids in dataset.file_name_to_image_ids.items()
                           if any(i in ids for i in rec_ids)]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        img_id = self.dataset.file_name_to_image_ids[file_name][0]
        info = self.dataset.images_by_id[img_id]
        img = load_image(self.dataset.resolve_image_path(img_id, self.train_img_dir, self.synthetic_img_dir))
        mask = union_mask(self.dataset, file_name, info["height"], info["width"])

        s = self.imgsz
        img = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (s, s), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            img = random_gamma(img)
            if random.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            if random.random() < 0.5:
                img, mask = img[::-1, :], mask[::-1, :]
            k = random.choice([0, 1, 2, 3])
            if k:
                img, mask = np.rot90(img, k), np.rot90(mask, k)

        img_t = torch.from_numpy(minmax_normalize(np.ascontiguousarray(img))).unsqueeze(0)
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)
        return img_t, mask_t
