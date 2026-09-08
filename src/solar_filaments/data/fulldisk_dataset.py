"""Full-disk semantic segmentation samples (one filaments-vs-background mask per
image) for the Flat U-Net branch.

Unlike the crop dataset (one sample per filament) this yields one sample per
*physical image*, with the mask built from the annotators' filaments according
to `annotator_merge` (see `coco.merge_annotator_records`).

Choosing that mode is a real trade-off, measured on this data. The `union` of
every annotator's filaments maximizes the recall signal, which is what a
candidate-validation branch nominally wants - but it also asks the network to
fit filaments that most annotators did *not* mark: 31% of clustered filaments
on multi-annotator images have support below 0.5. Those are precisely the
low-contrast cases the detector also misses (69% miss rate at support 0.33 vs
13% at support 1.0), so training on the union means training partly on
disagreement. `majority` trades some recall for a target the annotators
actually agree on.

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
from solar_filaments.data.coco import Dataset, load_image, merge_annotator_records, polygon_to_mask


def minmax_normalize(img: np.ndarray) -> np.ndarray:
    """Per-image min-max to [0, 1] (the paper's only preprocessing step)."""
    img = img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    return (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)


def semantic_mask(anns, height: int, width: int) -> np.ndarray:
    """Binary filaments-vs-background mask from a list of annotation dicts."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in anns:
        mask |= polygon_to_mask(ann["segmentation"], height, width)
    return mask


class FullDiskDataset(TorchDataset):
    def __init__(self, dataset: Dataset, train_img_dir: Path, image_ids, imgsz: int = 512,
                 augment: bool = False, synthetic_img_dir: Optional[Path] = None,
                 annotator_merge: str = "union"):
        self.dataset, self.train_img_dir, self.synthetic_img_dir = dataset, train_img_dir, synthetic_img_dir
        self.imgsz, self.augment = imgsz, augment
        ids = set(image_ids)
        self.file_names = [fn for fn, rec_ids in dataset.file_name_to_image_ids.items()
                           if any(i in ids for i in rec_ids)]
        # Cluster annotators once here, not per __getitem__: the clustering is O(0.3 s)
        # per multi-annotator image and would otherwise be repeated every epoch.
        self.anns_by_file = {fn: merge_annotator_records(dataset, fn, annotator_merge)
                             for fn in self.file_names}

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        img_id = self.dataset.file_name_to_image_ids[file_name][0]
        info = self.dataset.images_by_id[img_id]
        img = load_image(self.dataset.resolve_image_path(img_id, self.train_img_dir, self.synthetic_img_dir))
        mask = semantic_mask(self.anns_by_file[file_name], info["height"], info["width"])

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
