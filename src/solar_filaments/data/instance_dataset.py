"""Full-image instance targets (boxes + per-instance masks) for the Mask R-CNN
backend, one sample per *physical* image with annotator-consensus instances
(coco.merge_annotator_records)."""
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from solar_filaments.data.augment_ops import random_gamma
from solar_filaments.data.coco import Dataset, load_image, merged_records, polygon_to_mask


class FilamentInstanceDataset(TorchDataset):
    def __init__(self, dataset: Dataset, train_img_dir: Path, image_ids, imgsz: int = 1024,
                 augment: bool = False, annotator_merge: str = "majority",
                 synthetic_img_dir: Optional[Path] = None):
        self.dataset, self.train_img_dir, self.synthetic_img_dir = dataset, train_img_dir, synthetic_img_dir
        self.imgsz, self.augment = imgsz, augment
        self.records = merged_records(dataset, image_ids, annotator_merge)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        file_name, info, anns = self.records[idx]
        img_id = self.dataset.file_name_to_image_ids[file_name][0]
        img = load_image(self.dataset.resolve_image_path(img_id, self.train_img_dir, self.synthetic_img_dir))
        H, W = img.shape
        s = self.imgsz
        small = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)

        masks, boxes = [], []
        for ann in anns:
            m = polygon_to_mask(ann["segmentation"], H, W)
            m = cv2.resize(m, (s, s), interpolation=cv2.INTER_NEAREST)
            if not m.any():
                continue
            ys, xs = np.where(m)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            masks.append(m)

        if self.augment:
            small = random_gamma(small)
            if random.random() < 0.5:  # horizontal flip
                small = small[:, ::-1]
                masks = [m[:, ::-1] for m in masks]
                boxes = [[s - x2, y1, s - x1, y2] for x1, y1, x2, y2 in boxes]
            if random.random() < 0.5:  # vertical flip
                small = small[::-1, :]
                masks = [m[::-1, :] for m in masks]
                boxes = [[x1, s - y2, x2, s - y1] for x1, y1, x2, y2 in boxes]

        image = torch.from_numpy(np.ascontiguousarray(small)).float().div_(255.0).unsqueeze(0).repeat(3, 1, 1)
        if masks:
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.ones(len(masks), dtype=torch.int64),
                "masks": torch.from_numpy(np.stack([np.ascontiguousarray(m) for m in masks]).astype(np.uint8)),
            }
        else:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, s, s), dtype=torch.uint8),
            }
        return image, target


def collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
