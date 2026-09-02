"""Materialize the YOLO detection dataset (images symlinked, single-class boxes)."""
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from solar_filaments.data.coco import Dataset


def _symlink_split_images(dataset: Dataset, ids: set, split_img_dir: Path,
                           train_img_dir: Path, synthetic_img_dir: Optional[Path]) -> None:
    split_img_dir.mkdir(parents=True, exist_ok=True)
    for img_id in ids:
        info = dataset.images_by_id[img_id]
        src = dataset.resolve_image_path(img_id, train_img_dir, synthetic_img_dir)
        dst_img = split_img_dir / info["file_name"]
        if not dst_img.exists():
            os.symlink(src, dst_img)


def build_yolo_dataset(dataset: Dataset, train_img_dir: Path, yolo_dir: Path,
                        train_ids: set, val_ids: set, synthetic_img_dir: Optional[Path] = None) -> Path:
    """Symlinks images and writes axis-aligned YOLO-format label .txt files
    for train/val, plus a data.yaml. Returns the path to data.yaml."""
    for split, ids in [("train", train_ids), ("val", val_ids)]:
        _symlink_split_images(dataset, ids, yolo_dir / "images" / split, train_img_dir, synthetic_img_dir)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img_id in ids:
            info = dataset.images_by_id[img_id]
            w, h = info["width"], info["height"]
            lines = []
            for ann in dataset.anns_by_image[img_id]:
                x, y, bw, bh = ann["bbox"]
                cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
                nw, nh = bw / w, bh / h
                lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            label_path = yolo_dir / "labels" / split / (Path(info["file_name"]).stem + ".txt")
            label_path.write_text("\n".join(lines))

    data_yaml = f"""path: {yolo_dir}
train: images/train
val: images/val
names:
  0: filament
"""
    yaml_path = yolo_dir / "data.yaml"
    yaml_path.write_text(data_yaml)
    return yaml_path


def build_yolo_obb_dataset(dataset: Dataset, train_img_dir: Path, yolo_dir: Path,
                            train_ids: set, val_ids: set, synthetic_img_dir: Optional[Path] = None) -> Path:
    """Symlinks images and writes YOLO-OBB-format label .txt files (4 corner
    points per box, from the minimum-area rotated rectangle around each
    filament's segmentation polygon) for train/val, plus a data.yaml."""
    for split, ids in [("train", train_ids), ("val", val_ids)]:
        _symlink_split_images(dataset, ids, yolo_dir / "images" / split, train_img_dir, synthetic_img_dir)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img_id in ids:
            info = dataset.images_by_id[img_id]
            w, h = info["width"], info["height"]
            lines = []
            for ann in dataset.anns_by_image[img_id]:
                poly = np.asarray(ann["segmentation"], dtype=np.float32).reshape(-1, 2)
                corners = cv2.boxPoints(cv2.minAreaRect(poly))  # (4, 2), pixel coords
                norm = corners / np.array([w, h], dtype=np.float32)
                flat = " ".join(f"{v:.6f}" for v in norm.flatten())
                lines.append(f"0 {flat}")

            label_path = yolo_dir / "labels" / split / (Path(info["file_name"]).stem + ".txt")
            label_path.write_text("\n".join(lines))

    data_yaml = f"""path: {yolo_dir}
train: images/train
val: images/val
names:
  0: filament
"""
    yaml_path = yolo_dir / "data.yaml"
    yaml_path.write_text(data_yaml)
    return yaml_path
