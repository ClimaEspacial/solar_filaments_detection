"""Materialize YOLO datasets (images symlinked or preprocessed, one label file
per physical image) for the box, OBB and instance-segmentation trainers.

Labels are written per *physical image*, from `coco.merge_annotator_records`,
because MAGFiLO images can carry 1-3 annotator records sharing one file_name -
writing one file per record silently let the last record overwrite the others
for 42% of the training set. `annotator_merge` picks the consensus rule
("majority" is the PQ-optimal target set, see coco.py)."""
import os
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

from solar_filaments.data.coco import Dataset, load_image, merged_records


def _place_image(dataset: Dataset, img_id, dst_img: Path, train_img_dir: Path,
                 synthetic_img_dir: Optional[Path], preprocess: bool) -> None:
    src = dataset.resolve_image_path(img_id, train_img_dir, synthetic_img_dir)
    if dst_img.exists():
        return
    if preprocess:
        # YOLO reads image files directly off disk (not through load_image), so a
        # symlink to the raw file can't carry the preprocessing - write a real copy.
        cv2.imwrite(str(dst_img), load_image(src, preprocess=True))
    else:
        os.symlink(src, dst_img)


def _box_lines(anns: List[dict], w: int, h: int) -> List[str]:
    lines = []
    for ann in anns:
        x, y, bw, bh = ann["bbox"]
        lines.append(f"0 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
    return lines


def _obb_lines(anns: List[dict], w: int, h: int) -> List[str]:
    """4 corner points of the minimum-area rotated rectangle around the polygon."""
    lines = []
    for ann in anns:
        poly = np.asarray(ann["segmentation"], dtype=np.float32).reshape(-1, 2)
        corners = cv2.boxPoints(cv2.minAreaRect(poly))
        norm = corners / np.array([w, h], dtype=np.float32)
        lines.append("0 " + " ".join(f"{v:.6f}" for v in norm.flatten()))
    return lines


def _seg_lines(anns: List[dict], w: int, h: int) -> List[str]:
    """YOLO-seg format: class followed by the normalized polygon vertices."""
    lines = []
    for ann in anns:
        poly = np.asarray(ann["segmentation"], dtype=np.float32).reshape(-1, 2)
        if len(poly) < 3:
            continue
        norm = np.clip(poly / np.array([w, h], dtype=np.float32), 0.0, 1.0)
        lines.append("0 " + " ".join(f"{v:.6f}" for v in norm.flatten()))
    return lines


def _build(dataset: Dataset, train_img_dir: Path, yolo_dir: Path, train_ids: set, val_ids: set,
           synthetic_img_dir: Optional[Path], preprocess: bool, annotator_merge: str,
           label_fn: Callable[[List[dict], int, int], List[str]]) -> Path:
    for split, ids in [("train", train_ids), ("val", val_ids)]:
        img_dir, label_dir = yolo_dir / "images" / split, yolo_dir / "labels" / split
        # Wipe the split before repopulating. Without this, images from a *previous*
        # prepare with a different split linger: a later `--val-frac 0` run leaves the
        # old validation images in place while also adding them to train, so ultralytics
        # validates on data it trained on and selects `best.pt` on seen images. Observed
        # in practice (707 train / 70 stale val, val entirely contained in train).
        for d in (img_dir, label_dir):
            if d.exists():
                for f in d.iterdir():
                    if f.is_file() or f.is_symlink():
                        f.unlink()
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for file_name, info, anns in merged_records(dataset, ids, annotator_merge):
            img_id = dataset.file_name_to_image_ids[file_name][0]
            _place_image(dataset, img_id, img_dir / file_name, train_img_dir, synthetic_img_dir, preprocess)
            (label_dir / (Path(file_name).stem + ".txt")).write_text(
                "\n".join(label_fn(anns, info["width"], info["height"])))

    yaml_path = yolo_dir / "data.yaml"
    yaml_path.write_text(f"path: {yolo_dir}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n")
    return yaml_path


def build_yolo_dataset(dataset: Dataset, train_img_dir: Path, yolo_dir: Path,
                        train_ids: set, val_ids: set, synthetic_img_dir: Optional[Path] = None,
                        preprocess: bool = False, annotator_merge: str = "majority") -> Path:
    """Axis-aligned box labels. Returns the path to data.yaml."""
    return _build(dataset, train_img_dir, yolo_dir, train_ids, val_ids, synthetic_img_dir,
                  preprocess, annotator_merge, _box_lines)


def build_yolo_obb_dataset(dataset: Dataset, train_img_dir: Path, yolo_dir: Path,
                            train_ids: set, val_ids: set, synthetic_img_dir: Optional[Path] = None,
                            annotator_merge: str = "majority") -> Path:
    """Oriented-box labels (minAreaRect corners)."""
    return _build(dataset, train_img_dir, yolo_dir, train_ids, val_ids, synthetic_img_dir,
                  False, annotator_merge, _obb_lines)


def build_yolo_seg_dataset(dataset: Dataset, train_img_dir: Path, yolo_dir: Path,
                            train_ids: set, val_ids: set, synthetic_img_dir: Optional[Path] = None,
                            annotator_merge: str = "majority") -> Path:
    """Instance-segmentation polygon labels, for `yolo11*-seg` models."""
    return _build(dataset, train_img_dir, yolo_dir, train_ids, val_ids, synthetic_img_dir,
                  False, annotator_merge, _seg_lines)
