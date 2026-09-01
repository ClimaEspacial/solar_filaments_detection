"""Loading the MAGFiLO COCO-style annotations and image-level train/val split.

The annotation JSON is COCO-style but each `segmentation` is a single flat
polygon `[x0, y0, x1, y1, ...]` (not the usual COCO list-of-polygons-per-instance),
with an extra `spine` field (a polyline through the filament's long axis)
alongside the usual `bbox`. `category_id` encodes chirality, which is ignored
here for pure segmentation.
"""
import collections
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


class Dataset:
    """In-memory view over the COCO annotation file: images, per-image
    annotation lists, category names, and the detected annotator field."""

    def __init__(self, train_ann_path: Path):
        with open(train_ann_path) as f:
            coco = json.load(f)

        self.images_by_id: Dict[int, dict] = {im["id"]: im for im in coco["images"]}
        self.anns_by_image: Dict[int, list] = collections.defaultdict(list)
        for ann in coco["annotations"]:
            self.anns_by_image[ann["image_id"]].append(ann)

        self.cat_names = {c["id"]: c["name"] for c in coco.get("categories", [])}

        # MAGFiLO images can be labeled by 1-3 independent annotators. There is
        # no separate annotator field on the annotation objects - instead each
        # *image* record's own "id" is already "{annotator_code}-{image_stem}"
        # (multiple image records share the same file_name, one per
        # annotator), which is exactly the "{annotator}-{image_id}" prefix the
        # organizers' self-eval script expects for GT filament_id. So the
        # image dict key doubles as that prefix - no extra detection needed.
        self.file_name_to_image_ids: Dict[str, List] = collections.defaultdict(list)
        for img_id, info in self.images_by_id.items():
            self.file_name_to_image_ids[info["file_name"]].append(img_id)

    def summary(self) -> str:
        n_imgs = len(self.images_by_id)
        n_files = len(self.file_name_to_image_ids)
        n_anns = sum(len(v) for v in self.anns_by_image.values())
        n_multi_annot = sum(1 for ids in self.file_name_to_image_ids.values() if len(ids) > 1)
        return (
            f"{n_imgs} image records over {n_files} unique images "
            f"({n_multi_annot} with >1 annotator), {n_anns} filament instances "
            f"({n_anns / max(n_imgs, 1):.2f} filaments/image-record on average); "
            f"categories={self.cat_names}"
        )


def polygon_to_mask(segmentation: List[float], height: int, width: int) -> np.ndarray:
    """segmentation: flat [x0,y0,x1,y1,...] list (MAGFiLO format) -> uint8 HxW binary mask."""
    pts = np.asarray(segmentation, dtype=np.float32).reshape(-1, 2)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.round().astype(np.int32)], 1)
    return mask


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def train_val_split(dataset: "Dataset", seed: int, val_frac: float = 0.1) -> Tuple[set, set]:
    """Split at the physical-image (file_name) level, not the image-record
    (image_id) level, so that a photo one annotator labeled doesn't end up in
    train while the same photo under a different annotator's image_id ends up
    in val - MAGFiLO images can carry up to 3 independent annotator records
    sharing one file_name, and splitting on image_id alone would leak that
    photo's pixels into both splits."""
    file_names = list(dataset.file_name_to_image_ids.keys())
    random.Random(seed).shuffle(file_names)
    n_val_files = max(1, int(val_frac * len(file_names)))
    val_files = set(file_names[:n_val_files])
    train_files = set(file_names[n_val_files:])

    train_ids = {img_id for fn in train_files for img_id in dataset.file_name_to_image_ids[fn]}
    val_ids = {img_id for fn in val_files for img_id in dataset.file_name_to_image_ids[fn]}
    return train_ids, val_ids


def save_split(split_path: Path, train_ids: set, val_ids: set) -> None:
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump({"train": sorted(train_ids), "val": sorted(val_ids)}, f)


def load_split(split_path: Path) -> Tuple[set, set]:
    with open(split_path) as f:
        d = json.load(f)
    return set(d["train"]), set(d["val"])
