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
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from pycocotools import mask as mask_util


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

    def resolve_image_path(self, img_id, train_img_dir: Path,
                            synthetic_img_dir: Optional[Path] = None) -> Path:
        """Real images live under train_img_dir; copy-paste-augmented images
        (see data/augment.py) are flagged `synthetic=True` and live under
        synthetic_img_dir instead."""
        info = self.images_by_id[img_id]
        base = synthetic_img_dir if info.get("synthetic") and synthetic_img_dir is not None else train_img_dir
        return base / info["file_name"]

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
    """segmentation: flat [x0,y0,x1,y1,...] list (MAGFiLO format) -> uint8 HxW binary mask.

    Rasterized with pycocotools (`frPyObjects` + `merge` + `decode`, i.e. exactly what
    `COCO.annToMask` does), NOT `cv2.fillPoly`. This is the competition's own standard: the
    rules say to convert polygons with pycocotools' `annToMask`, and the submission format is
    pycocotools RLE, so the organizers' ground-truth masks are almost certainly built this way.

    The two conventions genuinely disagree: measured over 380 MAGFiLO filaments, cv2.fillPoly
    masks are ~10% *larger* (area ratio pycoco/cv2 = 0.904, mean IoU between the two = 0.881),
    and thin filaments suffer most (IoU 0.857 for below-median-area vs 0.904 above) since the
    discrepancy is a boundary effect and filaments have a high perimeter-to-area ratio. Using
    cv2 here inflated every local PQ (~+0.04) *and* taught the U-Net to predict masks ~10%
    fatter than the evaluation standard - invisible locally, penalized on the real board.
    """
    polygons = ([list(map(float, p)) for p in segmentation]
                if segmentation and isinstance(segmentation[0], list)
                else [list(map(float, segmentation))])
    rles = mask_util.frPyObjects(polygons, height, width)
    return mask_util.decode(mask_util.merge(rles)).astype(np.uint8)


def _bbox_of_mask(mask: np.ndarray) -> List[float]:
    ys, xs = np.where(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]


def _polygon_of_mask(mask: np.ndarray) -> List[float]:
    """Flat [x0,y0,x1,y1,...] polygon of the largest external contour of `mask`
    (the MAGFiLO single-polygon segmentation format)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2).astype(float).flatten().tolist()


def _mask_iou_in_window(a: np.ndarray, b: np.ndarray, bbox_a, bbox_b) -> float:
    """IoU of two full-image masks, computed only inside the union of their
    bboxes (a 2048x2048 boolean op per pair would dominate prepare time)."""
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b
    if ax > bx + bw or bx > ax + aw or ay > by + bh or by > ay + ah:
        return 0.0
    x0, y0 = int(min(ax, bx)), int(min(ay, by))
    x1, y1 = int(max(ax + aw, bx + bw)) + 1, int(max(ay + ah, by + bh)) + 1
    wa, wb = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
    union = np.logical_or(wa, wb).sum()
    return float(np.logical_and(wa, wb).sum() / union) if union else 0.0


ANNOTATOR_MERGE_MODES = ("single", "union", "majority")


def merge_annotator_records(dataset: "Dataset", file_name: str, mode: str = "majority") -> List[dict]:
    """One consensus instance list for a physical image that may carry 1-3
    independent annotator records.

    Instances are clustered across annotators with the evaluator's own matching
    rule (mask IoU > 0.5, at most one member per annotator per cluster, greedy
    best-IoU-first). Each cluster becomes one annotation dict in the usual
    MAGFiLO shape (`segmentation` flat polygon, `bbox`, `category_id`) whose mask
    is the *union* of its members' polygons - the target should capture a
    filament's full extent - plus `support` = fraction of annotators who marked
    it and `n_annotators`.

    mode: "single"   - the first record only (what a per-record label file
                       effectively did before, modulo which record won);
          "union"    - every cluster, whatever its support;
          "majority" - clusters with support >= 0.5. PQ is pooled over every
                       annotator-record separately, so predicting a filament
                       only a minority of annotators marked costs more (FP
                       against the others) than it earns - this is the
                       PQ-optimal target set to train a detector on.
    Single-annotator images are identical under all three modes."""
    if mode not in ANNOTATOR_MERGE_MODES:
        raise ValueError(f"unknown annotator merge mode {mode!r}, expected one of {ANNOTATOR_MERGE_MODES}")
    ids = dataset.file_name_to_image_ids[file_name]
    n_annotators = len(ids)
    if mode == "single" or n_annotators == 1:
        out = []
        for ann in dataset.anns_by_image[ids[0]]:
            merged = dict(ann)
            merged["support"], merged["n_annotators"] = 1.0, n_annotators
            out.append(merged)
        return out

    info = dataset.images_by_id[ids[0]]
    H, W = info["height"], info["width"]

    clusters: List[dict] = []  # {"mask", "bbox", "members": [(annotator_idx, ann)]}
    for k, img_id in enumerate(ids):
        anns = dataset.anns_by_image[img_id]
        masks = [polygon_to_mask(a["segmentation"], H, W).astype(bool) for a in anns]
        # best-IoU-first greedy assignment, one member per annotator per cluster
        pairs = []
        for ai, (a, m) in enumerate(zip(anns, masks)):
            for ci, c in enumerate(clusters):
                if any(mk == k for mk, _ in c["members"]):
                    continue
                iou = _mask_iou_in_window(m, c["mask"], a["bbox"], c["bbox"])
                if iou > 0.5:
                    pairs.append((iou, ai, ci))
        pairs.sort(reverse=True)
        assigned_anns, taken_clusters = set(), set()
        for iou, ai, ci in pairs:
            if ai in assigned_anns or ci in taken_clusters:
                continue
            c = clusters[ci]
            c["mask"] |= masks[ai]
            c["bbox"] = _bbox_of_mask(c["mask"])
            c["members"].append((k, anns[ai]))
            assigned_anns.add(ai)
            taken_clusters.add(ci)
        for ai, (a, m) in enumerate(zip(anns, masks)):
            if ai not in assigned_anns:
                clusters.append({"mask": m.copy(), "bbox": list(a["bbox"]), "members": [(k, a)]})

    out = []
    for c in clusters:
        support = len(c["members"]) / n_annotators
        if mode == "majority" and support < 0.5:
            continue
        polygon = _polygon_of_mask(c["mask"])
        if len(polygon) < 6:
            continue
        cats = collections.Counter(a.get("category_id", 1) for _, a in c["members"])
        first = c["members"][0][1]
        out.append({
            "segmentation": polygon,
            "bbox": _bbox_of_mask(c["mask"]),
            "area": float(c["mask"].sum()),
            "iscrowd": 0,
            "spine": first.get("spine", []),
            "category_id": cats.most_common(1)[0][0],
            "id": f"{first.get('id', '')}-merged",
            "image_id": ids[0],
            "support": support,
            "n_annotators": n_annotators,
        })
    return out


def merged_records(dataset: "Dataset", image_ids, mode: str = "majority") -> List[Tuple[str, dict, List[dict]]]:
    """Group `image_ids` (a split) by physical image and merge each one's
    annotator records. Splits are built per file_name, so every record of a
    file is either entirely in or entirely out of `image_ids`. Returns
    (file_name, image_info, merged_anns) per unique image, in a stable order."""
    ids = set(image_ids)
    seen = []
    for file_name, rec_ids in dataset.file_name_to_image_ids.items():
        if any(i in ids for i in rec_ids):
            seen.append(file_name)
    return [(fn, dataset.images_by_id[dataset.file_name_to_image_ids[fn][0]],
             merge_annotator_records(dataset, fn, mode)) for fn in seen]


def load_image(path: Path, preprocess: bool = False) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    if preprocess:
        from solar_filaments.data.preprocess import preprocess_image
        img = preprocess_image(img)
    return img


def train_val_split(dataset: "Dataset", seed: int, val_frac: float = 0.1) -> Tuple[set, set]:
    """Split at the physical-image (file_name) level, not the image-record
    (image_id) level, so that a photo one annotator labeled doesn't end up in
    train while the same photo under a different annotator's image_id ends up
    in val - MAGFiLO images can carry up to 3 independent annotator records
    sharing one file_name, and splitting on image_id alone would leak that
    photo's pixels into both splits.

    `val_frac <= 0` yields an empty val set (everything in train) - useful
    for a final model trained on all data once cross-validation has picked a
    config."""
    file_names = list(dataset.file_name_to_image_ids.keys())
    random.Random(seed).shuffle(file_names)
    n_val_files = 0 if val_frac <= 0 else max(1, int(val_frac * len(file_names)))
    val_files = set(file_names[:n_val_files])
    train_files = set(file_names[n_val_files:])

    train_ids = {img_id for fn in train_files for img_id in dataset.file_name_to_image_ids[fn]}
    val_ids = {img_id for fn in val_files for img_id in dataset.file_name_to_image_ids[fn]}
    return train_ids, val_ids


def kfold_split(dataset: "Dataset", seed: int, k: int) -> List[Tuple[set, set]]:
    """K-fold split at the physical-image level (same leak-safe grouping as
    train_val_split). Returns a list of (train_ids, val_ids) pairs, one per
    fold; the val sets partition every unique image exactly once."""
    file_names = list(dataset.file_name_to_image_ids.keys())
    random.Random(seed).shuffle(file_names)

    folds: List[List[str]] = [file_names[i::k] for i in range(k)]

    splits = []
    for i in range(k):
        val_files = set(folds[i])
        train_files = set(file_names) - val_files
        train_ids = {img_id for fn in train_files for img_id in dataset.file_name_to_image_ids[fn]}
        val_ids = {img_id for fn in val_files for img_id in dataset.file_name_to_image_ids[fn]}
        splits.append((train_ids, val_ids))
    return splits


def save_split(split_path: Path, train_ids: set, val_ids: set) -> None:
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump({"train": sorted(train_ids), "val": sorted(val_ids)}, f)


def load_split(split_path: Path) -> Tuple[set, set]:
    with open(split_path) as f:
        d = json.load(f)
    return set(d["train"]), set(d["val"])
