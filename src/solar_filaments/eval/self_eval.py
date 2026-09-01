"""Official self-evaluation logic, kept verbatim from the organizers' notebook
(https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook) so our
local PQ/plots match the real grader exactly.

- PQ is a single global score: TP/FP/FN are pooled across every image first,
  then `PQ = sum(IoU of TP pairs) / (|TP| + 0.5|FP| + 0.5|FN|)` - this is
  *not* the mean of per-image PQ values (that would be a different, more
  lenient number).
- Matching is a plain IoU > 0.5 threshold, not a 1-to-1 Hungarian/greedy
  match - a prediction that overlaps two different GT filaments above 0.5 IoU
  each (an over-merge) contributes to more than one "hit", which is
  intentional: it's exactly the many-to-one case `plot_m2n_counts` surfaces.
- Ground truth rows are keyed by "{annotator}-{image_id}_{n}" because MAGFiLO
  images can carry 1-3 independent annotator labelings; predictions are only
  keyed by `image_id` and get compared against *each* annotator's set
  separately.
"""
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch
from pycocotools import mask as mask_util

import matplotlib.pyplot as plt

from solar_filaments.data.coco import Dataset, polygon_to_mask
from solar_filaments.inference import predict_instances
from solar_filaments.rle import encode_rle_counts


def get_overlap_matrices(gt_layers: torch.Tensor, pred_layers: torch.Tensor):
    """Pairwise IoU and Dice between GT (n_gt,H,W) and predicted (n_pred,H,W) mask layers."""
    n_gt, height, width = gt_layers.shape
    n_pred = pred_layers.shape[0]

    gt_flat = gt_layers.reshape(n_gt, height * width)
    pred_flat = pred_layers.reshape(n_pred, height * width)

    intersection = torch.matmul(gt_flat, pred_flat.t())

    gt_areas = gt_flat.sum(dim=1).view(-1, 1)
    pred_areas = pred_flat.sum(dim=1).view(1, -1)

    union = gt_areas + pred_areas - intersection

    iou_matrix = torch.where(union == 0, torch.tensor(0., device=gt_layers.device), intersection / union)
    dice_matrix = torch.where(union == 0, torch.tensor(0., device=gt_layers.device),
                               2 * intersection / (gt_areas + pred_areas))
    return iou_matrix, dice_matrix


def fp_count_hit(hit_matrix: torch.Tensor) -> int:
    """Predictions (columns) with no qualifying GT overlap."""
    return (hit_matrix.sum(dim=0) == 0).sum().item()


def fn_count_hit(hit_matrix: torch.Tensor) -> int:
    """GT filaments (rows) with no qualifying predicted overlap."""
    return (hit_matrix.sum(dim=1) == 0).sum().item()


def rles_to_layers(rles: list, height: int = 2048, width: int = 2048) -> np.ndarray:
    """Decode compressed COCO RLE strings into an (n_masks, H, W) binary stack."""
    if not rles:
        return np.zeros((0, height, width), dtype=np.float32)
    rle_dicts = [{"size": [height, width], "counts": rle} for rle in rles]
    masks = mask_util.decode(rle_dicts)  # (H, W, n_masks)
    return masks.transpose(2, 0, 1).astype(np.float32)


def process_entry(annotator_image, gt_annotator_image_ids, pred_image_ids, gt_df, pred_df):
    image_id = annotator_image.split("-", maxsplit=1)[1]

    gt_rles = gt_df.loc[gt_annotator_image_ids == annotator_image, "segmentation_rle"].tolist()
    gt_layers = torch.from_numpy(rles_to_layers(gt_rles))

    pred_rles = pred_df.loc[pred_image_ids == image_id, "segmentation_rle"].tolist()
    pred_layers = torch.from_numpy(rles_to_layers(pred_rles))

    n_gt, n_pred = len(gt_rles), len(pred_rles)
    iou_matrix, dice_matrix = get_overlap_matrices(gt_layers, pred_layers)
    return iou_matrix, dice_matrix, n_gt, n_pred


def get_overlap_df(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    gt_annotator_image_ids = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_image_ids = pred_df["filament_id"].str.split("_", n=1).str[0]

    annotator_images = gt_annotator_image_ids.unique()
    overlap_df = pd.DataFrame({"annotator_image": annotator_images})

    results = overlap_df["annotator_image"].apply(
        process_entry,
        gt_annotator_image_ids=gt_annotator_image_ids,
        pred_image_ids=pred_image_ids,
        gt_df=gt_df,
        pred_df=pred_df,
    )
    (overlap_df["iou_matrix"], overlap_df["dice_matrix"],
     overlap_df["n_gt"], overlap_df["n_pred"]) = zip(*results)
    return overlap_df


def get_pq_score(overlap_df: pd.DataFrame) -> float:
    iou_threshold = 0.5
    tp_iou_scores, fp_count, fn_count = [], 0, 0

    for row in overlap_df.itertuples(index=False):
        iou_matrix, n_gt, n_pred = row.iou_matrix, row.n_gt, row.n_pred
        if n_gt == 0:
            fp_count += n_pred
            continue
        if n_pred == 0:
            fn_count += n_gt
            continue
        hit_matrix = iou_matrix > iou_threshold
        tp_iou_scores.extend(iou_matrix[hit_matrix].tolist())
        fp_count += fp_count_hit(hit_matrix)
        fn_count += fn_count_hit(hit_matrix)

    tp_count = len(tp_iou_scores)
    denominator = tp_count + 0.5 * fp_count + 0.5 * fn_count
    return (sum(tp_iou_scores) / denominator) if denominator > 0 else 0.0


def plot_distribution(matrices: pd.Series, title: str, x_label: str):
    scores = np.concatenate([np.asarray(m, dtype=float).ravel() for m in matrices] or [np.zeros(0)])
    scores = scores[scores > 0]
    x_max = 1.0 if scores.size == 0 or scores.max() <= 1.0 else scores.max()
    if x_max <= 1.0:
        bins = np.linspace(0.0, x_max, 51)
        bar_width = 0.02
    else:
        bins = np.arange(0, int(x_max) + 2)
        bar_width = 0.8
    counts, edges = np.histogram(scores, bins=bins)
    probabilities = counts / counts.sum() if counts.sum() > 0 else counts
    centers = (edges[:-1] + edges[1:]) / 2.0
    plt.bar(centers, probabilities, width=bar_width, edgecolor="black", alpha=0.7, color="navy")
    plt.grid(axis="y", linestyle="--")
    if scores.size > 0:
        mean_score = scores.mean()
        plt.axvline(mean_score, color="crimson", linestyle="--", linewidth=1.5, label=f"Mean = {mean_score:.3f}")
        plt.legend(fontsize=8)
    plt.xlabel(x_label); plt.ylabel("Probability"); plt.title(title)


def plot_m2n_counts(overlap_df: pd.DataFrame, title: str = "Many to One and One to Many Distribution"):
    offset = 0.5
    gt_match_counts, pred_match_counts = [], []
    for row in overlap_df.itertuples(index=False):
        iou_matrix, n_gt, n_pred = row.iou_matrix, row.n_gt, row.n_pred
        if n_gt == 0:
            pred_match_counts.extend([0] * n_pred)
            continue
        if n_pred == 0:
            gt_match_counts.extend([0] * n_gt)
            continue
        hit_matrix = iou_matrix > 0
        gt_match_counts.extend(hit_matrix.sum(dim=1).tolist())
        pred_match_counts.extend(hit_matrix.sum(dim=0).tolist())

    gt_degree_counts = Counter(gt_match_counts)
    pred_degree_counts = Counter(pred_match_counts)
    gt_miss_count = gt_degree_counts.get(0, 0)
    gt_hit_count = sum(c for d, c in gt_degree_counts.items() if d > 0)
    pred_miss_count = pred_degree_counts.get(0, 0)
    pred_hit_count = sum(c for d, c in pred_degree_counts.items() if d > 0)

    gt_bars = {d + offset: c for d, c in gt_degree_counts.items()}
    pred_bars = {-(d + offset): c for d, c in pred_degree_counts.items()}
    bar_width = 0.8

    for pos in sorted(gt_bars):
        degree = int(round(pos - offset))
        plt.bar(pos, gt_bars[pos], width=bar_width,
                color="crimson" if degree == 0 else "forestgreen", edgecolor="black", alpha=0.7)
    for pos in sorted(pred_bars):
        degree = int(round(-pos - offset))
        plt.bar(pos, pred_bars[pos], width=bar_width,
                color="darkorange" if degree == 0 else "steelblue", edgecolor="black", alpha=0.7)

    plt.axvline(0, color="black", linewidth=0.8)
    plt.grid(axis="y", linestyle="--")
    plt.xlabel("Matching Case"); plt.ylabel("Frequency")
    positions = sorted(gt_bars) + sorted(pred_bars)
    labels = [f"{abs(int(round(v + offset)))}:1" if v < 0 else f"1:{int(round(v - offset))}" for v in positions]
    plt.xticks(positions, labels, rotation=45, fontsize=9)
    plt.title(title)
    max_pos = max([abs(k) for k in positions], default=offset)
    plt.xlim(-max_pos - bar_width, max_pos + bar_width)
    plt.legend(handles=[
        Patch(facecolor="forestgreen", edgecolor="black", alpha=0.7, label=f"GT Hits (1:n, n>0): {gt_hit_count}"),
        Patch(facecolor="crimson", edgecolor="black", alpha=0.7, label=f"GT Misses (1:0): {gt_miss_count}"),
        Patch(facecolor="darkorange", edgecolor="black", alpha=0.7, label=f"Prediction Misses (0:1): {pred_miss_count}"),
        Patch(facecolor="steelblue", edgecolor="black", alpha=0.7, label=f"Prediction Hits (n:1, n>0): {pred_hit_count}"),
    ], fontsize=9)


def build_val_gt_pred_dataframes(dataset: Dataset, train_img_dir: Path, image_ids,
                                  yolo_model, unet_model, device: torch.device,
                                  crop_cfg, inference_cfg):
    """Ground-truth and prediction dataframes in the exact submission schema
    (filament_id, segmentation_rle), so they can be fed straight into the
    organizers' own scoring functions above."""
    gt_rows = []
    pred_rows_by_stem = {}  # image_stem -> rows; multiple annotator img_ids can
                             # share one physical file, so predict once per file.
    for img_id in image_ids:
        info = dataset.images_by_id[img_id]
        H, W = info["height"], info["width"]
        image_stem = Path(info["file_name"]).stem

        # img_id is already "{annotator_code}-{image_stem}" in this dataset
        # release (see Dataset.__init__), i.e. exactly the "{annotator}-{image_id}"
        # prefix the organizers' self-eval expects for GT filament_id.
        for n, ann in enumerate(dataset.anns_by_image[img_id], start=1):
            mask = polygon_to_mask(ann["segmentation"], H, W)
            gt_rows.append({"filament_id": f"{img_id}_{n}",
                             "segmentation_rle": encode_rle_counts(mask)})

        if image_stem in pred_rows_by_stem:
            continue  # already predicted for this physical image via another annotator's img_id

        label_map, _ = predict_instances(
            train_img_dir / info["file_name"], yolo_model, unet_model, device,
            conf_thres=inference_cfg.conf_thres, iou_thres=inference_cfg.iou_thres,
            imgsz=inference_cfg.imgsz, crop_size=crop_cfg.crop_size,
            pad_factor=crop_cfg.pad_factor, min_crop_side=crop_cfg.min_crop_side,
            mask_thresh=inference_cfg.mask_thresh, min_instance_area=inference_cfg.min_instance_area,
            mask_nms_thresh=inference_cfg.mask_nms_thresh, mask_conf_thres=inference_cfg.mask_conf_thres,
            use_spine_orientation=crop_cfg.use_spine_orientation,
        )
        stem_rows = []
        for inst_id in range(1, int(label_map.max()) + 1):
            mask = label_map == inst_id
            if not mask.any():
                continue
            stem_rows.append({"filament_id": f"{image_stem}_{inst_id}",
                               "segmentation_rle": encode_rle_counts(mask)})
        pred_rows_by_stem[image_stem] = stem_rows

    pred_rows = [row for rows in pred_rows_by_stem.values() for row in rows]
    cols = ["filament_id", "segmentation_rle"]
    return pd.DataFrame(gt_rows, columns=cols), pd.DataFrame(pred_rows, columns=cols)
