"""Full inference pipeline: detect (YOLO) -> crop -> segment (U-Net) -> stitch.

Detections are processed highest-confidence first. A candidate whose mask is
*mostly* already claimed by higher-confidence instances is suppressed
entirely (mask-level NMS) rather than accepted as a fragment - box IoU (what
YOLO's own NMS uses) is a poor proxy for "same object" on long, thin,
diagonal filaments, so duplicate detections of one filament can survive box
NMS and would otherwise show up as extra low-IoU fragments. A candidate that
only partially overlaps an already-placed instance (a genuinely different,
adjacent filament) keeps its free pixels as before.

A candidate is also rejected if the U-Net itself isn't confident about the
region it segmented (mean predicted probability inside its own thresholded
mask below `mask_conf_thres`) - this decouples recall from YOLO's box
confidence: sweeping `conf_thres` alone forces a single trade-off between
recall and precision (lower conf_thres always let more noise in than it
recovered real filaments, see notebook history), but a genuine filament
crop, even from a lower-confidence box, tends to give the U-Net a decisively
confident mask, while a spurious box (background mistaken for a filament)
tends to give a weak, wishy-washy one. Filtering on that lets `conf_thres`
be relaxed without paying the full precision cost.

With `use_spine_orientation=True`, each candidate gets a second pass: the
first (axis-aligned) crop's rough mask is used only to estimate the
filament's orientation (its principal axis), then a second crop - rotated so
that axis lies horizontal - is re-segmented for the final mask. This must be
paired with a U-Net trained the same way (see data/crop_dataset.py).
"""
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

from solar_filaments.data.coco import load_image
from solar_filaments.data.crop_dataset import letterbox_resize, padded_rect_crop, undo_letterbox
from solar_filaments.data.oriented_crop import (apply_affine_points, clip_patch_to_image,
                                                 crop_patch_with_padding, patch_half_size,
                                                 pca_angle_deg, rotation_matrix)


def _segment(unet_model, device, crop: np.ndarray, crop_size: int, mask_thresh: float):
    """Letterbox `crop`, run the U-Net, return (mask_letterboxed, prob, scale, left, top)."""
    crop_letterboxed, scale, left, top = letterbox_resize(crop, crop_size)
    crop_t = torch.from_numpy(crop_letterboxed).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    logits = unet_model(crop_t)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask_letterboxed = (prob > mask_thresh).astype(np.uint8)
    return mask_letterboxed, prob, scale, left, top


def _claim_candidate(label_map: np.ndarray, candidate: np.ndarray, sy1: int, sx1: int,
                      next_id: int, min_instance_area: int, mask_nms_thresh: float) -> bool:
    """Try to claim `candidate` (a bool array) at label_map[sy1:.., sx1:..] as
    instance `next_id`. Returns whether it was accepted (mutates label_map)."""
    candidate_area = candidate.sum()
    if candidate_area < min_instance_area:
        return False

    sy2, sx2 = sy1 + candidate.shape[0], sx1 + candidate.shape[1]
    region = label_map[sy1:sy2, sx1:sx2]
    already_claimed = candidate & (region != 0)
    if already_claimed.sum() / candidate_area > mask_nms_thresh:
        return False

    free = candidate & (region == 0)
    if free.sum() < min_instance_area:
        return False
    region[free] = next_id
    label_map[sy1:sy2, sx1:sx2] = region
    return True


@torch.no_grad()
def predict_instances(image_path: Path, yolo_model, unet_model, device: torch.device,
                       conf_thres: float = 0.25, iou_thres: float = 0.5, imgsz: int = 1024,
                       crop_size: int = 256, pad_factor: float = 1.35, min_crop_side: int = 48,
                       mask_thresh: float = 0.5, min_instance_area: int = 25,
                       mask_nms_thresh: float = 0.4, mask_conf_thres: float = 0.0,
                       use_spine_orientation: bool = False
                       ) -> Tuple[np.ndarray, List[float]]:
    img = load_image(image_path)
    H, W = img.shape

    det = yolo_model.predict(img, conf=conf_thres, iou=iou_thres, imgsz=imgsz, verbose=False)[0]
    if det.boxes is None or len(det.boxes) == 0:
        return np.zeros((H, W), dtype=np.uint16), []

    boxes = det.boxes.xyxy.cpu().numpy()
    scores = det.boxes.conf.cpu().numpy()
    order = np.argsort(-scores)

    label_map = np.zeros((H, W), dtype=np.uint16)
    instance_scores = []
    next_id = 1

    for idx in order:
        x1b, y1b, x2b, y2b = boxes[idx]
        x, y, w, h = x1b, y1b, x2b - x1b, y2b - y1b
        cx1, cy1, cx2, cy2 = padded_rect_crop(x, y, w, h, W, H, pad_factor, min_crop_side)
        if cx2 <= cx1 or cy2 <= cy1:
            continue

        crop = img[cy1:cy2, cx1:cx2]
        mask_letterboxed, prob, scale, left, top = _segment(unet_model, device, crop, crop_size, mask_thresh)

        if mask_conf_thres > 0.0:
            covered = mask_letterboxed > 0
            mask_confidence = prob[covered].mean() if covered.any() else 0.0
            if mask_confidence < mask_conf_thres:
                continue

        if not use_spine_orientation:
            mask_full_crop = undo_letterbox(mask_letterboxed, scale, left, top, cx2 - cx1, cy2 - cy1)
            candidate = mask_full_crop > 0
            if _claim_candidate(label_map, candidate, cy1, cx1, next_id, min_instance_area, mask_nms_thresh):
                instance_scores.append(float(scores[idx]))
                next_id += 1
            continue

        # --- spine-oriented second pass: estimate angle from the rough mask, re-segment rotated ---
        rough_mask = undo_letterbox(mask_letterboxed, scale, left, top, cx2 - cx1, cy2 - cy1)
        fg_local = np.argwhere(rough_mask > 0)  # (row, col) = (y, x)
        if fg_local.shape[0] < min_instance_area:
            continue
        fg_full = fg_local[:, ::-1].astype(np.float32) + np.array([cx1, cy1], dtype=np.float32)  # -> (x, y)

        angle = pca_angle_deg(fg_full)
        centroid = fg_full.mean(axis=0)
        diag = float(np.linalg.norm(fg_full.max(axis=0) - fg_full.min(axis=0)))
        half = patch_half_size(diag)
        patch, off_x, off_y = crop_patch_with_padding(img, centroid[0], centroid[1], half)

        local_center = (centroid[0] - off_x, centroid[1] - off_y)
        M = rotation_matrix(local_center, angle)
        rotated_patch = cv2.warpAffine(patch, M, (patch.shape[1], patch.shape[0]), flags=cv2.INTER_LINEAR)

        fg_rot = apply_affine_points(fg_full - np.array([off_x, off_y], dtype=np.float32), M)
        bx1, by1 = fg_rot.min(axis=0)
        bx2, by2 = fg_rot.max(axis=0)
        rcx1, rcy1, rcx2, rcy2 = padded_rect_crop(bx1, by1, bx2 - bx1, by2 - by1,
                                                   rotated_patch.shape[1], rotated_patch.shape[0],
                                                   pad_factor, min_crop_side)
        if rcx2 <= rcx1 or rcy2 <= rcy1:
            continue

        crop2 = rotated_patch[rcy1:rcy2, rcx1:rcx2]
        mask2_letterboxed, prob2, scale2, left2, top2 = _segment(unet_model, device, crop2, crop_size, mask_thresh)

        if mask_conf_thres > 0.0:
            covered2 = mask2_letterboxed > 0
            mask_confidence2 = prob2[covered2].mean() if covered2.any() else 0.0
            if mask_confidence2 < mask_conf_thres:
                continue

        mask2_full_crop = undo_letterbox(mask2_letterboxed, scale2, left2, top2, rcx2 - rcx1, rcy2 - rcy1)
        rotated_patch_mask = np.zeros(rotated_patch.shape, dtype=np.uint8)
        rotated_patch_mask[rcy1:rcy2, rcx1:rcx2] = mask2_full_crop

        M_inv = cv2.invertAffineTransform(M)
        patch_mask = cv2.warpAffine(rotated_patch_mask, M_inv, (patch.shape[1], patch.shape[0]),
                                     flags=cv2.INTER_NEAREST)

        sub_candidate, sx1, sy1, sx2, sy2 = clip_patch_to_image(patch_mask, off_x, off_y, H, W)
        candidate = sub_candidate > 0
        if _claim_candidate(label_map, candidate, sy1, sx1, next_id, min_instance_area, mask_nms_thresh):
            instance_scores.append(float(scores[idx]))
            next_id += 1

    return label_map, instance_scores
