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
paired with a U-Net trained the same way (see data/crop_dataset.py). A rough
mask that's too small or too round (elongation below `min_elongation`) gives
an unreliable angle estimate, so those candidates fall back to the plain
axis-aligned mask instead of compounding noise through a bad rotation.
"""
from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import numpy as np
import torch
from ultralytics.utils.ops import xyxyxyxy2xywhr

from solar_filaments.data.coco import load_image
from solar_filaments.data.crop_dataset import letterbox_resize, padded_rect_crop, undo_letterbox
from solar_filaments.data.oriented_crop import (apply_affine_points, clip_patch_to_image,
                                                 crop_patch_with_padding, patch_half_size,
                                                 pca_angle_and_elongation, rotation_matrix)


def _segment(unet_model, device, crop: np.ndarray, crop_size: int, mask_thresh: float):
    """Letterbox `crop`, run the U-Net, return (mask_letterboxed, prob, scale, left, top)."""
    crop_letterboxed, scale, left, top = letterbox_resize(crop, crop_size)
    crop_t = torch.from_numpy(crop_letterboxed).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    logits = unet_model(crop_t)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask_letterboxed = (prob > mask_thresh).astype(np.uint8)
    return mask_letterboxed, prob, scale, left, top


_TTA_FLIPS = ("identity", "hflip", "vflip", "hvflip")


def _apply_flip(arr: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return arr
    if transform == "hflip":
        return arr[:, ::-1]
    if transform == "vflip":
        return arr[::-1, :]
    if transform == "hvflip":
        return arr[::-1, ::-1]
    raise ValueError(f"unknown flip transform: {transform!r}")


def _segment_tta(unet_model, device, crop: np.ndarray, crop_size: int, mask_thresh: float,
                  area_ratio_min: float = 0.6, area_ratio_max: float = 1.7,
                  dice_min: float = 0.62, min_control_area: int = 32):
    """Like _segment, but averages the U-Net's probability map over 4 flips
    (identity/h/v/hv - each is its own inverse, so un-flipping the output
    just re-applies the same transform) and only trusts that average if it
    doesn't change the single-view mask's topology: area within
    [area_ratio_min, area_ratio_max] of the single-view mask, connected-
    component count not increased by more than 1 (no new fragmentation), and
    (once the single-view mask is large enough to make Dice meaningful) Dice
    overlap >= dice_min against it. Otherwise falls back to the plain
    single-view result. Cheap (extra forward passes only, no retraining) and
    orthogonal to every other change in this pipeline - flipping doesn't
    change letterbox_resize's scale/left/top (they depend only on crop
    shape), so the returned (scale, left, top) always match the single-view
    ones a caller already expects from _segment."""
    single_mask = single_prob = scale = left = top = None
    prob_sum = None
    for transform in _TTA_FLIPS:
        flipped_crop = _apply_flip(crop, transform)
        mask_lb, prob, s, l, t = _segment(unet_model, device, flipped_crop, crop_size, mask_thresh)
        prob = _apply_flip(prob, transform)
        if transform == "identity":
            single_mask = (prob > mask_thresh).astype(np.uint8)
            single_prob, scale, left, top = prob, s, l, t
        prob_sum = prob if prob_sum is None else prob_sum + prob

    tta_prob = prob_sum / len(_TTA_FLIPS)
    tta_mask = (tta_prob > mask_thresh).astype(np.uint8)

    single_area = int(single_mask.sum())
    tta_area = int(tta_mask.sum())
    if single_area == 0 or tta_area == 0:
        return single_mask, single_prob, scale, left, top

    area_ratio = tta_area / single_area
    if not (area_ratio_min <= area_ratio <= area_ratio_max):
        return single_mask, single_prob, scale, left, top

    single_components = cv2.connectedComponents(single_mask, connectivity=8)[0] - 1
    tta_components = cv2.connectedComponents(tta_mask, connectivity=8)[0] - 1
    if tta_components > max(single_components + 1, 2):
        return single_mask, single_prob, scale, left, top

    if single_area >= min_control_area:
        intersection = int(np.logical_and(single_mask > 0, tta_mask > 0).sum())
        denom = single_area + tta_area
        dice = 2 * intersection / denom if denom else 1.0
        if dice < dice_min:
            return single_mask, single_prob, scale, left, top

    return tta_mask, tta_prob, scale, left, top


def _segment_maybe_tta(unet_model, device, crop: np.ndarray, crop_size: int, mask_thresh: float, use_tta: bool):
    if use_tta:
        return _segment_tta(unet_model, device, crop, crop_size, mask_thresh)
    return _segment(unet_model, device, crop, crop_size, mask_thresh)


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
                       use_spine_orientation: bool = False, min_elongation: float = 1.5,
                       preprocess: bool = False, use_tta: bool = False
                       ) -> Tuple[np.ndarray, List[float]]:
    img = load_image(image_path, preprocess=preprocess)
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
        mask_letterboxed, prob, scale, left, top = _segment_maybe_tta(unet_model, device, crop, crop_size,
                                                                       mask_thresh, use_tta)

        if mask_conf_thres > 0.0:
            covered = mask_letterboxed > 0
            mask_confidence = prob[covered].mean() if covered.any() else 0.0
            if mask_confidence < mask_conf_thres:
                continue

        rough_mask = undo_letterbox(mask_letterboxed, scale, left, top, cx2 - cx1, cy2 - cy1)

        # Decide whether to trust an estimated orientation for this candidate:
        # a rough mask that's too small or too round gives a PCA angle that's
        # noise, not signal - fall back to the plain axis-aligned mask then.
        use_oriented = False
        if use_spine_orientation:
            fg_local = np.argwhere(rough_mask > 0)  # (row, col) = (y, x)
            if fg_local.shape[0] >= min_instance_area:
                fg_full = fg_local[:, ::-1].astype(np.float32) + np.array([cx1, cy1], dtype=np.float32)
                angle, elongation = pca_angle_and_elongation(fg_full)
                use_oriented = elongation >= min_elongation

        if not use_oriented:
            candidate = rough_mask > 0
            if _claim_candidate(label_map, candidate, cy1, cx1, next_id, min_instance_area, mask_nms_thresh):
                instance_scores.append(float(scores[idx]))
                next_id += 1
            continue

        # --- spine-oriented second pass: re-segment rotated so the estimated axis is horizontal ---
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
        mask2_letterboxed, prob2, scale2, left2, top2 = _segment_maybe_tta(unet_model, device, crop2, crop_size,
                                                                            mask_thresh, use_tta)

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


@torch.no_grad()
def predict_instances_obb(image_path: Path, yolo_model, unet_model, device: torch.device,
                           conf_thres: float = 0.25, iou_thres: float = 0.5, imgsz: int = 1024,
                           crop_size: int = 256, pad_factor: float = 1.35, min_crop_side: int = 48,
                           mask_thresh: float = 0.5, min_instance_area: int = 25,
                           mask_nms_thresh: float = 0.4, mask_conf_thres: float = 0.0,
                           use_tta: bool = False
                           ) -> Tuple[np.ndarray, List[float]]:
    """Single-pass instance pipeline for an OBB-trained YOLO: the detector
    already predicts the filament's orientation directly (no bootstrapping
    from a first segmentation pass, unlike the old spine-oriented approach) -
    rotate a local patch by that angle, crop to the box's own known extent,
    segment, then invert the rotation to place the mask back in image
    coordinates. Must be paired with a U-Net trained via
    data/crop_dataset.py::obb_oriented_sample (`--obb` on train-unet)."""
    img = load_image(image_path)
    H, W = img.shape

    det = yolo_model.predict(img, conf=conf_thres, iou=iou_thres, imgsz=imgsz, verbose=False)[0]
    if det.obb is None or len(det.obb) == 0:
        return np.zeros((H, W), dtype=np.uint16), []

    corners = det.obb.xyxyxyxy.cpu().numpy().astype(np.float32)  # (N, 4, 2)
    scores = det.obb.conf.cpu().numpy()
    # canonicalize: predictions aren't guaranteed long-edge/angle-normalized the
    # way training labels are (see ultralytics OBB.xywhr docs) - re-derive via
    # the same function used to build training crops (obb_oriented_sample).
    xywhr = xyxyxyxy2xywhr(corners.reshape(-1, 8))  # (N, 5): cx, cy, w(long), h(short), angle_rad
    order = np.argsort(-scores)

    label_map = np.zeros((H, W), dtype=np.uint16)
    instance_scores = []
    next_id = 1

    for idx in order:
        cx, cy, long_w, short_h, theta = xywhr[idx]
        angle_deg = float(np.degrees(theta))

        half = patch_half_size(max(long_w, short_h))
        patch, off_x, off_y = crop_patch_with_padding(img, cx, cy, half)
        local_center = (cx - off_x, cy - off_y)
        M = rotation_matrix(local_center, angle_deg)
        rotated_patch = cv2.warpAffine(patch, M, (patch.shape[1], patch.shape[0]), flags=cv2.INTER_LINEAR)

        rcx1, rcy1, rcx2, rcy2 = padded_rect_crop(
            local_center[0] - long_w / 2, local_center[1] - short_h / 2, long_w, short_h,
            rotated_patch.shape[1], rotated_patch.shape[0], pad_factor, min_crop_side)
        if rcx2 <= rcx1 or rcy2 <= rcy1:
            continue

        crop = rotated_patch[rcy1:rcy2, rcx1:rcx2]
        mask_letterboxed, prob, scale, left, top = _segment_maybe_tta(unet_model, device, crop, crop_size,
                                                                       mask_thresh, use_tta)

        if mask_conf_thres > 0.0:
            covered = mask_letterboxed > 0
            mask_confidence = prob[covered].mean() if covered.any() else 0.0
            if mask_confidence < mask_conf_thres:
                continue

        mask_full_crop = undo_letterbox(mask_letterboxed, scale, left, top, rcx2 - rcx1, rcy2 - rcy1)
        rotated_patch_mask = np.zeros(rotated_patch.shape, dtype=np.uint8)
        rotated_patch_mask[rcy1:rcy2, rcx1:rcx2] = mask_full_crop

        M_inv = cv2.invertAffineTransform(M)
        patch_mask = cv2.warpAffine(rotated_patch_mask, M_inv, (patch.shape[1], patch.shape[0]),
                                     flags=cv2.INTER_NEAREST)

        sub_candidate, sx1, sy1, sx2, sy2 = clip_patch_to_image(patch_mask, off_x, off_y, H, W)
        candidate = sub_candidate > 0
        if _claim_candidate(label_map, candidate, sy1, sx1, next_id, min_instance_area, mask_nms_thresh):
            instance_scores.append(float(scores[idx]))
            next_id += 1

    return label_map, instance_scores


def coarse_label_map(candidates, H: int, W: int, min_instance_area: int = 25,
                     mask_nms_thresh: float = 0.4) -> Tuple[np.ndarray, List[float]]:
    """Stitch a backend's own coarse masks (no refiner) into an instance map -
    the detector-alone view, used for checkpoint selection and ablation."""
    label_map = np.zeros((H, W), dtype=np.uint16)
    scores, next_id = [], 1
    for c in sorted(candidates, key=lambda c: -c.score):
        if c.coarse_mask is None:
            continue
        if _claim_candidate(label_map, c.coarse_mask, 0, 0, next_id, min_instance_area, mask_nms_thresh):
            scores.append(c.score)
            next_id += 1
    return label_map, scores


@torch.no_grad()
def refine_candidates(img: np.ndarray, candidates, unet_model, device: torch.device,
                      crop_size: int = 256, pad_factor: float = 1.35, min_crop_side: int = 48,
                      mask_thresh: float = 0.5, min_instance_area: int = 25,
                      mask_nms_thresh: float = 0.4, mask_conf_thres: float = 0.0,
                      use_tta: bool = False, fallback_coarse: bool = False
                      ) -> Tuple[np.ndarray, List[float]]:
    """The shared second stage for every detector backend: crop each candidate
    (highest score first) around its tightest known box, segment it with the
    U-Net (optionally flip-TTA), place the mask back, and claim it via
    mask-level NMS. With `fallback_coarse`, a candidate whose refined mask
    comes back empty/too small but which carries a coarse mask from the
    detector is claimed with that coarse mask instead - protects recall on
    faint filaments the refiner is unsure about, at the cost of a coarser
    boundary."""
    H, W = img.shape
    label_map = np.zeros((H, W), dtype=np.uint16)
    scores, next_id = [], 1

    for c in sorted(candidates, key=lambda c: -c.score):
        x1b, y1b, x2b, y2b = c.tight_box()
        cx1, cy1, cx2, cy2 = padded_rect_crop(x1b, y1b, x2b - x1b, y2b - y1b, W, H, pad_factor, min_crop_side)
        if cx2 <= cx1 or cy2 <= cy1:
            continue

        crop = img[cy1:cy2, cx1:cx2]
        mask_lb, prob, scale, left, top = _segment_maybe_tta(unet_model, device, crop, crop_size, mask_thresh, use_tta)

        if mask_conf_thres > 0.0:
            covered = mask_lb > 0
            if (prob[covered].mean() if covered.any() else 0.0) < mask_conf_thres:
                continue

        candidate = undo_letterbox(mask_lb, scale, left, top, cx2 - cx1, cy2 - cy1) > 0
        if candidate.sum() < min_instance_area and fallback_coarse and c.coarse_mask is not None:
            candidate = c.coarse_mask[cy1:cy2, cx1:cx2]
        if _claim_candidate(label_map, candidate, cy1, cx1, next_id, min_instance_area, mask_nms_thresh):
            scores.append(float(c.score))
            next_id += 1

    return label_map, scores


def predict_instances_from_candidates(image_path: Path, detector, unet_model, device: torch.device,
                                      preprocess: bool = False, refine: bool = True, **refine_kwargs
                                      ) -> Tuple[np.ndarray, List[float]]:
    """Backend-agnostic entry point: detector.detect -> refine_candidates (or
    the detector's coarse masks alone with refine=False)."""
    img = load_image(image_path, preprocess=preprocess)
    candidates = detector.detect(img)
    if not refine:
        return coarse_label_map(candidates, *img.shape,
                                min_instance_area=refine_kwargs.get("min_instance_area", 25),
                                mask_nms_thresh=refine_kwargs.get("mask_nms_thresh", 0.4))
    return refine_candidates(img, candidates, unet_model, device, **refine_kwargs)


def predict_instances_ensemble(sources: List[Tuple[Callable[[], Tuple[np.ndarray, List[float]]], float]],
                                mask_nms_thresh: float = 0.3, min_instance_area: int = 25
                                ) -> Tuple[np.ndarray, List[float]]:
    """Merge multiple already-independent detectors into one non-overlapping
    instance map, by applying the same greedy mask-claim procedure each
    individual pipeline already uses internally (_claim_candidate) to the
    *union* of every source's own already-decided instances - so a source's
    instance only loses to another source's if they actually overlap and the
    other has higher (weighted) confidence.

    `sources` is a list of (predict_call, weight) pairs, where predict_call
    is a zero-argument callable returning (label_map, scores) - this lets
    predict_instances/predict_instances_obb (need yolo_model, unet_model,
    device) and classical.detect_filaments_classical (needs a loaded image
    array, no device) all be wrapped uniformly by the caller without changing
    any of their signatures. `weight` rescales a source's confidence before
    ranking - e.g. the classical detector's constant 1.0 "confidence" should
    be given a low weight so it only fills gaps instead of out-competing real
    detections from the learned models.
    """
    candidates = []  # (mask, weighted_score)
    label_shape = None
    for predict_call, weight in sources:
        label_map, scores = predict_call()
        if label_shape is None:
            label_shape = label_map.shape
        for inst_id, score in enumerate(scores, start=1):
            candidates.append((label_map == inst_id, score * weight))

    if label_shape is None:
        return np.zeros((0, 0), dtype=np.uint16), []

    candidates.sort(key=lambda c: -c[1])
    final_label_map = np.zeros(label_shape, dtype=np.uint16)
    accepted_scores: List[float] = []
    next_id = 1
    for mask, score in candidates:
        if _claim_candidate(final_label_map, mask, 0, 0, next_id, min_instance_area, mask_nms_thresh):
            accepted_scores.append(score)
            next_id += 1

    return final_label_map, accepted_scores
