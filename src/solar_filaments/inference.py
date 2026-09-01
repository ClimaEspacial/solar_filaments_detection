"""Full inference pipeline: detect (YOLO) -> crop -> segment (U-Net) -> stitch.

Detections are processed highest-confidence first. A candidate whose mask is
*mostly* already claimed by higher-confidence instances is suppressed
entirely (mask-level NMS) rather than accepted as a fragment - box IoU (what
YOLO's own NMS uses) is a poor proxy for "same object" on long, thin,
diagonal filaments, so duplicate detections of one filament can survive box
NMS and would otherwise show up as extra low-IoU fragments. A candidate that
only partially overlaps an already-placed instance (a genuinely different,
adjacent filament) keeps its free pixels as before.
"""
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from solar_filaments.data.coco import load_image
from solar_filaments.data.crop_dataset import letterbox_resize, padded_rect_crop, undo_letterbox


@torch.no_grad()
def predict_instances(image_path: Path, yolo_model, unet_model, device: torch.device,
                       conf_thres: float = 0.25, iou_thres: float = 0.5, imgsz: int = 1024,
                       crop_size: int = 256, pad_factor: float = 1.35, min_crop_side: int = 48,
                       mask_thresh: float = 0.5, min_instance_area: int = 25,
                       mask_nms_thresh: float = 0.4
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
        crop_letterboxed, scale, left, top = letterbox_resize(crop, crop_size)
        crop_t = torch.from_numpy(crop_letterboxed).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

        logits = unet_model(crop_t)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        mask_letterboxed = (prob > mask_thresh).astype(np.uint8)
        mask_full_crop = undo_letterbox(mask_letterboxed, scale, left, top, cx2 - cx1, cy2 - cy1)

        candidate = mask_full_crop > 0
        candidate_area = candidate.sum()
        if candidate_area < min_instance_area:
            continue

        region = label_map[cy1:cy2, cx1:cx2]
        already_claimed = candidate & (region != 0)
        # Mask-level NMS: if most of this candidate is already covered by a
        # higher-confidence instance, it's a duplicate detection of the same
        # filament (box NMS missed it) - suppress it whole instead of keeping
        # a low-IoU sliver of its free pixels.
        if already_claimed.sum() / candidate_area > mask_nms_thresh:
            continue

        free = candidate & (region == 0)
        if free.sum() < min_instance_area:
            continue
        region[free] = next_id
        label_map[cy1:cy2, cx1:cx2] = region
        instance_scores.append(float(scores[idx]))
        next_id += 1

    return label_map, instance_scores
