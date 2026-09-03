"""Ultralytics backends: axis-aligned boxes (`yolo11*.pt`) and instance
segmentation (`yolo11*-seg.pt`)."""
from typing import List

import cv2
import numpy as np

from solar_filaments.detectors.base import Candidate


class YoloBoxDetector:
    """Boxes only - the original pipeline's detector, as a backend."""

    def __init__(self, model, conf_thres: float = 0.35, iou_thres: float = 0.5, imgsz: int = 1024):
        self.model, self.conf_thres, self.iou_thres, self.imgsz = model, conf_thres, iou_thres, imgsz

    def detect(self, img: np.ndarray) -> List[Candidate]:
        det = self.model.predict(img, conf=self.conf_thres, iou=self.iou_thres, imgsz=self.imgsz, verbose=False)[0]
        if det.boxes is None or len(det.boxes) == 0:
            return []
        boxes = det.boxes.xyxy.cpu().numpy()
        scores = det.boxes.conf.cpu().numpy()
        return [Candidate(b.astype(np.float32), float(s)) for b, s in zip(boxes, scores)]


class YoloSegDetector:
    """Instance masks from a YOLO-seg model. `masks.xy` gives each instance's
    polygon already in original-image pixel coordinates, so the coarse mask is
    rasterized at full resolution directly (no letterbox bookkeeping)."""

    def __init__(self, model, conf_thres: float = 0.35, iou_thres: float = 0.5, imgsz: int = 1024):
        self.model, self.conf_thres, self.iou_thres, self.imgsz = model, conf_thres, iou_thres, imgsz

    def detect(self, img: np.ndarray) -> List[Candidate]:
        det = self.model.predict(img, conf=self.conf_thres, iou=self.iou_thres, imgsz=self.imgsz, verbose=False)[0]
        if det.boxes is None or len(det.boxes) == 0 or det.masks is None:
            return []
        H, W = img.shape
        boxes = det.boxes.xyxy.cpu().numpy()
        scores = det.boxes.conf.cpu().numpy()
        out = []
        for b, s, poly in zip(boxes, scores, det.masks.xy):
            mask = np.zeros((H, W), dtype=np.uint8)
            if len(poly) >= 3:
                cv2.fillPoly(mask, [np.asarray(poly, dtype=np.float32).round().astype(np.int32)], 1)
            out.append(Candidate(b.astype(np.float32), float(s), mask.astype(bool) if mask.any() else None))
        return out
