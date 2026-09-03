"""Full-disk semantic branch: turns a Flat U-Net probability map into either
(a) an independent per-candidate confidence signal, or (b) standalone instance
candidates from connected components.

(a) is the point. The measured bottleneck is that at a low detector threshold
the YOLO boxes *do* cover ~93% of the ground-truth filaments, but neither the
box confidence nor the refiner's own mask confidence (which saturates above
0.93 inside its own mask, measured) separates the real ones from the noise.
A segmenter that never saw the detector's proposals gives a genuinely
independent opinion on "is there a filament here", which is the missing signal.
"""
from typing import List, Optional

import cv2
import numpy as np
import torch

from solar_filaments.detectors.base import Candidate
from solar_filaments.data.fulldisk_dataset import minmax_normalize
from solar_filaments.models.flat_unet import FlatUNet


def load_flat_unet(weights: str, device) -> tuple:
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    model = FlatUNet(width=ckpt["width"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["imgsz"]


class FullDiskSegmenter:
    """Wraps a trained Flat U-Net. `probability_map` is the full-resolution
    probability used for candidate rescoring; `detect` is the standalone
    instance path (connected components), used for ablation/ensembling."""

    def __init__(self, model, device, imgsz: int = 512, prob_thres: float = 0.5,
                 min_area: int = 25):
        self.model, self.device, self.imgsz = model, device, imgsz
        self.prob_thres, self.min_area = prob_thres, min_area

    @torch.no_grad()
    def probability_map(self, img: np.ndarray) -> np.ndarray:
        """Grayscale full-resolution image -> float32 probability map at the
        same resolution (bilinear upsampling from the network's working size)."""
        H, W = img.shape
        small = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(minmax_normalize(small)).unsqueeze(0).unsqueeze(0).to(self.device)
        prob = torch.sigmoid(self.model(t))[0, 0].float().cpu().numpy()
        return cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)

    def detect(self, img: np.ndarray) -> List[Candidate]:
        """Connected components of the thresholded probability map, each scored
        by its own mean probability."""
        prob = self.probability_map(img)
        binary = (prob > self.prob_thres).astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        out = []
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] < self.min_area:
                continue
            mask = labels == i
            x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                          stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            out.append(Candidate(np.array([x, y, x + w, y + h], dtype=np.float32),
                                 float(prob[mask].mean()), mask))
        return out


def semantic_score(prob_map: Optional[np.ndarray], mask: np.ndarray, y0: int, x0: int) -> float:
    """Mean semantic probability inside a candidate's mask, where `mask` is a
    crop-sized array placed at (y0, x0) in the full image."""
    if prob_map is None or not mask.any():
        return 1.0
    window = prob_map[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]
    return float(window[mask].mean())
