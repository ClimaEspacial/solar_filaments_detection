"""Detector backends share one output type so the refine/stitch stage
(inference.refine_candidates) doesn't care which model proposed a filament."""
from dataclasses import dataclass
from typing import List, Optional, Protocol

import numpy as np


@dataclass
class Candidate:
    box_xyxy: np.ndarray                 # (4,) float, full-resolution pixel coords
    score: float
    coarse_mask: Optional[np.ndarray] = None  # full-resolution bool mask, when the backend predicts one

    def tight_box(self) -> np.ndarray:
        """The coarse mask's own bbox when there is one (tighter and better
        centered than a regressed box), else the detector box."""
        if self.coarse_mask is not None and self.coarse_mask.any():
            ys, xs = np.where(self.coarse_mask)
            return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)
        return self.box_xyxy


class Detector(Protocol):
    def detect(self, img: np.ndarray) -> List[Candidate]:
        """Grayscale uint8 full-resolution image -> candidates (any order)."""
        ...
