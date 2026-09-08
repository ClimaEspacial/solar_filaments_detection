"""Candidate re-scoring: predict how many annotators would have marked a candidate.

Motivation, from measurement rather than intuition. At a low detector threshold the
pipeline's proposals already overlap 93% of the ground-truth filaments, but 28% of them are
spurious, and neither of the two confidences available at that point separates the two:

* the detector's box confidence is the thing we are trying to replace, and
* the refiner's mask confidence is structurally useless - the mask is *defined* as the region
  where probability exceeds the threshold, so the mean over that region necessarily exceeds
  it (measured: filtering at 0.85/0.90/0.93 gives byte-identical output).

Measured single-feature separability of true from false candidates (AUC, fold0 val):
box confidence 0.765, full-disk semantic probability 0.733, mask/surround contrast 0.692,
solidity 0.661, area 0.636. Several of these are independent of the detector score, and
solidity carries a physically sensible prior: real filaments are elongated and irregular
(median solidity 0.634) while false positives are blobbier and more convex (0.779).

The target is deliberately not a binary label. Because Panoptic Quality pools over annotator
records, the value of predicting a filament depends on *how many* annotators marked it: a
candidate matching a cluster with support k/n contributes k true positives and n-k false
positives. The marginal analysis gives a break-even support of PQ / (2 * mean IoU) ~ 0.32 at
our operating point. We therefore regress the annotator support directly and threshold it at
inference, which turns that analysis into an operational decision rule.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

FEATURES = ["box_conf", "sem_mean", "sem_max", "contrast", "area", "thickness",
            "elongation", "solidity", "r_norm", "n_components"]


@dataclass
class DiskGeometry:
    """Per-image quantities reused across all candidates of that image."""
    disk: np.ndarray
    cy: float
    cx: float
    radius: float

    @classmethod
    def from_image(cls, img: np.ndarray) -> "DiskGeometry":
        from solar_filaments.data.preprocess import estimate_disk_mask
        disk = estimate_disk_mask(img)
        ys, xs = np.where(disk)
        cy, cx = float(ys.mean()), float(xs.mean())
        radius = float(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).max())
        return cls(disk, cy, cx, radius)


def candidate_features(img: np.ndarray, mask: np.ndarray, box_conf: float,
                       geom: DiskGeometry, semantic_prob: Optional[np.ndarray] = None,
                       y0: int = 0, x0: int = 0) -> Dict[str, float]:
    """Shape, photometric and context descriptors for one candidate instance.

    `mask` may be a window placed at (y0, x0) in the full image rather than a
    full-resolution array; every quantity is computed inside that window, and only the
    centroid is mapped back to full-image coordinates for the limb distance. Allocating a
    2048x2048 mask per candidate would dominate inference time."""
    h, w = mask.shape
    img = img[y0:y0 + h, x0:x0 + w]
    disk = geom.disk[y0:y0 + h, x0:x0 + w]
    if semantic_prob is not None:
        semantic_prob = semantic_prob[y0:y0 + h, x0:x0 + w]

    area = int(mask.sum())
    if area == 0:
        return {k: 0.0 for k in FEATURES}
    ys, xs = np.where(mask)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea) if contours else None
    perimeter = max(cv2.arcLength(largest, True), 1.0) if largest is not None else 1.0
    hull_area = max(cv2.contourArea(cv2.convexHull(largest)), 1.0) if largest is not None else 1.0

    ring = cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~mask & disk
    contrast = float(img[ring].mean() - img[mask].mean()) if ring.any() else 0.0

    if area > 2:
        cov = np.cov(np.vstack([xs, ys]))
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
        elongation = float(np.sqrt(ev[0] / max(ev[1], 1e-6)))
    else:
        elongation = 1.0

    return {
        "box_conf": float(box_conf),
        "sem_mean": float(semantic_prob[mask].mean()) if semantic_prob is not None else 0.0,
        "sem_max": float(semantic_prob[mask].max()) if semantic_prob is not None else 0.0,
        "contrast": contrast,
        "area": float(area),
        "thickness": 2.0 * area / perimeter,
        "elongation": elongation,
        "solidity": area / hull_area,
        "r_norm": float(np.sqrt((xs.mean() + x0 - geom.cx) ** 2 + (ys.mean() + y0 - geom.cy) ** 2) / geom.radius),
        "n_components": float(cv2.connectedComponents(mask.astype(np.uint8))[0] - 1),
    }


def features_to_array(feats: List[Dict[str, float]]) -> np.ndarray:
    return np.array([[f[k] for k in FEATURES] for f in feats], dtype=np.float32)


class Rescorer:
    """Wraps the fitted regressor plus the decision threshold."""

    def __init__(self, model, threshold: float = 0.32):
        self.model, self.threshold = model, threshold

    def predict_support(self, feats: List[Dict[str, float]]) -> np.ndarray:
        if not feats:
            return np.zeros(0, dtype=np.float32)
        return self.model.predict(features_to_array(feats))

    def keep(self, feats: List[Dict[str, float]]) -> np.ndarray:
        return self.predict_support(feats) >= self.threshold

    def save(self, path: Path) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "threshold": self.threshold, "features": FEATURES}, f)

    @classmethod
    def load(cls, path: Path, threshold: Optional[float] = None) -> "Rescorer":
        import pickle
        with open(path, "rb") as f:
            blob = pickle.load(f)
        if blob.get("features") != FEATURES:
            raise ValueError(f"rescorer was fitted on different features: {blob.get('features')}")
        return cls(blob["model"], blob["threshold"] if threshold is None else threshold)


def fit_rescorer(feats: List[Dict[str, float]], support: np.ndarray, seed: int = 0):
    """Gradient-boosted regression of annotator support on candidate descriptors.

    Early stopping on an internal split is not optional here: with only ~10 features and a
    few thousand candidates, an unregularized booster reaches an in-sample AUC of 0.998,
    which measures memorization rather than skill."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=3,
                                          l2_regularization=1.0, min_samples_leaf=30,
                                          early_stopping=True, validation_fraction=0.2,
                                          n_iter_no_change=25, random_state=seed)
    model.fit(features_to_array(feats), support)
    return model
