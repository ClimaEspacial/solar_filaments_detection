"""Small image-array augmentations shared by the crop (U-Net) and instance
(Mask R-CNN) training datasets, so both stages see the same photometric jitter."""
import random

import numpy as np


def random_gamma(img: np.ndarray, low: float = 0.5, high: float = 1.5, p: float = 0.9) -> np.ndarray:
    """Gamma correction with gamma ~ U(low, high), applied with probability p.
    A nonlinear brightness jitter (unlike a linear gain/bias) - trains invariance
    to the limb-darkening / exposure / seeing variation real across this
    dataset; range and probability follow "A Modern ConvNet for Solar Filament
    Detection" (arXiv:2607.24525, Table 1)."""
    if random.random() >= p:
        return img
    gamma = random.uniform(low, high)
    normalized = img.astype(np.float32) / 255.0
    return np.clip((normalized ** gamma) * 255.0, 0, 255).astype(np.uint8)
