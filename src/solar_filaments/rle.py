"""pycocotools compressed-RLE encode/decode helpers, matching the submission
schema exactly (counts only, no `size`, fixed 2048x2048 per image)."""
import numpy as np
from pycocotools import mask as mask_util


def encode_rle_counts(binary_mask: np.ndarray) -> str:
    """HxW {0,1} mask -> pycocotools compressed-RLE 'counts' string."""
    rle = mask_util.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    counts = rle["counts"]
    return counts.decode("utf-8") if isinstance(counts, bytes) else counts


def decode_rle_counts(counts: str, height: int, width: int) -> np.ndarray:
    return mask_util.decode({"size": [height, width], "counts": counts})
