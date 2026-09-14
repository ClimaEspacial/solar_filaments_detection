"""The submission path: RLE encoding, and its survival through CSV.

A corrupted mask here is invisible locally and only shows up as a low leaderboard score, so
it is worth pinning.
"""
import numpy as np
import pandas as pd
import pytest
from pycocotools import mask as mask_util

from solar_filaments.rle import encode_rle_counts


@pytest.fixture
def masks():
    rng = np.random.default_rng(0)
    out = []
    for _ in range(12):
        m = np.zeros((2048, 2048), bool)
        y, x = rng.integers(100, 1800, 2)
        h, w = rng.integers(20, 300, 2)
        m[y:y + h, x:x + w] = True
        out.append(m & (rng.random((2048, 2048)) > 0.15))   # ragged edges, like a real mask
    return out


def test_rle_round_trips_through_pycocotools(masks):
    for m in masks:
        back = mask_util.decode({"size": [2048, 2048], "counts": encode_rle_counts(m).encode()})
        assert np.array_equal(back.astype(bool), m)


def test_rle_survives_a_csv_write_and_read(masks, tmp_path):
    """The counts stream is ASCII but contains characters a careless CSV writer could mangle."""
    df = pd.DataFrame({"filament_id": [f"img_{i}" for i in range(len(masks))],
                       "segmentation_rle": [encode_rle_counts(m) for m in masks]})
    f = tmp_path / "submission.csv"
    df.to_csv(f, index=False)
    back = pd.read_csv(f)
    assert len(back) == len(df)
    for m, s in zip(masks, back["segmentation_rle"]):
        got = mask_util.decode({"size": [2048, 2048], "counts": s.encode()})
        assert np.array_equal(got.astype(bool), m)


def test_empty_mask_encodes_to_zero_area():
    rle = encode_rle_counts(np.zeros((2048, 2048), bool))
    assert int(mask_util.area({"size": [2048, 2048], "counts": rle.encode()})) == 0
