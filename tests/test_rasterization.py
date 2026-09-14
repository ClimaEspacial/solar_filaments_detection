"""The polygon rasterization convention.

This is the single detail that changed our reported metric most, and it is invisible unless
tested: `cv2.fillPoly` and `pycocotools` disagree on the boundary, systematically and worst
on thin structures. The evaluator uses pycocotools, so `polygon_to_mask` must too.
"""
import numpy as np
import pytest
from pycocotools import mask as mask_util

from solar_filaments.data.coco import polygon_to_mask


def _coco_reference(poly, h, w):
    rles = mask_util.frPyObjects([list(map(float, poly))], h, w)
    return mask_util.decode(mask_util.merge(rles)).astype(bool)


@pytest.mark.parametrize("poly", [
    [10.0, 10.0, 40.0, 12.0, 70.0, 30.0, 68.0, 34.0, 38.0, 18.0, 9.0, 15.0],   # thin diagonal
    [20.5, 20.5, 60.5, 22.5, 62.5, 60.5, 18.5, 58.5],                          # blob, half-pixel
])
def test_matches_pycocotools_exactly(poly):
    got = polygon_to_mask(poly, 96, 96).astype(bool)
    assert np.array_equal(got, _coco_reference(poly, 96, 96))


def test_accepts_nested_polygon_format():
    """COCO segmentations arrive either flat or wrapped in a list."""
    flat = [10.0, 10.0, 40.0, 10.0, 40.0, 40.0, 10.0, 40.0]
    assert np.array_equal(polygon_to_mask(flat, 64, 64), polygon_to_mask([flat], 64, 64))


def test_disagrees_with_opencv_and_opencv_is_larger():
    """Guards the finding itself: if this ever stops holding, the two conventions have
    converged and the caveat in the report is stale."""
    import cv2
    poly = [10.0, 10.0, 40.0, 12.0, 70.0, 30.0, 68.0, 34.0, 38.0, 18.0, 9.0, 15.0]
    coco = polygon_to_mask(poly, 96, 96).astype(bool)
    ocv = np.zeros((96, 96), np.uint8)
    cv2.fillPoly(ocv, [np.asarray(poly, np.float64).reshape(-1, 2).round().astype(np.int32)], 1)
    ocv = ocv.astype(bool)
    assert not np.array_equal(coco, ocv)
    assert ocv.sum() > coco.sum()
