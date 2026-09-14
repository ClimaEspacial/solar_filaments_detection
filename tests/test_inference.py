"""Inference helpers must be safe to call directly, not only through the CLI entry points."""
import numpy as np
import torch

from solar_filaments.inference import _segment, _segment_maybe_tta
from solar_filaments.models.refiner import build_refiner


def _model():
    m = build_refiner("base", pretrained=False)
    m.eval()
    return m


def test_segment_works_outside_a_no_grad_block():
    """`model.eval()` does not disable gradient tracking, so a direct call used to raise
    'Can't call numpy() on Tensor that requires grad'."""
    crop = (np.random.default_rng(0).random((120, 300)) * 255).astype(np.uint8)
    mask, prob, scale, left, top = _segment(_model(), torch.device("cpu"), crop, 128, 0.5)
    assert mask.shape == prob.shape == (128, 128)
    assert mask.dtype == np.uint8 and 0.0 <= prob.min() <= prob.max() <= 1.0


def test_flip_tta_works_outside_a_no_grad_block():
    crop = (np.random.default_rng(1).random((200, 200)) * 255).astype(np.uint8)
    mask, prob, *_ = _segment_maybe_tta(_model(), torch.device("cpu"), crop, 128, 0.5, True)
    assert mask.shape == (128, 128)
