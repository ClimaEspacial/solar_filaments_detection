"""The Panoptic Quality implementation, against cases computable by hand.

PQ = sum(IoU of TPs) / (|TP| + 0.5|FP| + 0.5|FN|), pooled globally rather than averaged per
image. The pooling is the part that is easy to get wrong and impossible to notice.
"""
import numpy as np
import pandas as pd
import pytest

from solar_filaments.eval.self_eval import get_overlap_df, get_pq_score, summarize_overlap
from solar_filaments.rle import encode_rle_counts

H = W = 2048   # o avaliador decodifica RLE nesse tamanho fixo
COLS = ["filament_id", "segmentation_rle"]


def _rect(y0, y1, x0, x1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def _df(rows):
    return pd.DataFrame([{"filament_id": i, "segmentation_rle": encode_rle_counts(m)}
                         for i, m in rows], columns=COLS)


def _pq(gt, pred):
    return get_pq_score(get_overlap_df(_df(gt), _df(pred)))


def test_perfect_prediction_scores_one():
    m = _rect(100, 300, 100, 300)
    assert _pq([("a-img_1", m)], [("img_1", m)]) == pytest.approx(1.0)


def test_empty_prediction_scores_zero():
    assert _pq([("a-img_1", _rect(100, 300, 100, 300))], []) == pytest.approx(0.0)


def test_predictions_on_images_without_reference_are_ignored():
    """A documented gap in the organizers' implementation, kept here so it is not mistaken
    for a bug in ours: `get_overlap_df` iterates over reference images, so a prediction on an
    image carrying no annotation never becomes a false positive. It does not arise in practice
    because every evaluated image is annotated, but it is the same class of gap that the
    public leaderboard evaluator was shown to have in July 2026."""
    m = _rect(100, 300, 100, 300)
    with_extra = _pq([("a-i1_1", m)], [("i1_1", m), ("i2_1", m)])
    without = _pq([("a-i1_1", m)], [("i1_1", m)])
    assert with_extra == pytest.approx(without) == pytest.approx(1.0)


def test_single_false_positive_halves_the_denominator_term():
    """One perfect match plus one spurious prediction: PQ = 1 / (1 + 0.5)."""
    gt = _rect(40, 120, 40, 120)
    got = _pq([("a-img_1", gt)], [("img_1", gt), ("img_2", _rect(400, 500, 400, 500))])
    assert got == pytest.approx(1.0 / 1.5)


def test_overlap_below_half_is_not_a_match():
    """The matching rule is IoU > 0.5; a 40% overlap must count as FP *and* FN, not as a
    weak true positive."""
    gt = _rect(0, 100, 0, 100)        # 10000 px
    pred = _rect(0, 100, 0, 40)       # 4000 px, IoU = 0.4
    assert _pq([("a-img_1", gt)], [("img_1", pred)]) == pytest.approx(0.0)


def test_pq_is_pooled_not_averaged_per_image():
    """One image perfect, one image missed. Pooled PQ is 1/(1+0.5) = 0.667; a per-image mean
    would give 0.5. The organizers pool, and so must we."""
    m = _rect(100, 300, 100, 300)
    got = _pq([("a-i1_1", m), ("a-i2_1", m)], [("i1_1", m)])
    assert got == pytest.approx(1.0 / 1.5)
    assert got != pytest.approx(0.5)


def test_summary_factorises_pq_into_sq_times_rq():
    m, n = _rect(40, 120, 40, 120), _rect(400, 500, 400, 500)
    s = summarize_overlap(get_overlap_df(_df([("a-img_1", m)]), _df([("img_1", m), ("img_2", n)])))
    assert s["PQ"] == pytest.approx(s["SQ"] * s["RQ"])
    assert (s["TP"], s["FP"], s["FN"]) == (1, 1, 0)
