"""Split hygiene and annotator-merge targets.

Two bugs in this project's history motivate these. Splitting on annotation-record identifiers
leaked the same photograph into both sides whenever it carried several annotators; and a
dataset rebuild that did not wipe its output left stale validation images that were also in
train.
"""
import pytest

from solar_filaments.data.coco import (ANNOTATOR_MERGE_MODES, load_split, merge_annotator_records,
                                        train_val_split)


def test_split_is_disjoint_at_the_level_of_the_photograph(dataset):
    train, val = train_val_split(dataset, seed=0, val_frac=0.1)
    f = lambda ids: {dataset.images_by_id[i]["file_name"] for i in ids}
    assert not (f(train) & f(val)), "the same photograph appears in both splits"


def test_split_is_deterministic_given_seed_and_fraction(dataset):
    a = train_val_split(dataset, seed=0, val_frac=0.05)
    b = train_val_split(dataset, seed=0, val_frac=0.05)
    assert a == b


@pytest.mark.parametrize("mode", ANNOTATOR_MERGE_MODES)
def test_every_merge_mode_returns_instances(dataset, mode):
    multi = next(fn for fn in {im["file_name"] for im in dataset.images_by_id.values()}
                 if len(dataset.file_name_to_image_ids[fn]) > 1)
    out = merge_annotator_records(dataset, multi, mode)
    assert out, f"{mode} produced no instances"
    assert all("segmentation" in a and "bbox" in a for a in out)


def test_merge_modes_are_ordered_single_le_majority_le_union(dataset):
    """union keeps every clustered filament, majority those at least half the annotators
    marked, so counts must be monotone. This is what makes the annotator-support analysis
    meaningful."""
    files = [fn for fn in sorted({im["file_name"] for im in dataset.images_by_id.values()})
             if len(dataset.file_name_to_image_ids[fn]) == 3][:12]
    if not files:
        pytest.skip("no three-annotator images available")
    for fn in files:
        n = {m: len(merge_annotator_records(dataset, fn, m)) for m in ("single", "majority", "union")}
        assert n["majority"] <= n["union"], f"{fn}: majority {n['majority']} > union {n['union']}"


def test_support_is_a_fraction_and_union_covers_every_level(dataset):
    fn = next(f for f in {im["file_name"] for im in dataset.images_by_id.values()}
              if len(dataset.file_name_to_image_ids[f]) == 3)
    sup = [a["support"] for a in merge_annotator_records(dataset, fn, "union")]
    assert all(0.0 < s <= 1.0 for s in sup)
    assert all(abs(s * 3 - round(s * 3)) < 1e-6 for s in sup), "support must be k/3 on a 3-annotator image"
