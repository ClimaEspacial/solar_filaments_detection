"""Shared fixtures. Tests that need the MAGFiLO download skip cleanly without it, so the
suite is runnable on a clean checkout."""
import pytest
from solar_filaments.config import Paths


@pytest.fixture(scope="session")
def paths():
    return Paths()


@pytest.fixture(scope="session")
def dataset(paths):
    if not paths.train_ann_path.exists():
        pytest.skip(f"MAGFiLO annotations not found at {paths.train_ann_path}")
    from solar_filaments.data.coco import Dataset
    return Dataset(paths.train_ann_path)
