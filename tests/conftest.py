import src  # noqa: F401  (thread caps before numerical imports)
import pytest

from src.data_r2 import load_r2
from src.features_r2 import build_features_r2, sample_timestamps


@pytest.fixture(scope="session")
def dev():
    return load_r2(800)


@pytest.fixture(scope="session")
def dev_feats(dev):
    return build_features_r2(dev[0])[0]


@pytest.fixture(scope="session")
def full():
    return load_r2(900)
