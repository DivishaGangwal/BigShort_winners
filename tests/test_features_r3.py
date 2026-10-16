import numpy as np
import pandas as pd
import pytest

from src.data_r2 import REPO_ROOT
from src.features_r3 import build_features_r3, reconstruct_close

CSV3 = REPO_ROOT / "KESTREL_EM_D01_round3.csv"
pytestmark = pytest.mark.skipif(not CSV3.exists(), reason="round-three file not present")


@pytest.mark.parametrize("t", [851, 855, 870, 899])
def test_rebuilt_close_and_features_ignore_later_rows(t):
    raw = pd.read_csv(CSV3)
    base_f, base_r = build_features_r3(raw)
    pert = raw.copy()
    later = pert["timestamp"] > t
    for c in ["open", "high", "low", "sma_10", "sma_30", "momentum_5", "volatility_20", "rsi_14", "volume_z_20", "order_flow"]:
        pert.loc[later, c] = pert.loc[later, c] * 1.37 + 3
    f, r = build_features_r3(pert)
    pd.testing.assert_frame_equal(base_f.loc[:t], f.loc[:t])
    np.testing.assert_array_equal(base_r["close"][:t], r["close"][:t])
    f2, _ = build_features_r3(raw.loc[raw["timestamp"] <= t])
    pd.testing.assert_frame_equal(base_f.loc[:t], f2)


def test_rebuild_matches_known_closes_801_850():
    raw = pd.read_csv(CSV3)
    hidden = raw.copy()
    hidden.loc[hidden["timestamp"] > 800, "close"] = np.nan
    rec = reconstruct_close(hidden.loc[hidden["timestamp"] <= 850])
    act = raw.loc[raw["timestamp"] <= 850, "close"].to_numpy()
    assert np.max(np.abs(rec["close"][800:] - act[800:])) < 0.05
