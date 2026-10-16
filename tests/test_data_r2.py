import numpy as np
import pandas as pd
import pytest

from src.audit_r2 import verify_round2_facts
from src.data_r2 import ALLOWED_INPUTS_R2, CSV_PATH_R2, load_r2


def test_round2_facts_hold():
    out = verify_round2_facts()
    assert out["problems"] == []
    f = out["facts"]
    assert f["shape"] == [900, 17]
    assert f["target_counts"] == {"ones": 440, "zeros": 410}
    assert f["target_rule_matches"] == f["target_rule_checkable"] == 849
    assert f["rows_1_800_identical_to_round1"]
    assert f["volatility_20"]["pred_rows_853_900_min"] >= 0.0398
    assert f["volatility_20"]["labelled_max"] < 0.021


def test_indicator_formulas_reproduce_dataset_within_rounding():
    check = verify_round2_facts()["facts"]["formula_check"]
    for name, c in check.items():
        assert c["rows_beyond_rounding"] == 0, name
        assert c["rows_exact"] >= c["rows"] - 3, name


def test_hash_check_rejects_modified_file(tmp_path):
    p = tmp_path / "x.csv"
    p.write_bytes(CSV_PATH_R2.read_bytes().replace(b"KESTREL_EM", b"KESTREL_XX", 1))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_r2(800, p)


@pytest.mark.parametrize("cut", [800, 850, 900])
def test_loader_cuts_rows_and_drops_banned_columns(cut):
    X, y = load_r2(cut)
    assert X["timestamp"].max() == cut and len(X) == cut
    assert list(X.columns) == ["timestamp", *ALLOWED_INPUTS_R2]
    assert y.index.max() == cut


def test_dev_load_never_contains_801_onwards(dev):
    X, y = dev
    assert X["timestamp"].max() <= 800 and y.index.max() <= 800
    assert y.notna().all()
