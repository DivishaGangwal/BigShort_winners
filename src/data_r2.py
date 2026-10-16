"""Round-two dataset contract: hash check, schema checks and time cuts."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH_R2 = REPO_ROOT / "KESTREL_EM_D01_round2.csv"
ROUND1_PATH = REPO_ROOT / "KESTREL_EM_D01_round1.csv"

# SHA-256 supplied by the user; it is the hash of the file with CRLF line endings.
EXPECTED_SHA256_R2 = "462729cd85bff30c41bdd59a4ecd929ffa6c4050846b73c6cfa3811da052fb53"

COLUMNS_R2 = [
    "timestamp", "date", "stock_id", "open", "high", "low", "close", "volume",
    "return_1d", "sma_10", "sma_30", "momentum_5", "volatility_20", "rsi_14",
    "volume_z_20", "order_flow", "target",
]
SHAPE_R2 = (900, 17)
LABELLED_R2 = (1, 850)
HIDDEN_R2 = (851, 900)
HIDDEN_COLUMNS_R2 = ["close", "return_1d", "target"]

# The only model inputs (same row or earlier).
ALLOWED_INPUTS_R2 = [
    "open", "high", "low", "volume", "sma_10", "sma_30", "momentum_5",
    "volatility_20", "rsi_14", "volume_z_20", "order_flow",
]
BANNED_INPUTS_R2 = ["close", "return_1d", "target", "timestamp", "date", "stock_id"]

DEV_MAX = 800          # development folds use rows <= 800 only
RESERVED = (801, 850)  # evaluated once, after selection is frozen
PREDICT = (851, 900)   # prediction only


def crlf_sha256(path: Path | str) -> str:
    data = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    return hashlib.sha256(data).hexdigest()


def verify_source_r2(path: Path | str = CSV_PATH_R2) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing")
    raw, crlf = hashlib.sha256(path.read_bytes()).hexdigest(), crlf_sha256(path)
    if crlf != EXPECTED_SHA256_R2:
        raise ValueError(f"round-two hash mismatch: {crlf} != {EXPECTED_SHA256_R2}")
    return {"sha256_file": raw, "sha256_crlf_normalised": crlf}


def check_schema_r2(df: pd.DataFrame) -> None:
    if df.shape != SHAPE_R2:
        raise ValueError(f"expected shape {SHAPE_R2}, got {df.shape}")
    if list(df.columns) != COLUMNS_R2:
        raise ValueError(f"unexpected columns {list(df.columns)}")
    if not (df["timestamp"].to_numpy() == range(1, SHAPE_R2[0] + 1)).all():
        raise ValueError("timestamps must be 1..900, unique and ascending")
    lab = df["timestamp"].between(*LABELLED_R2)
    hid = df["timestamp"].between(*HIDDEN_R2)
    if df.loc[lab].isna().any().any():
        raise ValueError("labelled rows 1-850 must be complete")
    if df.loc[hid, HIDDEN_COLUMNS_R2].notna().any().any():
        raise ValueError("rows 851-900 must hide close, return_1d and target")
    if df.loc[hid].drop(columns=HIDDEN_COLUMNS_R2).isna().any().any():
        raise ValueError("rows 851-900 must show every other column")
    if not df.loc[lab, "target"].isin([0, 1]).all():
        raise ValueError("target must be 0/1")


def load_r2(max_timestamp: int, path: Path | str = CSV_PATH_R2) -> tuple[pd.DataFrame, pd.Series]:
    """Return (inputs, target) for rows <= max_timestamp.

    The file is read once for the schema check; rows after max_timestamp are
    then dropped before anything else. close and return_1d are dropped here
    and never reach a caller. target is returned separately (NaN for 851+).
    """
    verify_source_r2(path)
    df = pd.read_csv(path)
    check_schema_r2(df)
    df = df.loc[df["timestamp"] <= max_timestamp].reset_index(drop=True)
    inputs = df.loc[:, ["timestamp", *ALLOWED_INPUTS_R2]].astype({c: float for c in ALLOWED_INPUTS_R2})
    target = pd.Series(df["target"].to_numpy(), index=pd.Index(df["timestamp"].to_numpy(), name="timestamp"), name="target")
    return inputs, target
