"""Round-two folds: seven expanding development folds, the reserved check and the final fit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

# (name, train_end, purged, eval_start, eval_end); training starts at the first sample row.
DEV_FOLDS = [
    ("dev1", 449, 450, 451, 500), ("dev2", 499, 500, 501, 550), ("dev3", 549, 550, 551, 600),
    ("dev4", 599, 600, 601, 650), ("dev5", 649, 650, 651, 700), ("dev6", 699, 700, 701, 750),
    ("dev7", 749, 750, 751, 800),
]
RESERVED_FOLD = ("reserved", 799, 800, 801, 850)
FINAL_FOLD = ("final", 850, None, 851, 900)
MANIFEST_VERSION_R2 = "round2_folds_v1"


def fold_ts(spec, sample_ts) -> dict:
    name, train_end, purged, e0, e1 = spec
    s = [int(t) for t in sample_ts]
    return {"name": name, "train_range": [min(s), train_end], "purged": purged, "eval_range": [e0, e1],
            "train_ts": [t for t in s if t <= train_end], "eval_ts": [t for t in s if e0 <= t <= e1]}


def manifest_bytes(dev_folds: list[dict], first_sample_ts: int, feature_version: str, warmup: dict) -> bytes:
    doc = {
        "manifest_version": MANIFEST_VERSION_R2, "feature_version": feature_version,
        "first_sample_timestamp": first_sample_ts, "warmup": warmup, "purge_rows": 1,
        "dev_folds": dev_folds,
        "reserved": {"train_range": [first_sample_ts, RESERVED_FOLD[1]], "purged": RESERVED_FOLD[2],
                     "eval_range": [RESERVED_FOLD[3], RESERVED_FOLD[4]]},
        "final": {"train_range": [first_sample_ts, FINAL_FOLD[1]], "predict_range": [FINAL_FOLD[3], FINAL_FOLD[4]]},
    }
    return (json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_manifest(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f"{path} exists with different contents; refusing to overwrite")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()
