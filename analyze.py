#!/usr/bin/env python3
"""
Same-rider detection (feature-friendly version)
──────────────────────────────────────────────
• Trains novelty models (One-Class SVM, Isolation Forest) on laps 1-4.
• Evaluates every sample in lap 5:  +1 = looks like the same rider, −1 = not.

Feature list lives in `FEATURE_KEYS` – add keys there and re-run.
Nested JSON paths use dot notation, e.g.   "userAccel.x".
"""

# ──────────────── standard imports ────────────────
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

# ─────────────── editable knobs ───────────────
LAP_DIR       = "./Segmented/P02"
TRAIN_LAPS    = (1, 2, 3, 4)
TEST_LAP      = 5

NU            = 0.05   # One-Class SVM  – expected outlier frac in *training*
CONTAMINATION = 0.05   # IsolationForest – expected outlier frac in *training*

FEATURE_KEYS: List[str] = [
    "rel_time",          # ← generated automatically, always available
    # "brakeData",
    # add more keys whenever you like:
    # "cadence",
    # "pedalWeight.L",
    # "pedalWeight.R",
    "userAccel.x",
    "userAccel.y",
    "userAccel.z",
]

# ──────────── JSON-flatten helper ────────────
def _flatten(d: Dict[str, Any], parent: str = "", sep: str = ".") -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            out.update(_flatten(v, key, sep))
        else:
            out[key] = v
    return out

# ── convert **one** record to Series ──
def _record_to_series(rec: Dict[str, Any]) -> pd.Series:
    flat = _flatten(rec)
    flat["rel_time"] = float(rec["unixTimeStamp"]) - _record_to_series.t0
    return pd.Series(flat, dtype="object")

_record_to_series.t0 = 0.0  # will be set per-lap

# ─────────────── I/O helpers ───────────────
def lap_path(idx: int) -> Path:
    return Path(LAP_DIR) / f"lap_{idx}.json"

def load_lap(idx: int) -> pd.DataFrame:
    path = lap_path(idx)
    with open(path) as fp:
        recs = json.load(fp)
    if not recs:
        raise ValueError(f"{path} is empty")

    _record_to_series.t0 = float(recs[0]["unixTimeStamp"])
    df = pd.DataFrame(_record_to_series(r) for r in recs)

    # keep only requested feature cols that actually exist
    present = [k for k in FEATURE_KEYS if k in df.columns]
    return df[present].apply(pd.to_numeric, errors="coerce")  # convert to float / NaN

def build_train_test():
    # concatenate laps 1-4
    train_df = pd.concat([load_lap(i) for i in TRAIN_LAPS], ignore_index=True)
    test_df  = load_lap(TEST_LAP)

    # fill NaNs with column means (computed on training set)
    means = train_df.mean(numeric_only=True)
    train_df = train_df.fillna(means)
    test_df  = test_df.fillna(means)     

    #  Standardize (normalize) the data so all features have the same scale. 
    #  For example, acceleration in meters/sec² and time in seconds are on very different scales
    scaler  = StandardScaler().fit(train_df.values)
    X_train = scaler.transform(train_df.values)
    X_test  = scaler.transform(test_df.values)
    return X_train, X_test

# ──────────── fit both novelty models ────────────
def fit_models(X_train):
    ocsvm = OneClassSVM(kernel="rbf", nu=NU, gamma="scale").fit(X_train)
    iso   = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=42,
    ).fit(X_train)
    return ocsvm, iso

# ───────────── main routine ─────────────
def main():
    # check files exist
    missing = [i for i in (*TRAIN_LAPS, TEST_LAP) if not lap_path(i).exists()]
    if missing:
        raise SystemExit("Missing lap files: " + ", ".join(str(i) for i in missing))

    X_train, X_test = build_train_test()
    ocsvm, iso = fit_models(X_train)


   # How many predictions are "inliers" and take the average 
   # E.g., if 0.97, then 97% of lap 5 looks like the same rider.
    oc_ratio  = (ocsvm.predict(X_test) == 1).mean()
    iso_ratio = (iso.predict(X_test)   == 1).mean()

    print("\nSame-rider detection on lap 5")
    print("─" * 50)
    print(f"One-Class SVM    inlier ratio : {oc_ratio :.3f}")
    print(f"IsolationForest  inlier ratio : {iso_ratio:.3f}")

    def verdict(r: float) -> str:
        return "✅  lap 5 matches SAME rider" if r >= 0.95 else \
               "⚠︎  lap 5 may NOT be same rider"
    print("\nVerdicts")
    print("OC-SVM :", verdict(oc_ratio))
    print("IsoFor :", verdict(iso_ratio))

    print("\nFeatures actually used:", ", ".join(FEATURE_KEYS))

if __name__ == "__main__":
    main()
