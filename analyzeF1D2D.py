#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rider-ID — Day-split experiment
──────────────────────────────
• Train on 5 laps from Day 1
• Test  on 5 laps from Day 2
• Random-Forest classifier
• Confusion-matrix plot
• Per-class Precision / Recall / F1
• Feature importances
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder


# ─────────────────── stopwatch ──────────────────────────────────────────────
def now() -> float:
    """Return elapsed seconds since first call."""
    if not hasattr(now, "_t0"):
        now._t0 = time.perf_counter()
    return time.perf_counter() - now._t0


# ─────────────────── configuration ──────────────────────────────────────────
ROOT_DIR   = Path("./Segmented")   # root data folder
TRAIN_DAY  = "Day1"                # sub-folder used for training
TEST_DAY   = "Day2"                # sub-folder used for testing

PARTICIPANTS = [f"P{str(i).zfill(2)}" for i in range(1, 16)]  # P01…P15
LAPS         = range(1, 6)                                   # laps 1-5

FEATURES = [
    "brake_value",
    "accel_x", "accel_y", "accel_z",
    "accel_mag",
]


# ─────────────────── helpers ────────────────────────────────────────────────
def to_float(x):
    """Convert value to float, NaN on failure."""
    try:
        return float(x)
    except Exception:
        return np.nan


def load_lap_json(day: str, pid: str, lap: int):
    """Load one lap JSON file."""
    with open(ROOT_DIR / day / pid / f"lap_{lap}.json") as f:
        return json.load(f)


def lap_to_df(lap_data, pid):
    """Flatten one lap (brake + accel) to a DataFrame row set."""
    rows = []
    for d in lap_data:
        acc = d.get("userAccel") or d.get("acceleration") or {}
        ax, ay, az = map(to_float, (acc.get("x"), acc.get("y"), acc.get("z")))
        rows.append(
            dict(
                brake_value=to_float(d.get("brakeData")),
                accel_x=ax,
                accel_y=ay,
                accel_z=az,
                accel_mag=(
                    np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
                    if not np.isnan(ax) and not np.isnan(ay) and not np.isnan(az)
                    else np.nan
                ),
                participant_id=pid,
            )
        )
    return pd.DataFrame(rows)


def build_day_split(train_day: str, test_day: str):
    """Create train/test DataFrames for the two-day split."""
    tr, te = [], []
    for pid in PARTICIPANTS:
        for lp in LAPS:
            tr.append(lap_to_df(load_lap_json(train_day, pid, lp), pid))
            te.append(lap_to_df(load_lap_json(test_day, pid, lp), pid))
    return (
        pd.concat(tr, ignore_index=True),
        pd.concat(te, ignore_index=True),
    )


def encode(tr_df, te_df):
    """Label-encode participant IDs into numeric codes."""
    le = LabelEncoder().fit(tr_df["participant_id"])
    tr_df["pid_code"] = le.transform(tr_df["participant_id"])
    te_df["pid_code"] = le.transform(te_df["participant_id"])
    return tr_df, te_df, le


def fit_rf(X, y):
    """Random-Forest pipeline (median imputation → RF)."""
    rf = RandomForestClassifier(
        n_estimators=300, n_jobs=-1, random_state=42
    )
    return make_pipeline(SimpleImputer(strategy="median"), rf).fit(X, y)


# ─────────────────── main ───────────────────────────────────────────────────
def main():
    print(f"[{now():6.2f}s] ===== START =====")
    print("Features used:", FEATURES)
    print(f"Train day: {TRAIN_DAY}  ·  Test day: {TEST_DAY}")

    # ── assemble data ───────────────────────────────────────────────────────
    tr_df, te_df = build_day_split(TRAIN_DAY, TEST_DAY)
    tr_df, te_df, le = encode(tr_df, te_df)

    Xtr, ytr = tr_df[FEATURES].astype(float), tr_df["pid_code"]
    Xte, yte = te_df[FEATURES].astype(float), te_df["pid_code"]

    # ── fit & predict ───────────────────────────────────────────────────────
    t0 = now()
    model = fit_rf(Xtr, ytr)
    preds = model.predict(Xte)
    print(f"[{now():6.2f}s] model trained & evaluated  (Δ {now()-t0:.2f}s)")

    # ── feature importances ─────────────────────────────────────────────────
    rf_imp = model.named_steps["randomforestclassifier"].feature_importances_
    print("\n=== Random-Forest feature importance ===")
    print(pd.Series(rf_imp, index=FEATURES).round(3).to_string())

    # ── per-class Precision / Recall / F1 ───────────────────────────────────
    prec = precision_score(
        yte, preds, average=None, labels=range(len(le.classes_))
    )
    rec = recall_score(
        yte, preds, average=None, labels=range(len(le.classes_))
    )
    f1s = f1_score(
        yte, preds, average=None, labels=range(len(le.classes_))
    )
    pr_df = pd.DataFrame(
        {"Precision": prec, "Recall": rec, "F1": f1s},
        index=le.classes_,
    ).round(3)

    print("\n=== Per-participant Precision / Recall / F1 ===")
    print(pr_df.to_string())

    # ── overall macro metrics ───────────────────────────────────────────────
    print(
        "\nOverall macro Precision :",
        f"{precision_score(yte, preds, average='macro'):.3f}",
    )
    print(
        "Overall macro Recall    :",
        f"{recall_score(yte, preds, average='macro'):.3f}",
    )
    print(
        "Overall macro F1        :",
        f"{f1_score(yte, preds, average='macro'):.3f}",
    )

    # ── confusion matrix ────────────────────────────────────────────────────
    cm = confusion_matrix(
        yte, preds, labels=range(len(le.classes_))
    )
    fig, ax = plt.subplots(figsize=(9, 9))
    ConfusionMatrixDisplay(
        cm, display_labels=le.classes_
    ).plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format=".0f",
    )
    plt.xticks(rotation=45, ha="right")
    plt.title(
        f"Random-Forest confusion matrix\nTrain {TRAIN_DAY} → Test {TEST_DAY}"
    )
    plt.tight_layout()
    plt.show()

    print(f"\n[{now():6.2f}s] ===== FINISHED =====")


if __name__ == "__main__":
    main()
