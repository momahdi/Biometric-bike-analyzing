#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Five-fold rider-ID model with brake + acceleration features,
side-by-side F1 tables, and an overall confusion-matrix plot.
"""

import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


# ───────────────────────── stopwatch ────────────────────────────────────────
def now() -> float:
    if not hasattr(now, "_t0"):
        now._t0 = time.perf_counter()
    return time.perf_counter() - now._t0


# ─────────────────────── configuration section ──────────────────────────────
BASE_DIR     = Path("./Segmented")                     # root folder of laps
PARTICIPANTS = [f"P{str(i).zfill(2)}" for i in range(1, 16)]   # P01 … P15

FOLDS = [([1, 2, 3, 4], 5),
         ([2, 3, 4, 5], 1),
         ([3, 4, 5, 1], 2),
         ([4, 5, 1, 2], 3),
         ([5, 1, 2, 3], 4)]

FOLD_NAMES = [f"F{idx}" for idx in range(1, len(FOLDS) + 1)]

FEATURES = ["timestamp", "brake_value",
            "accel_x", "accel_y", "accel_z", "accel_mag"]


# ───────────────────────── helper functions ─────────────────────────────────
def to_float(x):
    """Return float(x) or np.nan if empty / invalid."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load_lap_json(pid: str, lap: int):
    with open(BASE_DIR / pid / f"lap_{lap}.json") as f:
        return json.load(f)


def lap_to_df(lap_data, pid):
    """
    Convert one lap (list of dicts) to a DataFrame with:

      • timestamp
      • brake_value
      • accel_(x|y|z)
      • accel_mag = √(x² + y² + z²)
    """
    rows = []
    for d in lap_data:
        acc_src = d.get("userAccel") or d.get("acceleration") or {}
        ax = to_float(acc_src.get("x"))
        ay = to_float(acc_src.get("y"))
        az = to_float(acc_src.get("z"))

        rows.append(
            {
                "timestamp"   : to_float(d.get("unixTimeStamp")),
                "brake_value" : to_float(d.get("brakeData")),
                "accel_x"     : ax,
                "accel_y"     : ay,
                "accel_z"     : az,
                "accel_mag"   : np.sqrt(ax**2 + ay**2 + az**2)
                                if not np.isnan(ax) and not np.isnan(ay) and not np.isnan(az)
                                else np.nan,
                "participant_id": pid,
            }
        )
    return pd.DataFrame(rows)


def build(train_laps, test_lap):
    """Return train_df, test_df for one fold."""
    tr, te = [], []
    for pid in PARTICIPANTS:
        for lp in train_laps:
            tr.append(lap_to_df(load_lap_json(pid, lp), pid))
        te.append(lap_to_df(load_lap_json(pid, test_lap), pid))
    return pd.concat(tr, ignore_index=True), pd.concat(te, ignore_index=True)


def encode(train_df, test_df):
    le = LabelEncoder().fit(train_df["participant_id"])
    train_df["pid_code"] = le.transform(train_df["participant_id"])
    test_df ["pid_code"] = le.transform(test_df["participant_id"])
    return train_df, test_df, le


def fit_rf(X, y):
    """Pipeline: impute NaNs → Random-Forest."""
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    return make_pipeline(SimpleImputer(strategy="median"), rf).fit(X, y)


def f1_per_pid(y_true, y_pred, le):
    return {name: f1_score(y_true == code, y_pred == code,
                           average="binary", zero_division=0)
            for code, name in enumerate(le.classes_)}


# ─────────────────────────────── main ────────────────────────────────────────
def main():
    print(f"[{now():6.2f}s] ===== START =====")

    # containers for tables & confusion matrix
    pid_history       = {pid: [] for pid in PARTICIPANTS}
    macro_scores, weighted_scores = [], []
    y_true_all, y_pred_all = [], []

    for fold_idx, (train_laps, test_lap) in enumerate(FOLDS, 1):
        print(f"\n[{now():6.2f}s] --- Fold {fold_idx} | train {train_laps}  test {test_lap} ---")
        t0 = now()

        train_df, test_df = build(train_laps, test_lap)
        train_df, test_df, le = encode(train_df, test_df)

        Xtr = train_df[FEATURES].astype(np.float64).replace([np.inf, -np.inf], np.nan)
        ytr = train_df["pid_code"]
        Xte = test_df [FEATURES].astype(np.float64).replace([np.inf, -np.inf], np.nan)
        yte = test_df ["pid_code"]

        model = fit_rf(Xtr, ytr)
        preds = model.predict(Xte)

        per_pid   = f1_per_pid(yte, preds, le)
        macro_f1  = f1_score(yte, preds, average="macro")
        weight_f1 = f1_score(yte, preds, average="weighted")

        for pid, sc in per_pid.items():
            pid_history[pid].append(sc)
        macro_scores.append(macro_f1)
        weighted_scores.append(weight_f1)

        y_true_all.extend(yte)
        y_pred_all.extend(preds)

        print(f"  macro F1: {macro_f1:.3f}   weighted F1: {weight_f1:.3f}   time: {now()-t0:.2f}s")

    # ── build and print tables ───────────────────────────────────────────────
    pid_df = pd.DataFrame(pid_history, index=FOLD_NAMES).T
    pid_df["Mean"] = pid_df.mean(axis=1)

    overall_df = pd.DataFrame([macro_scores, weighted_scores],
                              index=["Macro", "Weighted"],
                              columns=FOLD_NAMES)
    overall_df["Mean"] = overall_df.mean(axis=1)

    pd.options.display.float_format = "{:6.3f}".format
    print("\n=== Per-participant F1 (rows) × Fold (columns) ===")
    print(pid_df.to_string())

    print("\n=== Overall F1 scores ===")
    print(overall_df.to_string())

    # ── confusion-matrix plot ────────────────────────────────────────────────
    
    cm = confusion_matrix(y_true_all, y_pred_all, labels=range(len(le.classes_)))
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)

    fig, ax = plt.subplots(figsize=(9, 9))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".0f")
    plt.xticks(rotation=45, ha="right")
    plt.title("Random-Forest confusion matrix (all 5 folds combined)")
    plt.tight_layout()
    plt.show()

    print(f"\n[{now():6.2f}s] ===== FINISHED =====")


if __name__ == "__main__":
    main()
