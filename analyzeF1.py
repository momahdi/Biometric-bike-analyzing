#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Five-fold rider-ID model

•  brake + acceleration features  (timestamp removed)
•  per-fold and averaged Precision / Recall / F1
•  raw confusion-matrix plot
•  Random-Forest feature importance for **every fold** + mean column
"""

import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


# ─────────────────── stopwatch ──────────────────────────────────────────────
def now() -> float:
    if not hasattr(now, "_t0"):
        now._t0 = time.perf_counter()
    return time.perf_counter() - now._t0


# ─────────────────── configuration ──────────────────────────────────────────
BASE_DIR     = Path("./Segmented")
PARTICIPANTS = [f"P{str(i).zfill(2)}" for i in range(1, 16)]          # P01…P15

FOLDS = [([1, 2, 3, 4], 5),
         ([2, 3, 4, 5], 1),
         ([3, 4, 5, 1], 2),
         ([4, 5, 1, 2], 3),
         ([5, 1, 2, 3], 4)]
FOLD_NAMES = [f"F{idx}" for idx in range(1, len(FOLDS) + 1)]

FEATURES = [
    "brake_value",
    "accel_x", "accel_y", "accel_z",
    "accel_mag",
]


# ─────────────────── helpers ────────────────────────────────────────────────
def to_float(x):
    try:           return float(x)
    except Exception: return np.nan


def load_lap_json(pid: str, lap: int):
    with open(BASE_DIR / pid / f"lap_{lap}.json") as f:
        return json.load(f)


def lap_to_df(lap_data, pid):
    """Flatten one lap to a DataFrame (brake + accel)."""
    rows = []
    for d in lap_data:
        acc = d.get("userAccel") or d.get("acceleration") or {}
        ax, ay, az = map(to_float, (acc.get("x"), acc.get("y"), acc.get("z")))
        rows.append(
            dict(
                brake_value = to_float(d.get("brakeData")),
                accel_x     = ax,
                accel_y     = ay,
                accel_z     = az,
                accel_mag   = np.sqrt(ax**2 + ay**2 + az**2)
                              if not np.isnan(ax) and not np.isnan(ay) and not np.isnan(az)
                              else np.nan,
                participant_id = pid,
            )
        )
    return pd.DataFrame(rows)


def build(train_laps, test_lap):
    tr, te = [], []
    for pid in PARTICIPANTS:
        for lp in train_laps:
            tr.append(lap_to_df(load_lap_json(pid, lp), pid))
        te.append(lap_to_df(load_lap_json(pid, test_lap), pid))
    return pd.concat(tr, ignore_index=True), pd.concat(te, ignore_index=True)


def encode(tr_df, te_df):
    le = LabelEncoder().fit(tr_df["participant_id"])
    tr_df["pid_code"] = le.transform(tr_df["participant_id"])
    te_df["pid_code"] = le.transform(te_df["participant_id"])
    return tr_df, te_df, le


def fit_rf(X, y):
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    return make_pipeline(SimpleImputer(strategy="median"), rf).fit(X, y)


# ─────────────────── main ───────────────────────────────────────────────────
def main():
    print(f"[{now():6.2f}s] ===== START =====")
    print("Features used:", FEATURES)

    pid_hist = {p: [] for p in PARTICIPANTS}
    macro, weighted = [], []
    y_true_all, y_pred_all = [], []
    importances = []             # store per-fold feature importances

    for k, (tr_laps, te_lap) in enumerate(FOLDS, 1):
        print(f"\n[{now():6.2f}s] --- Fold {k} | train {tr_laps}  test {te_lap} ---")
        t0 = now()

        tr_df, te_df = build(tr_laps, te_lap)
        tr_df, te_df, le = encode(tr_df, te_df)

        Xtr, ytr = tr_df[FEATURES].astype(float), tr_df["pid_code"]
        Xte, yte = te_df[FEATURES].astype(float), te_df["pid_code"]

        model  = fit_rf(Xtr, ytr)
        preds  = model.predict(Xte)

        # pull RF estimator out of pipeline & store fold-level importances
        importances.append(model.named_steps["randomforestclassifier"].feature_importances_)

        # per-class F1 bookkeeping
        for p, f1v in {n: f1_score(yte == c, preds == c, average="binary", zero_division=0)
                       for c, n in enumerate(le.classes_)}.items():
            pid_hist[p].append(f1v)
        macro.append(   f1_score(yte, preds, average="macro") )
        weighted.append(f1_score(yte, preds, average="weighted"))

        y_true_all.extend(yte)
        y_pred_all.extend(preds)

        print("  macro-F1:", f"{macro[-1]:.3f}",
              "weighted-F1:", f"{weighted[-1]:.3f}",
              "time:", f"{now()-t0:.2f}s")

    # ── feature importance by fold ──────────────────────────────────────────
    imp_matrix          = pd.DataFrame(importances, columns=FEATURES, index=FOLD_NAMES).T
    imp_matrix["Mean"]  = imp_matrix.mean(axis=1)

    pd.options.display.float_format = "{:6.3f}".format
    print("\n=== Random-Forest feature importance by fold ===")
    print(imp_matrix.to_string())

    # ── per-participant F1 table ────────────────────────────────────────────
    pid_df = pd.DataFrame(pid_hist, index=FOLD_NAMES).T
    pid_df["Mean"] = pid_df.mean(axis=1)

    overall_df = pd.DataFrame([macro, weighted],
                              index=["Macro-F1", "Weighted-F1"],
                              columns=FOLD_NAMES)
    overall_df["Mean"] = overall_df.mean(axis=1)

    print("\n=== Per-participant F1 by fold ===")
    print(pid_df.to_string())
    print("\n=== Overall macro / weighted F1 ===")
    print(overall_df.to_string())

    # ── per-class Precision / Recall / F1 ───────────────────────────────────
    prec = precision_score(y_true_all, y_pred_all, average=None, labels=range(len(le.classes_)))
    rec  = recall_score   (y_true_all, y_pred_all, average=None, labels=range(len(le.classes_)))
    f1s  = f1_score       (y_true_all, y_pred_all, average=None, labels=range(len(le.classes_)))
    pr_df = pd.DataFrame({"Precision": prec, "Recall": rec, "F1": f1s},
                         index=le.classes_).round(3)

    print("\n=== Per-class Precision / Recall / F1 ===")
    print(pr_df.to_string())

    # overall macro metrics
    print("\nOverall macro Precision :", f"{precision_score(y_true_all, y_pred_all, average='macro'):.3f}")
    print("Overall macro Recall    :", f"{recall_score   (y_true_all, y_pred_all, average='macro'):.3f}")
    print("Overall macro F1        :", f"{f1_score       (y_true_all, y_pred_all, average='macro'):.3f}")

    # ── confusion matrix (counts) ───────────────────────────────────────────
    cm = confusion_matrix(y_true_all, y_pred_all, labels=range(len(le.classes_)))
    fig, ax = plt.subplots(figsize=(9, 9))
    ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format=".0f")
    plt.xticks(rotation=45, ha="right")
    plt.title("Random-Forest confusion matrix (5-fold pooled)")
    plt.tight_layout()
    plt.show()

    print(f"\n[{now():6.2f}s] ===== FINISHED =====")


if __name__ == "__main__":
    main()
