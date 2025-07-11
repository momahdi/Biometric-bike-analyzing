#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Five‑fold rider‑ID model — 25‑feature linear‑SVM
================================================
Same structure as the Random‑Forest version but with a LinearSVC.
Feature‑importance reporting has been removed because SVMs do not
have native feature‑importance scores (you can inspect the linear
coefficients separately if needed).
"""

import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

# ─────────────────── stopwatch ────────────────────────────────────────────


def now() -> float:
    if not hasattr(now, "_t0"):
        now._t0 = time.perf_counter()
    return time.perf_counter() - now._t0


# ─────────────────── configuration ────────────────────────────────────────
DAY = "Day1"
BASE_DIR = Path(f".././Segmented/{DAY}")
PARTICIPANTS = [
    "P01",
    "P02",
    "P03",
    "P04",
    "P05",
    "P06",
    "P07",
    "P09",
    "P11",
    "P12",
    "P13",
    "P14",
    "P15",
    "P16",
]
FOLDS = [
    ([1, 2, 3, 4], 5),
    ([2, 3, 4, 5], 1),
    ([3, 4, 5, 1], 2),
    ([4, 5, 1, 2], 3),
    ([5, 1, 2, 3], 4),
]
FOLD_NAMES = [f"F{i}" for i in range(1, 6)]

FEATURES = [
    "brake_value",
    "accel_x",
    "accel_y",
    "accel_z",
    "accel_mag",
    "cadence",
    "phone_cadence",
    "velocity",
    "pedal_L",
    "pedal_R",
    "power_surrogate",
    "roll",
    "pitch",
    "yaw",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "mag_x",
    "mag_y",
    "mag_z",
    "quat_x",
    "quat_y",
    "quat_z",
    "quat_w",
    "altitude",
]

# ─────────────────── helpers ─────────────────────────────────────────────


def to_f(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def load_lap(pid, lap):
    with open(BASE_DIR / pid / f"lap_{lap}.json") as fh:
        return json.load(fh)


def lap_df(lap, pid):
    rows = []
    for d in lap:
        acc = d.get("userAccel") or d.get("acceleration") or {}
        ax, ay, az = map(to_f, (acc.get("x"), acc.get("y"), acc.get("z")))
        accel_mag = (
            np.sqrt(ax**2 + ay**2 + az**2)
            if not any(np.isnan([ax, ay, az]))
            else np.nan
        )
        rot, mag, quat = (
            d.get("rotationRate", {}),
            d.get("magneticField", {}),
            d.get("quaternion", {}),
        )
        ped = d.get("pedalWeight", {})
        cadence = to_f(d.get("cadence"))
        pedal_L, pedal_R = to_f(ped.get("L")), to_f(ped.get("R"))
        pedal_sum = pedal_L + pedal_R
        power_sur = (
            cadence * pedal_sum
            if not np.isnan(cadence) and not np.isnan(pedal_sum)
            else np.nan
        )
        rows.append(
            dict(
                brake_value=to_f(d.get("brakeData")),
                accel_x=ax,
                accel_y=ay,
                accel_z=az,
                accel_mag=accel_mag,
                cadence=cadence,
                phone_cadence=to_f(d.get("phoneCadence")),
                velocity=to_f(d.get("locationData", {}).get("velocity")),
                pedal_L=pedal_L,
                pedal_R=pedal_R,
                power_surrogate=power_sur,
                roll=to_f(d.get("roll")),
                pitch=to_f(d.get("pitch")),
                yaw=to_f(d.get("yaw")),
                gyro_x=to_f(rot.get("x")),
                gyro_y=to_f(rot.get("y")),
                gyro_z=to_f(rot.get("z")),
                mag_x=to_f(mag.get("x")),
                mag_y=to_f(mag.get("y")),
                mag_z=to_f(mag.get("z")),
                quat_x=to_f(quat.get("x")),
                quat_y=to_f(quat.get("y")),
                quat_z=to_f(quat.get("z")),
                quat_w=to_f(quat.get("w")),
                altitude=to_f(d.get("locationData", {}).get("altitude")),
                participant_id=pid,
            )
        )
    return pd.DataFrame(rows)


def build(train_laps, test_lap):
    tr, te = [], []
    for pid in PARTICIPANTS:
        for lp in train_laps:
            tr.append(lap_df(load_lap(pid, lp), pid))
        te.append(lap_df(load_lap(pid, test_lap), pid))
    return pd.concat(tr, ignore_index=True), pd.concat(te, ignore_index=True)


def encode(tr, te):
    le = LabelEncoder().fit(tr["participant_id"])
    tr["y"], te["y"] = le.transform(tr["participant_id"]), le.transform(
        te["participant_id"]
    )
    return tr, te, le


def svm_pipe():
    """Median‑impute, scale, then LinearSVC."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LinearSVC(C=1.0, class_weight="balanced", random_state=42, dual=False),
    )


# ─────────────────── main ───────────────────────────────────────────────


def main():
    print(f"[{now():5.2f}s] START")
    pid_hist, macro, weighted, y_true_all, y_pred_all = (
        {p: [] for p in PARTICIPANTS},
        [],
        [],
        [],
        [],
    )
    for (tr_laps, te_lap), name in zip(FOLDS, FOLD_NAMES):
        print(f"\n[{now():5.2f}s] {name}: train {tr_laps} → test {te_lap}")
        tr_df, te_df = build(tr_laps, te_lap)
        tr_df, te_df, le = encode(tr_df, te_df)
        Xtr, ytr, Xte, yte = (
            tr_df[FEATURES].astype(float),
            tr_df["y"],
            te_df[FEATURES].astype(float),
            te_df["y"],
        )
        model = svm_pipe().fit(Xtr, ytr)
        preds = model.predict(Xte)
        for pid, f1v in {
            n: f1_score(yte == i, preds == i, average="binary", zero_division=0)
            for i, n in enumerate(le.classes_)
        }.items():
            pid_hist[pid].append(f1v)
        macro.append(f1_score(yte, preds, average="macro"))
        weighted.append(f1_score(yte, preds, average="weighted"))
        y_true_all.extend(yte)
        y_pred_all.extend(preds)
        print("  macro-F1", f"{macro[-1]:.3f}")
    pid_df = pd.DataFrame(pid_hist, index=FOLD_NAMES).T
    pid_df["Mean"] = pid_df.mean(axis=1)
    overall = pd.DataFrame(
        [macro, weighted], index=["Macro-F1", "Weighted-F1"], columns=FOLD_NAMES
    )
    overall["Mean"] = overall.mean(axis=1)
    print("\n=== Per-participant F1 by fold ===\n", pid_df.to_string())
    print("\n=== Overall macro / weighted F1 ===\n", overall.to_string())
    prec, rec, f1s = (
        precision_score(y_true_all, y_pred_all, average=None),
        recall_score(y_true_all, y_pred_all, average=None),
        f1_score(y_true_all, y_pred_all, average=None),
    )
    print(
        "\n=== Pooled per-class PRF ===\n",
        pd.DataFrame({"Precision": prec, "Recall": rec, "F1": f1s}, index=le.classes_)
        .round(3)
        .to_string(),
    )
    print(
        "\nOverall pooled macro F1:",
        f"{f1_score(y_true_all, y_pred_all, average='macro'):.3f}",
    )
    cm = confusion_matrix(y_true_all, y_pred_all, labels=range(len(le.classes_)))
    fig, ax = plt.subplots(figsize=(9, 9))
    ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format=".0f"
    )
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Linear‑SVM confusion matrix |  Track B {DAY} - 5 Laps on same day ")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
