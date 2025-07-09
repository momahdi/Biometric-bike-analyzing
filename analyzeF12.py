#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rider-ID — Day-split experiment (rich features + guess table)
===========================================================

• Train on laps 1-5 from Day 1, test on same laps Day 2
• Random-Forest (median imputation)
• 25-feature vektor (brake / accel / cadence / power-surrogate / orientation / gyro / mag / quaternion / altitude)
• Prints RF feature-importance list, per-class PRF, macro scores
• “Top wrong guess per rider”-tabell
• Confusion-matrix heat-map
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

# ───────── config ────────────────────────────────────────────────────────
ROOT_DIR   = Path("./Segmented")
TRAIN_DAY  = "Day1"
TEST_DAY   = "Day2"
PARTICIPANTS = [
    "P01", "P02", "P03", "P04", "P05", "P06", "P07",
    "P09", "P10", "P11", "P12", "P13", "P14", "P15", "P16",
]
LAPS = range(1, 6)

FEATURES = [
    # brake + accel
    "brake_value", "accel_x", "accel_y", "accel_z", "accel_mag",
    # cadence / power surrogate
    "cadence", "phone_cadence", "velocity", "pedal_L", "pedal_R", "power_surrogate",
    # orientation, gyro, magnetometer, quaternion
    "roll", "pitch", "yaw",
    "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y", "mag_z",
    "quat_x", "quat_y", "quat_z", "quat_w",
    # altitude
    "altitude",
]

# ───────── helpers ────────────────────────────────────────────────────────
def f(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def load_lap(day: str, pid: str, lap: int):
    with open(ROOT_DIR / day / pid / f"lap_{lap}.json") as fh:
        return json.load(fh)

def lap_to_df(records: list, pid: str) -> pd.DataFrame:
    rows = []
    for d in records:
        acc  = d.get("userAccel") or d.get("acceleration") or {}
        ax, ay, az = map(f, (acc.get("x"), acc.get("y"), acc.get("z")))
        accel_mag = np.sqrt(ax**2 + ay**2 + az**2) if not any(np.isnan([ax, ay, az])) else np.nan

        rot  = d.get("rotationRate", {})
        mag  = d.get("magneticField", {})
        quat = d.get("quaternion",    {})
        ped  = d.get("pedalWeight",   {})

        cadence = f(d.get("cadence"))
        pedal_sum = f(ped.get("L")) + f(ped.get("R"))
        power_surrogate = cadence * pedal_sum if not np.isnan(cadence) and not np.isnan(pedal_sum) else np.nan

        rows.append(dict(
            brake_value=f(d.get("brakeData")),
            accel_x=ax, accel_y=ay, accel_z=az, accel_mag=accel_mag,
            cadence=cadence,
            phone_cadence=f(d.get("phoneCadence")),
            velocity=f(d.get("locationData", {}).get("velocity")),
            pedal_L=f(ped.get("L")), pedal_R=f(ped.get("R")),
            power_surrogate=power_surrogate,
            roll=f(d.get("roll")), pitch=f(d.get("pitch")), yaw=f(d.get("yaw")),
            gyro_x=f(rot.get("x")), gyro_y=f(rot.get("y")), gyro_z=f(rot.get("z")),
            mag_x=f(mag.get("x")),  mag_y=f(mag.get("y")),  mag_z=f(mag.get("z")),
            quat_x=f(quat.get("x")), quat_y=f(quat.get("y")), quat_z=f(quat.get("z")), quat_w=f(quat.get("w")),
            altitude=f(d.get("locationData", {}).get("altitude")),
            participant_id=pid,
        ))
    return pd.DataFrame(rows)

def build_data():
    tr, te = [], []
    for pid in PARTICIPANTS:
        for lp in LAPS:
            tr.append(lap_to_df(load_lap(TRAIN_DAY, pid, lp), pid))
            te.append(lap_to_df(load_lap(TEST_DAY,  pid, lp), pid))
    return pd.concat(tr, ignore_index=True), pd.concat(te, ignore_index=True)

# ───────── main ──────────────────────────────────────────────────────────
def main():
    print("Running Day-split RF with 25 features …")

    tr_df, te_df = build_data()
    le = LabelEncoder().fit(tr_df["participant_id"])
    tr_df["y"], te_df["y"] = le.transform(tr_df["participant_id"]), le.transform(te_df["participant_id"])

    Xtr, ytr = tr_df[FEATURES].astype(float), tr_df["y"]
    Xte, yte = te_df[FEATURES].astype(float), te_df["y"]

    model = make_pipeline(SimpleImputer(strategy="median"),
                          RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=42))
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)

    # ---------- feature importances -------------------------------------
    rf = model.named_steps["randomforestclassifier"]
    fi_raw = dict(zip(Xtr.columns[: len(rf.feature_importances_)], rf.feature_importances_))
    fi = pd.Series({f: fi_raw.get(f, 0.0) for f in FEATURES}).sort_values(ascending=False)
    print("\n=== RF feature importance (all features) ===")
    print(fi.round(3).to_string())

    # ---------- metrics --------------------------------------------------
    print("\nMacro  P:{:.3f}  R:{:.3f}  F1:{:.3f}".format(
        precision_score(yte, preds, average="macro"),
        recall_score   (yte, preds, average="macro"),
        f1_score       (yte, preds, average="macro")
    ))

    per_cls = pd.DataFrame({
        "Precision": precision_score(yte, preds, average=None),
        "Recall":    recall_score   (yte, preds, average=None),
        "F1":        f1_score       (yte, preds, average=None),
    }, index=le.classes_).round(3)
    print("\n=== Per-class Precision / Recall / F1 ===\n", per_cls.to_string())

    # ---------- confusion matrix & top-guess table -----------------------
    cm = confusion_matrix(yte, preds)

    off_diag = cm.copy(); np.fill_diagonal(off_diag, 0)
    top_idx = off_diag.argmax(axis=1)
    guess_tbl = pd.DataFrame({
        "True": le.classes_,
        "Top guess": [le.classes_[j] for j in top_idx],
        "Count": off_diag.max(axis=1),
        "Total": cm.sum(axis=1),
    })
    guess_tbl["Pct"] = (guess_tbl["Count"] / guess_tbl["Total"]).round(3)
    print("\n=== Top wrong guess per rider ===\n", guess_tbl.to_string(index=False))

    # plot matrix
    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format=".0f")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"RF confusion matrix  |  Train {TRAIN_DAY} → Test {TEST_DAY}")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
