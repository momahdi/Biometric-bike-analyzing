#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

# ───────── config ────────────────────────────────────────────────────────
ROOT_DIR   = Path(".././Unsegmented")
TRAIN_DAY  = "Day1"
TEST_DAY   = "Day2"
PARTICIPANTS = [
    "P01", "P02", "P03", "P04", "P05", "P06", "P07",
    "P09", "P10", "P11", "P12", "P13", "P14", "P15", "P16",
]

FEATURES = [
    "brake_value", "accel_x", "accel_y", "accel_z", "accel_mag",
    "cadence", "phone_cadence", "velocity", "pedal_L", "pedal_R", "power_surrogate",
    "roll", "pitch", "yaw",
    "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y", "mag_z",
    "quat_x", "quat_y", "quat_z", "quat_w",
    "altitude",
]

# ───────── helpers ────────────────────────────────────────────────────────

def f(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def find_cadence_file(day: str, pid: str) -> Path:
    """Return the *single* Cadence.removed.json we care about.
    • Ignorerar filer som ligger i en underkatalog som heter "plot".
    • Om flera ändå återstår – välj den med senaste mtime.
    """
    day_dir = ROOT_DIR / day
    all_matches = [p for p in day_dir.rglob(f"*{pid}*.Cadence.removed.json")
                   if "plot" not in {parent.name for parent in p.parents}]
    if not all_matches:
        raise FileNotFoundError(f"No Cadence.removed.json for {pid} in {day}")
    if len(all_matches) > 1:
        all_matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return all_matches[0]


def load_rider_day(day: str, pid: str):
    with open(find_cadence_file(day, pid)) as fh:
        return json.load(fh)


def to_df(recs: list, pid: str) -> pd.DataFrame:
    rows = []
    for d in recs:
        acc  = d.get("userAccel") or d.get("acceleration") or {}
        ax, ay, az = map(f, (acc.get("x"), acc.get("y"), acc.get("z")))
        accel_mag = np.sqrt(ax**2 + ay**2 + az**2) if not any(np.isnan([ax, ay, az])) else np.nan
        rot  = d.get("rotationRate", {})
        mag  = d.get("magneticField", {})
        quat = d.get("quaternion",    {})
        ped  = d.get("pedalWeight",   {})
        cadence   = f(d.get("cadence"))
        pedal_sum = f(ped.get("L")) + f(ped.get("R"))
        power = cadence * pedal_sum if not np.isnan(cadence) and not np.isnan(pedal_sum) else np.nan
        rows.append(dict(
            brake_value=f(d.get("brakeData")),
            accel_x=ax, accel_y=ay, accel_z=az, accel_mag=accel_mag,
            cadence=cadence, phone_cadence=f(d.get("phoneCadence")),
            velocity=f(d.get("locationData", {}).get("velocity")),
            pedal_L=f(ped.get("L")), pedal_R=f(ped.get("R")), power_surrogate=power,
            roll=f(d.get("roll")), pitch=f(d.get("pitch")), yaw=f(d.get("yaw")),
            gyro_x=f(rot.get("x")), gyro_y=f(rot.get("y")), gyro_z=f(rot.get("z")),
            mag_x=f(mag.get("x")),  mag_y=f(mag.get("y")),  mag_z=f(mag.get("z")),
            quat_x=f(quat.get("x")), quat_y=f(quat.get("y")), quat_z=f(quat.get("z")), quat_w=f(quat.get("w")),
            altitude=f(d.get("locationData", {}).get("altitude")),
            participant_id=pid,
        ))
    return pd.DataFrame(rows)


def build_split():
    tr, te = [], []
    for pid in PARTICIPANTS:
        tr.append(to_df(load_rider_day(TRAIN_DAY, pid)["timestamps"], pid))
        te.append(to_df(load_rider_day(TEST_DAY,  pid)["timestamps"], pid))
    return pd.concat(tr, ignore_index=True), pd.concat(te, ignore_index=True)

# ───────── main ──────────────────────────────────────────────────────────

def main():
    print("Running Day-split Linear SVM with", len(FEATURES), "features …")

    tr_df, te_df = build_split()
    le = LabelEncoder().fit(tr_df["participant_id"])
    tr_df["y"], te_df["y"] = le.transform(tr_df["participant_id"]), le.transform(te_df["participant_id"])

    Xtr, ytr = tr_df[FEATURES].astype(float), tr_df["y"]
    Xte, yte = te_df[FEATURES].astype(float), te_df["y"]

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LinearSVC(C=1.0, dual=False, random_state=42)
    )
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)

    # feature weights (|coef|) averaged over classes
    svc = model.named_steps["linearsvc"]
    if hasattr(svc, "coef_"):
        weights = np.mean(np.abs(svc.coef_), axis=0)
        fi = pd.Series(weights, index=Xtr.columns).sort_values(ascending=False)
        print("\n=== SVM feature weights ===\n", fi.round(3).to_string())
    else:
        print("\n(No feature weights available for this SVM kernel)")

    # macro metrics
    print("\nMacro  P:{:.3f} R:{:.3f} F1:{:.3f}".format(
        precision_score(yte, preds, average="macro"),
        recall_score   (yte, preds, average="macro"),
        f1_score       (yte, preds, average="macro")
    ))

    # per-class table
    per_cls = pd.DataFrame({
        "Precision": precision_score(yte, preds, average=None),
        "Recall":    recall_score   (yte, preds, average=None),
        "F1":        f1_score       (yte, preds, average=None),
    }, index=le.classes_).round(3)
    print("\n=== Per-class PRF ===\n", per_cls.to_string())

    # top wrong guess
    cm = confusion_matrix(yte, preds)
    off = cm.copy(); np.fill_diagonal(off, 0)
    top = off.argmax(axis=1)
    tbl = pd.DataFrame({
        "True": le.classes_,
        "Top guess": [le.classes_[j] for j in top],
        "Count": off.max(axis=1),
        "Total": cm.sum(axis=1),
    })
    tbl["Pct"] = (tbl["Count"] / tbl["Total"]).round(3)
    print("\n=== Top wrong guess per rider ===\n", tbl.to_string(index=False))

    # plot
    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(ax=ax, cmap="Blues", colorbar=False, values_format=".0f")
    plt.xticks(rotation=45, ha="right"); plt.title(f"Linear SVM confusion matrix - Track A  |  {TRAIN_DAY} → {TEST_DAY}"); plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
