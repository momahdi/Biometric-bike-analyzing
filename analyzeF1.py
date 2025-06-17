import json, glob, os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

# ---------- 1. data utilities -------------------------------------------------
BASE_DIR      = Path("./Segmented")
PARTICIPANTS  = [f"P{str(i).zfill(2)}" for i in range(1, 16)]   # P01 … P15
TRAIN_LAPS    = [1, 2, 3, 4]
TEST_LAP      = 5

def load_lap_json(participant: str, lap: int) -> list[dict]:
    """Read one lap JSON file and return the loaded list-of-dicts."""
    file_path = BASE_DIR / participant / f"lap_{lap}.json"
    with open(file_path, "r") as f:
        return json.load(f)

def lap_to_df(lap_data: list[dict], participant: str) -> pd.DataFrame:
    """Convert one lap’s raw dicts into a tidy DataFrame."""
    return pd.DataFrame(
        {
            "timestamp"      : [float(d["unixTimeStamp"]) for d in lap_data],
            "brake_value"    : [float(d["brakeData"])     for d in lap_data],
            "participant_id" : participant,
        }
    )

def build_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return concatenated train and test DataFrames."""
    train_frames, test_frames = [], []

    for pid in PARTICIPANTS:
        # laps 1-4 → training
        for lap in TRAIN_LAPS:
            train_frames.append(lap_to_df(load_lap_json(pid, lap), pid))
        # lap 5 → test
        test_frames.append(lap_to_df(load_lap_json(pid, TEST_LAP), pid))

    train_df = pd.concat(train_frames, ignore_index=True)
    test_df  = pd.concat(test_frames,  ignore_index=True)
    return train_df, test_df

# ---------- 2. modelling pipeline --------------------------------------------
def encode_labels(train_df: pd.DataFrame, test_df: pd.DataFrame
                  ) -> tuple[pd.DataFrame, pd.DataFrame, LabelEncoder]:
    """Add numeric column 'pid_code' shared across train & test."""
    le = LabelEncoder().fit(train_df["participant_id"])
    train_df = train_df.assign(pid_code = le.transform(train_df["participant_id"]))
    test_df  = test_df.assign(pid_code  = le.transform(test_df["participant_id"]))
    return train_df, test_df, le

def fit_models(X_train, y_train):
    rf  = RandomForestClassifier(n_estimators=300, random_state=42)
    svm = SVC(kernel="linear", probability=False, random_state=42)
    rf .fit(X_train, y_train)
    svm.fit(X_train, y_train)
    return {"RandomForest": rf, "SVM": svm}

def f1_per_participant(y_true, y_pred, participants) -> dict:
    """Compute F1 score separately for each participant subset."""
    scores = {}
    for pid_code, pid_name in participants.items():
        mask = y_true == pid_code
        scores[pid_name] = f1_score(
            y_true[mask], y_pred[mask], average="macro", zero_division=0
        )
    return scores

def evaluate(models: dict, X_test, y_test, le: LabelEncoder):
    """Return two dicts: model->per-PID F1 and model->overall F1."""
    results_per_pid, results_overall = {}, {}
    pid_lookup = dict(enumerate(le.classes_))

    for name, model in models.items():
        preds = model.predict(X_test)
        results_per_pid[name] = f1_per_participant(y_test, preds, pid_lookup)
        results_overall[name] = {
            "macro_F1"   : f1_score(y_test, preds, average="macro"),
            "weighted_F1": f1_score(y_test, preds, average="weighted"),
        }
    return results_per_pid, results_overall

# ---------- 3. run the whole pipeline ----------------------------------------
def main():
    train_df, test_df = build_datasets()
    train_df, test_df, le = encode_labels(train_df, test_df)

    X_train, y_train = train_df[["timestamp", "brake_value"]], train_df["pid_code"]
    X_test,  y_test  = test_df[["timestamp", "brake_value"]],  test_df["pid_code"]

    models = fit_models(X_train, y_train)
    per_pid, overall = evaluate(models, X_test, y_test, le)

    # nicely print
    for mdl, scores in per_pid.items():
        print(f"\n=== {mdl} : F1 by participant ===")
        for pid, val in scores.items():
            print(f"{pid}: {val:.3f}")
        print(f"Overall macro F1   : {overall[mdl]['macro_F1']:.3f}")
        print(f"Overall weighted F1: {overall[mdl]['weighted_F1']:.3f}")

if __name__ == "__main__":
    main()