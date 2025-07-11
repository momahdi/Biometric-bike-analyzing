#!/usr/bin/env python3
"""
addCadence.py
-------------

Walk every folder

    ./Unsegmented/Day*/Track-*/P??

where  P??  is listed in PARTICIPANTS, merge Garmin cadence into the phone
JSON, and write the patched file as

    <original-stem>.Cadence.json
"""

from pathlib import Path
import json, pandas as pd, numpy as np, datetime as dt, re, sys, argparse

# ─────────── configuration ───────────
BASE_DIR = Path("./Unsegmented/")
PARTICIPANTS = [
    "P01", "P02", "P03", "P04", "P05", "P06", "P07",
    "P09", "P10", "P11", "P12", "P13", "P14", "P15", "P16",
]
TOLERANCE  = 0.5    # ± seconds
OFFSET_SEC = 0.0    # manual clock-shift if needed
# ─────────────────────────────────────

# ===== helper functions (same logic as before) =====
def _find(df, *names):
    s = {n.lower() for n in names}
    return next(c for c in df.columns if c.lower() in s)

def read_garmin(p: Path):
    df = pd.read_csv(p).rename(columns=str.strip)
    t = _find(df, "unix_timestamp", "unixtimestamp", "unix", "unixtime", "time")
    c = _find(df, "cadence", "rpm")
    df = (df.assign(sec=df[t].astype(int), cadence=df[c].astype(float))
            .sort_values("sec")
            .reset_index(drop=True))
    rnk = df.groupby("sec", sort=False).cumcount()
    size = df.groupby("sec", sort=False)[c].transform("size")
    df["unix"] = df["sec"] + (rnk + 1) / (size + 1)  # synthetic spacing
    return df[["unix", "cadence"]]

def iso_to_unix(s: str):
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S %z").timestamp()

def derive_offset(ts):
    for rec in ts:
        iso = rec.get("locationData", {}).get("timestamp")
        if iso:
            return iso_to_unix(iso) - float(rec["timestamp"])
    raise RuntimeError("locationData.timestamp missing")

def read_phone(p: Path):
    js = json.loads(p.read_text())
    ts = js["timestamps"]
    offset = derive_offset(ts) + OFFSET_SEC
    df = (pd.json_normalize(ts)
            .assign(unix=lambda d: d["timestamp"].astype(float) + offset)
            .sort_values("unix")
            .reset_index(drop=True))
    return js, df

def fmt(x: float):
    s = f"{x:.2f}"
    return s.rstrip("0").rstrip(".")

def merge_one_to_one(garmin, phone):
    filled = np.zeros(len(phone), bool)
    out = pd.Series(np.nan, index=phone.index)
    g = p = matched = 0
    while g < len(garmin) and p < len(phone):
        g_t, g_c = garmin.iloc[g]
        while p < len(phone) and phone.at[p, "unix"] < g_t - TOLERANCE:
            p += 1
        j, cand = p, None
        while j < len(phone) and phone.at[j, "unix"] <= g_t + TOLERANCE:
            if not filled[j]:
                cand = j
                break
            j += 1
        if cand is not None:
            out.iat[cand] = g_c
            filled[cand]  = True
            matched += 1
        g += 1
    return out, matched
# ===================================================


def find_single(pattern: str, folder: Path) -> Path:
    lst = list(folder.glob(pattern))
    if len(lst) != 1:
        raise FileNotFoundError(f"expected exactly one '{pattern}' in {folder}")
    return lst[0]


def process_participant_folder(p_folder: Path):
    try:
        csv_path  = find_single("*_Cadence.csv", p_folder)
        json_path = find_single("*logfile-subject-*.json", p_folder)
    except Exception as exc:
        print(f"[{p_folder.relative_to(BASE_DIR)}] skip → {exc}")
        return

    garmin = read_garmin(csv_path)
    js, phone = read_phone(json_path)
    cadence, matched = merge_one_to_one(garmin, phone)

    for row, val in zip(js["timestamps"], cadence):
        if not pd.isna(val):
            row["cadence"] = fmt(val)

    out_path = json_path.with_suffix(".Cadence.json")
    out_path.write_text(json.dumps(js, indent=2))

    print(f"[{p_folder.relative_to(BASE_DIR)}]  "
          f"garmin:{len(garmin):3d}  matched:{matched:3d}")


def main():
    if not BASE_DIR.exists():
        sys.exit(f"BASE_DIR '{BASE_DIR}' not found")

    for day_dir in sorted(BASE_DIR.glob("Day*")):
        for track_dir in sorted(day_dir.glob("Track-*")):
            for p in PARTICIPANTS:
                p_folder = track_dir / p
                if p_folder.is_dir():
                    process_participant_folder(p_folder)

    print("✔ batch finished")


if __name__ == "__main__":
    main()
