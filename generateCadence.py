#!/usr/bin/env python3
"""
addCadence.py
─────────────
Walk every folder

    ./Unsegmented/Day*/Track-*/P??

merge Garmin cadence AND phone-derived cadence into the phone JSON and
save it as

    <original-stem>.Cadence.json        (over-writes in place)

Fields added / updated inside each timestamp object
    cadence        – Garmin sensor, string (rpm, 2 decimals)
    phoneCadence   – velocity-based, string (rpm, 2 decimals)
"""

from pathlib import Path
import json, math, pandas as pd, numpy as np, datetime as dt, sys

# ─────────── configuration ───────────
BASE_DIR  = Path("./Unsegmented/")
PARTICIPANTS = [
    "P01", "P02", "P03", "P04", "P05", "P06", "P07",
    "P09", "P10", "P11", "P12", "P13", "P14", "P15", "P16",
]
TOLERANCE   = 0.5        # ± seconds for Garmin ↔ phone match
OFFSET_SEC  = 0.0        # manual clock-shift if needed

# phone-side cadence
WHEEL_CIRCUMFERENCE_M = 2.3     # wheel roll-out in metres
DECIMAL_PLACES = 2
_FMT = f"{{:.{DECIMAL_PLACES}f}}".format   # e.g. _FMT(78.256) → "78.26"
# ────────────────────────────────────────


# ═══ helper functions ════════════════════════════════════════════════════════
def _find(df, *names):
    s = {n.lower() for n in names}
    return next(c for c in df.columns if c.lower() in s)


def read_garmin(p: Path):
    df = pd.read_csv(p).rename(columns=str.strip)
    t = _find(df, "unix_timestamp", "unixtimestamp", "unix", "unixtime", "time")
    c = _find(df, "cadence", "rpm")

    df = (
        df.assign(sec=df[t].astype(int),
                  cadence=df[c].astype(float))
          .sort_values("sec")
          .reset_index(drop=True)
    )

    # spread multiple samples that share the same whole-second Unix time
    rnk  = df.groupby("sec", sort=False).cumcount()
    size = df.groupby("sec", sort=False)[c].transform("size")
    df["unix"] = df["sec"] + (rnk + 1) / (size + 1)

    return df[["unix", "cadence"]]


def iso_to_unix(s: str):
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S %z").timestamp()


def derive_offset(ts):
    """Phone-sensor Unix = recorded timestamp + offset"""
    for rec in ts:
        iso = rec.get("locationData", {}).get("timestamp")
        if iso:
            try:
                return iso_to_unix(iso) - float(rec["timestamp"])
            except (TypeError, ValueError):
                continue
    raise RuntimeError("locationData.timestamp missing")


def read_phone(json_path: Path):
    js = json.loads(json_path.read_text())
    ts = js["timestamps"]

    offset = derive_offset(ts) + OFFSET_SEC

    df = (
        pd.json_normalize(ts)
          .assign(
              timestamp=lambda d: pd.to_numeric(d["timestamp"], errors="coerce"),
              unix=lambda d: d["timestamp"] + offset,
          )
          .dropna(subset=["timestamp", "unix"])
          .sort_values("unix")
          .reset_index(drop=True)
    )
    return js, df


def fmt(x: float) -> str:
    """Float → trimmed string, 2 d.p."""
    s = f"{x:.2f}"
    return s.rstrip("0").rstrip(".")


def velo_to_cad_str(vel) -> str:
    """velocity [m/s] → cadence [rpm] (string) or 'NaN'"""
    try:
        vel = float(vel)
    except (TypeError, ValueError):
        vel = math.nan
    if math.isnan(vel):
        return "NaN"
    return _FMT((vel * 60) / WHEEL_CIRCUMFERENCE_M)


def merge_one_to_one(garmin, phone):
    """Greedy best-match within ±TOLERANCE seconds."""
    filled = np.zeros(len(phone), bool)
    out    = pd.Series(np.nan, index=phone.index)
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
            matched      += 1
        g += 1
    return out, matched
# ═════════════════════════════════════════════════════════════════════════════


def find_single(pattern: str, folder: Path) -> Path:
    files = [p for p in folder.glob(pattern)
             if not p.name.endswith(".removed.json")]
    if len(files) != 1:
        raise FileNotFoundError(
            f"expected exactly one '{pattern}' (ignoring *.removed.json) in {folder}"
        )
    return files[0]


def process_participant_folder(p_folder: Path):
    try:
        csv_path  = find_single("*_Cadence.csv",             p_folder)
        json_path = find_single("*logfile-subject-*.json",   p_folder)
    except Exception as exc:
        print(f"[{p_folder.relative_to(BASE_DIR)}] skip → {exc}")
        return

    # ── load data ─────────────────────────────────────────
    garmin_df         = read_garmin(csv_path)
    js, phone_df      = read_phone(json_path)
    g_cad, matched_g  = merge_one_to_one(garmin_df, phone_df)

    # ── inject cadence values ─────────────────────────────
    for ts_obj, g_val in zip(js["timestamps"], g_cad):
        if not pd.isna(g_val):
            ts_obj["cadence"] = fmt(g_val)

        vel = ts_obj.get("locationData", {}).get("velocity", math.nan)
        ts_obj["phoneCadence"] = velo_to_cad_str(vel)

    # ── write result ──────────────────────────────────────
    out_path = json_path.with_suffix(".Cadence.json")
    out_path.write_text(json.dumps(js, indent=2))

    print(f"[{p_folder.relative_to(BASE_DIR)}] "
          f"garmin:{len(garmin_df):3d} matched:{matched_g:3d}")


def main():
    if not BASE_DIR.exists():
        sys.exit(f"BASE_DIR '{BASE_DIR}' not found")

    for day_dir in sorted(BASE_DIR.glob("Day*")):
        for track_dir in sorted(day_dir.glob("Track-*")):
            for pid in PARTICIPANTS:
                p_folder = track_dir / pid if (track_dir / pid).is_dir() else None
                if p_folder and p_folder.is_dir():
                    process_participant_folder(p_folder)

    print("✔ batch finished")


if __name__ == "__main__":
    main()
