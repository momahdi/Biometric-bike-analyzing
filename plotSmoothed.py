#!/usr/bin/env python3
"""
smooth_and_connect_clean.py
---------------------------

Detect core GPS outliers → drop → interpolate → median-filter.

Outputs (overwritten every run)
  <stem>.removed.json
  plot/<stem>.smoothed.html   (blue+red+orange)
  plot/<stem>.removed.html    (blue+red only)
"""

from pathlib import Path
import json
import numpy as np
import folium
from scipy.interpolate import interp1d
from scipy.signal import medfilt

# ─── CONFIG ────────────────────────────────────────────────────────────
BASE_DIR = Path("./Unsegmented")
DAYS     = ["Day1", "Day2"]
TRACKS   = ["Track-A", "Track-B"]

JUMP_M, SPEED_MPS = 15.0, 10.0
HAMPEL_WIN, HAMPEL_SIGMA = 11, 3.0
MEDFILT_WIN = 7
ZOOM_START  = 16
FILL_MISSING_TIME = True
# ────────────────────────────────────────────────────────────────────────


# ---------- helpers -------------------------------------------------------
def haversine(a, b, c, d):
    R = 6_371_000
    p1, p2 = np.radians(a), np.radians(c)
    dp, dl = p2 - p1, np.radians(d - b)
    return 2 * R * np.arcsin(np.sqrt(np.sin(dp / 2) ** 2 +
                                     np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2))


def to_xy(lat, lon):
    return lon * np.cos(np.radians(np.nanmean(lat))) * 111_320, lat * 110_574


def hampel(arr, k=HAMPEL_WIN, sigma=HAMPEL_SIGMA):
    n = len(arr)
    if n < k or k % 2 == 0:
        return np.zeros(n, bool)
    h = k // 2
    mask = np.zeros(n, bool)
    for i in range(h, n - h):
        win = arr[i - h:i + h + 1]
        med = np.nanmedian(win)
        mad = np.nanmedian(np.abs(win - med)) or 1e-6
        mask[i] = np.abs(arr[i] - med) > sigma * mad
    return mask


def ffloat(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def load(fp: Path):
    rows = json.load(fp.open()).get("timestamps", [])
    lat, lon, ts = [], [], []
    for r in rows:
        lat.append(ffloat(r.get("locationData", {}).get("latitude")))
        lon.append(ffloat(r.get("locationData", {}).get("longitude")))
        ts.append(ffloat(r.get("timestamp")))
    lat, lon, ts = np.array(lat), np.array(lon), np.array(ts, float)
    if FILL_MISSING_TIME or np.isnan(ts).any() or len(np.unique(ts)) != len(ts):
        ts = np.arange(len(lat), dtype=float)
    return lat, lon, ts, rows


def core_mask(lat, lon, ts):
    n = len(lat)
    mask = np.isnan(lat) | np.isnan(lon)
    good = ~mask
    if good.sum() >= 2:
        d_prev = np.zeros(n)
        d_next = np.zeros(n)
        d_prev[1:] = haversine(lat[1:], lon[1:], lat[:-1], lon[:-1])
        d_next[:-1] = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
        speed = d_prev / np.clip(np.diff(ts, prepend=ts[0]), 1e-3, None)
        mask |= (d_prev > JUMP_M) | (d_next > JUMP_M) | (speed > SPEED_MPS)
    x, y = to_xy(lat, lon)
    mask |= hampel(x) | hampel(y)
    mask[0] = mask[-1] = False
    return mask


def smooth(lat, lon, keep_mask):
    idx = np.arange(len(lat))
    f_lat = interp1d(idx[keep_mask], lat[keep_mask], kind="linear", fill_value="extrapolate")
    f_lon = interp1d(idx[keep_mask], lon[keep_mask], kind="linear", fill_value="extrapolate")
    lat_s, lon_s = f_lat(idx), f_lon(idx)
    if len(lat_s) >= MEDFILT_WIN and MEDFILT_WIN % 2 == 1:
        lat_s = medfilt(lat_s, MEDFILT_WIN)
        lon_s = medfilt(lon_s, MEDFILT_WIN)
    return lat_s, lon_s


# ---------- HTML map creators --------------------------------------------
def add_track(lat, lon, m):
    for la, lo in zip(lat, lon):
        folium.CircleMarker((la, lo), 3,
                            color="blue", fill=True, fill_color="blue",
                            fill_opacity=.6).add_to(m)
    folium.PolyLine(list(zip(lat, lon)), color="red", weight=2.5).add_to(m)


def add_start_end(lat, lon, m):
    folium.Marker((lat[0],  lon[0]), icon=folium.Icon(color="green"), popup="Start").add_to(m)
    folium.Marker((lat[-1], lon[-1]), icon=folium.Icon(color="red"),   popup="End").add_to(m)


def write_smoothed_map(lat, lon, o_lat, o_lon, out_html):
    m = folium.Map((lat[0], lon[0]), zoom_start=ZOOM_START)
    add_track(lat, lon, m)
    for la, lo in zip(o_lat, o_lon):
        folium.CircleMarker((la, lo), 4,
                            color="orange", fill=True, fill_color="orange",
                            fill_opacity=.9).add_to(m)
    add_start_end(lat, lon, m)
    m.save(out_html)


def write_removed_map(lat, lon, out_html):
    m = folium.Map((lat[0], lon[0]), zoom_start=ZOOM_START)
    add_track(lat, lon, m)
    add_start_end(lat, lon, m)
    m.save(out_html)


# ---------- per-file processing ------------------------------------------
def process(jf: Path):
    stem = jf.stem
    if stem.endswith(".smoothed") or stem.endswith(".removed"):
        return  # never re-process generated files

    lat, lon, ts, rows = load(jf)
    mask  = core_mask(lat, lon, ts)
    keep  = ~mask
    lat_s, lon_s = smooth(lat, lon, keep)

    # ----- JSON with only valid rows -------------------------------------
    kept_rows = []
    for ok, row, la, lo in zip(keep, rows, lat_s, lon_s):
        if not ok:
            continue
        r2 = dict(row)
        r2.setdefault("locationData", {})
        r2["locationData"]["latitude"]  = f"{la}"
        r2["locationData"]["longitude"] = f"{lo}"
        kept_rows.append(r2)

    out_json = jf.with_suffix("").parent / f"{stem}.removed.json"
    out_json.unlink(missing_ok=True)           # ensure overwrite
    with out_json.open("w") as fp:
        json.dump({"timestamps": kept_rows}, fp, indent=2)

    # ----- HTML maps ------------------------------------------------------
    plot_dir = jf.parent / "plot"
    plot_dir.mkdir(exist_ok=True)

    sm_html = plot_dir / f"{stem}.smoothed.html"
    rm_html = plot_dir / f"{stem}.removed.html"
    sm_html.unlink(missing_ok=True)
    rm_html.unlink(missing_ok=True)

    write_smoothed_map(lat_s, lon_s,
                       lat[mask & ~np.isnan(lat)],
                       lon[mask & ~np.isnan(lon)],
                       sm_html)

    write_removed_map(lat_s, lon_s, rm_html)

    print(f"✓ {stem}: kept {keep.sum()}/{len(lat)}")


# ---------- batch driver --------------------------------------------------
def main():
    for day in DAYS:
        for track in TRACKS:
            root = BASE_DIR / day / track
            if not root.is_dir():
                continue
            for p in root.glob("P*"):
                if not p.is_dir():
                    continue
                for jf in p.glob("*.json"):
                    process(jf)


if __name__ == "__main__":
    main()
