#!/usr/bin/env python3
"""
smooth_and_connect_clean_centerline.py
--------------------------------------

• Skalar bort GPS-punkter > MAX_DIST_M från mittlinjen
• Fångar outliers → interpolerar → medianfiltrerar
• Skriver
      <stem>.removed.json
      plot/<stem>.smoothed.html   (blå prickar + röd linje + GULA borttagna noder)
      plot/<stem>.removed.html    (blå prickar + röd linje)
"""

from pathlib import Path
import json, numpy as np, folium
from shapely.geometry import LineString, Point
from pyproj import Transformer
from scipy.interpolate import interp1d
from scipy.signal import medfilt

# ─── CONFIG ────────────────────────────────────────────────────────────
BASE_DIR      = Path("Unsegmented")
DAYS          = ["Day1", "Day2"]
TRACKS        = ["Track-A", "Track-B"]

MAX_DIST_M    = 5.0
JUMP_M        = 15.0
SPEED_MPS     = 10.0
HAMPEL_WIN, HAMPEL_SIGMA = 11, 3.0
MEDFILT_WIN   = 7
ZOOM_START    = 16
FILL_TIME     = True
# ────────────────────────────────────────────────────────────────────────


# ---------- geo helpers -------------------------------------------------
_tf = Transformer.from_crs(4326, 3857, always_xy=True)  # lon/lat → meter-plan

def haversine(a, b, c, d):
    a, b, c, d = map(np.radians, (a, b, c, d))
    da, dl = c - a, d - b
    h = np.sin(da/2)**2 + np.cos(a)*np.cos(c)*np.sin(dl/2)**2
    return 2 * 6_371_000 * np.arcsin(np.sqrt(h))

def to_xy(lat, lon):
    return _tf.transform(lon, lat)

def hampel(arr, k=HAMPEL_WIN, s=HAMPEL_SIGMA):
    if len(arr) < k or k % 2 == 0:
        return np.zeros(len(arr), bool)
    h, mask = k//2, np.zeros(len(arr), bool)
    for i in range(h, len(arr)-h):
        win = arr[i-h:i+h+1]
        med = np.median(win)
        mad = np.median(np.abs(win-med)) or 1e-6
        mask[i] = abs(arr[i]-med) > s*mad
    return mask


# ---------- profil-mittlinje -------------------------------------------
def load_profile(track):
    p = BASE_DIR / f"profile-{track}.geojson"
    if not p.is_file():
        p = BASE_DIR / "plot" / f"profile-{track}.geojson"
    if not p.is_file():
        raise FileNotFoundError(f"GeoJSON för {track} saknas")
    coords_ll = json.load(p.open())["features"][0]["geometry"]["coordinates"]
    coords_xy = [_tf.transform(lon, lat) for lon, lat in coords_ll]
    return LineString(coords_xy)

PROFILE = {trk: load_profile(trk) for trk in TRACKS}

def dist_to_center(lat, lon, track):
    return PROFILE[track].distance(Point(*to_xy(lat, lon)))


# ---------- json helper -------------------------------------------------
def ffloat(v):
    try: return float(v)
    except (TypeError, ValueError): return np.nan

def load_json(fp):
    rows = json.load(fp.open()).get("timestamps", [])
    lat, lon, ts = [], [], []
    for r in rows:
        lat.append(ffloat(r.get("locationData", {}).get("latitude")))
        lon.append(ffloat(r.get("locationData", {}).get("longitude")))
        ts.append(ffloat(r.get("timestamp")))
    lat, lon, ts = map(np.array, (lat, lon, ts))
    if FILL_TIME or np.isnan(ts).any() or len(np.unique(ts)) != len(ts):
        ts = np.arange(len(lat), dtype=float)
    return lat, lon, ts, rows


# ---------- mask / smoothing -------------------------------------------
def core_mask(lat, lon, ts):
    n = len(lat)
    mask = np.isnan(lat) | np.isnan(lon)
    if (~mask).sum() >= 2:
        d_prev = np.zeros(n); d_next = np.zeros(n)
        d_prev[1:]  = haversine(lat[1:], lon[1:], lat[:-1], lon[:-1])
        d_next[:-1] = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
        speed = d_prev / np.clip(np.diff(ts, prepend=ts[0]), 1e-3, None)
        mask |= (d_prev > JUMP_M) | (d_next > JUMP_M) | (speed > SPEED_MPS)
    x, y = to_xy(lat, lon)
    mask |= hampel(x) | hampel(y)
    mask[0] = mask[-1] = False
    return mask

def smooth(lat, lon, keep):
    idx = np.arange(len(lat))
    f_lat = interp1d(idx[keep], lat[keep], kind='linear', fill_value='extrapolate')
    f_lon = interp1d(idx[keep], lon[keep], kind='linear', fill_value='extrapolate')
    lat_s, lon_s = f_lat(idx), f_lon(idx)
    if len(lat_s) >= MEDFILT_WIN:
        lat_s = medfilt(lat_s, MEDFILT_WIN)
        lon_s = medfilt(lon_s, MEDFILT_WIN)
    return lat_s, lon_s


# ---------- folium-ritning ---------------------------------------------
def add_track(lat, lon, m):
    for la, lo in zip(lat, lon):
        folium.CircleMarker((la, lo), 3,
                            color="blue", fill=True, fill_color="blue",
                            fill_opacity=.6).add_to(m)
    folium.PolyLine(list(zip(lat, lon)), color="red", weight=2.5).add_to(m)

def add_start_end(lat, lon, m):
    folium.Marker((lat[0],  lon[0]),
                  icon=folium.Icon(color="green"), popup="Start").add_to(m)
    folium.Marker((lat[-1], lon[-1]),
                  icon=folium.Icon(color="red"),   popup="End").add_to(m)

def write_smoothed_map(lat, lon,
                       rm_lat, rm_lon, out_html):
    """
    lat/lon  – smoothed spår
    rm_*     – alla borttagna punkter (gul markering)
    """
    m = folium.Map((lat[0], lon[0]), zoom_start=ZOOM_START)
    add_track(lat, lon, m)
    for la, lo in zip(rm_lat, rm_lon):
        folium.CircleMarker((la, lo), 4,
                            color="yellow", fill=True, fill_color="yellow",
                            fill_opacity=.9).add_to(m)
    add_start_end(lat, lon, m)
    m.save(out_html)

def write_removed_map(lat, lon, out_html):
    m = folium.Map((lat[0], lon[0]), zoom_start=ZOOM_START)
    add_track(lat, lon, m)
    add_start_end(lat, lon, m)
    m.save(out_html)


# ---------- per-fil -----------------------------------------------------
def process(jf: Path, track: str):
    stem = jf.stem
    if stem.endswith(".smoothed") or stem.endswith(".removed"):
        return

    # ----- ladda rådata --------------------------------------------------
    lat_all, lon_all, ts_all, rows_all = load_json(jf)

    # ----- 1) släng punkter långt från mittlinjen ------------------------
    keep_center = np.array([dist_to_center(la, lo, track) <= MAX_DIST_M
                            for la, lo in zip(lat_all, lon_all)])
    removed_center = ~keep_center & ~np.isnan(lat_all)

    if keep_center.sum() < 2:
        print(f"✗ {stem}: <2 punkter inom {MAX_DIST_M} m – hoppar"); return

    lat, lon, ts = lat_all[keep_center], lon_all[keep_center], ts_all[keep_center]
    rows = [rows_all[i] for i,k in enumerate(keep_center) if k]

    # ----- 2) core mask (outliers, hopp, fart) ---------------------------
    bad = core_mask(lat, lon, ts)             # True => slängs
    removed_core = bad

    lat_s, lon_s = smooth(lat, lon, ~bad)

    # ----- JSON med validerade punkter ----------------------------------
    kept_rows = []
    for ok, row, la, lo in zip(~bad, rows, lat_s, lon_s):
        if not ok:
            continue
        r = dict(row)
        r.setdefault("locationData", {})
        r["locationData"]["latitude"]  = f"{la}"
        r["locationData"]["longitude"] = f"{lo}"
        kept_rows.append(r)

    out_json = jf.with_suffix("").parent / f"{stem}.removed.json"
    out_json.unlink(missing_ok=True)
    json.dump({"timestamps": kept_rows}, out_json.open("w"), indent=2)

    # ----- HTML kartor ---------------------------------------------------
    plot = jf.parent / "plot"
    plot.mkdir(exist_ok=True)

    # samtliga borttagna – center + core
    rm_lat = np.concatenate([lat_all[removed_center],
                             lat[removed_core]])
    rm_lon = np.concatenate([lon_all[removed_center],
                             lon[removed_core]])

    write_smoothed_map(lat_s, lon_s,
                       rm_lat, rm_lon,
                       plot / f"{stem}.smoothed.html")

    write_removed_map(lat_s, lon_s,
                      plot / f"{stem}.removed.html")

    print(f"✓ {stem}: kept {len(kept_rows)}/{len(lat_all)}")


# ---------- batch-körning ----------------------------------------------
def main():
    for day in DAYS:
        for track in TRACKS:
            root = BASE_DIR / day / track
            if not root.is_dir(): continue
            for pdir in root.glob("P*"):
                for jf in pdir.glob("*.Cadence.json"):
                    process(jf, track)

if __name__ == "__main__":
    main()
