#!/usr/bin/env python3
"""
smooth_and_connect_clean.py  –  distance-to-centre-line version
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
_tf = Transformer.from_crs(4326, 3857, always_xy=True)  # lon/lat → metres

def haversine(a, b, c, d):                # ▲ vectorised version
    """
    a,b = lat,lon of P1   (arrays OK)
    c,d = lat,lon of P2
    returns metres
    """
    a, b, c, d = map(np.radians, (a, b, c, d))
    da = c - a
    dl = d - b
    h = np.sin(da/2)**2 + np.cos(a)*np.cos(c)*np.sin(dl/2)**2
    return 2 * 6_371_000 * np.arcsin(np.sqrt(h))

def to_xy(lat, lon):
    return _tf.transform(lon, lat)        # arrays in → arrays out

def hampel(arr, k=HAMPEL_WIN, s=HAMPEL_SIGMA):
    if len(arr) < k or k % 2 == 0:
        return np.zeros(len(arr), bool)
    h = k//2
    mask = np.zeros(len(arr), bool)
    for i in range(h, len(arr)-h):
        win = arr[i-h:i+h+1]
        med = np.median(win)
        mad = np.median(np.abs(win-med)) or 1e-6
        mask[i] = abs(arr[i]-med) > s*mad
    return mask


# ---------- load profile line (projected) ------------------------------
def load_profile(track):
    p = BASE_DIR / f"profile-{track}.geojson"
    if not p.is_file():
        p = BASE_DIR / "plot" / f"profile-{track}.geojson"
    if not p.is_file():
        raise FileNotFoundError(f"GeoJSON for {track} not found")
    coords_ll = json.load(p.open())["features"][0]["geometry"]["coordinates"]
    coords_xy = [_tf.transform(lon, lat) for lon, lat in coords_ll]
    return LineString(coords_xy)

PROFILE = {trk: load_profile(trk) for trk in TRACKS}

def dist_to_center(lat, lon, track):
    return PROFILE[track].distance(Point(*to_xy(lat, lon)))


# ---------- json helpers -----------------------------------------------
def ffloat(v):
    try: return float(v)
    except (TypeError, ValueError): return np.nan

def load_json(fp):
    rows = json.load(fp.open())["timestamps"]
    lat, lon, ts = [], [], []
    for r in rows:
        lat.append(ffloat(r.get("locationData",{}).get("latitude")))
        lon.append(ffloat(r.get("locationData",{}).get("longitude")))
        ts.append(ffloat(r.get("timestamp")))
    lat, lon, ts = map(np.array, (lat, lon, ts))
    if FILL_TIME or np.isnan(ts).any() or len(np.unique(ts))!=len(ts):
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
    x, y = to_xy(lat, lon)                # ▲ direct transformer
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


# ---------- folium drawing ---------------------------------------------
def draw(lat, lon, m, color="blue", w=2):
    folium.PolyLine(list(zip(lat, lon)), color=color, weight=w).add_to(m)

def draw_map(lat, lon, o_lat, o_lon, html):
    m = folium.Map((lat[0], lon[0]), zoom_start=ZOOM_START)
    draw(lat, lon, m, "red", 2)
    for la, lo in zip(o_lat, o_lon):
        folium.CircleMarker((la, lo), 4, color="orange",
                            fill=True, fill_color="orange",
                            fill_opacity=.9).add_to(m)
    folium.Marker((lat[0], lon[0]), icon=folium.Icon(color="green"),
                  popup="Start").add_to(m)
    folium.Marker((lat[-1], lon[-1]), icon=folium.Icon(color="red"),
                  popup="End").add_to(m)
    m.save(html)


# ---------- per-file ----------------------------------------------------
def process(jf, track):
    lat, lon, ts, rows = load_json(jf)
    keep = np.array([dist_to_center(la, lo, track) <= MAX_DIST_M
                     for la, lo in zip(lat, lon)])
    if keep.sum() < 2:
        print(f"✗ {jf.stem}: <2 pts within {MAX_DIST_M} m – skipped"); return

    lat, lon, ts = lat[keep], lon[keep], ts[keep]
    rows = [rows[i] for i,k in enumerate(keep) if k]

    bad = core_mask(lat, lon, ts)
    lat_s, lon_s = smooth(lat, lon, ~bad)

    kept=[]
    for ok,row,la,lo in zip(~bad, rows, lat_s, lon_s):
        if ok:
            r=dict(row); r.setdefault("locationData",{})
            r["locationData"].update({"latitude":f"{la}",
                                      "longitude":f"{lo}"})
            kept.append(r)

    out = jf.with_suffix("").parent / f"{jf.stem}.removed.json"
    out.unlink(missing_ok=True)
    json.dump({"timestamps": kept}, out.open("w"), indent=2)

    plot = jf.parent / "plot"; plot.mkdir(exist_ok=True)
    draw_map(lat_s, lon_s, lat[bad], lon[bad],
             plot / f"{jf.stem}.smoothed.html")
    draw_map(lat_s, lon_s, [], [],
             plot / f"{jf.stem}.removed.html")

    print(f"✓ {jf.stem}: kept {len(kept)}/{len(lat)+bad.sum()}")

# ---------- batch driver -----------------------------------------------
def main():
    for day in DAYS:
        for track in TRACKS:
            root = BASE_DIR / day / track
            if not root.is_dir(): continue
            for pdir in root.glob("P*"):
                for jf in pdir.glob("*.json"):
                    process(jf, track)

if __name__ == "__main__":
    main()
