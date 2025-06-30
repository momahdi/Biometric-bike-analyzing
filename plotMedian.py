#!/usr/bin/env python3
"""
dual_profile_builder.py
-----------------------

• Track-A  → unchanged: your index-median + 4× K-nearest refinement
• Track-B  → medoid reference + iterative nearest-point centre,
             output is ONE line that follows the centre of the '8'

Outputs
  Unsegmented/profile-Track-A.geojson
  Unsegmented/plot/Track-A-profile.html
  Unsegmented/profile-Track-B.geojson
  Unsegmented/plot/Track-B-profile.html
"""

from pathlib import Path
import json, math, numpy as np, folium
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree, distance

# ─── CONFIG ────────────────────────────────────────────────────────────
BASE_DIR           = Path("Unsegmented")
N_POINTS           = 500
HAMPEL_K, HAMPEL_S = 11, 3.0

# centre-line refinement
ITERATIONS_A       = 4     # Track-A   (unchanged)
K_NEIGH_A          = 150
ITERATIONS_B       = 5     # Track-B   (little extra tightening)
K_NEIGH_B          = 200

ZOOM_START         = 16
# ────────────────────────────────────────────────────────────────────────


# ---------- generic helpers -------------------------------------------
def pretty(p: Path):
    try: return p.relative_to(Path.cwd())
    except ValueError: return p

def haversine(lat1,lon1,lat2,lon2):
    R = 6_371_000
    p1,p2=map(math.radians,(lat1,lat2))
    dlat,dlon=p2-p1,math.radians(lon2-lon1)
    return 2*R*math.asin(math.sqrt(
        math.sin(dlat/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dlon/2)**2))

def to_xy(lat,lon):
    R=6_371_000; lat0=np.nanmean(lat)
    x=np.radians(lon-lon.mean())*R*math.cos(math.radians(lat0))
    y=np.radians(lat-lat.mean())*R
    return x,y

def hampel(arr,k=HAMPEL_K,s=HAMPEL_S):
    if len(arr)<k or k%2==0: return arr
    h=k//2; out=arr.copy()
    for i in range(h,len(arr)-h):
        win=arr[i-h:i+h+1]
        med=np.median(win)
        mad=np.median(np.abs(win-med)) or 1e-6
        if abs(arr[i]-med)>s*mad:
            out[i]=med
    return out

def load_lat_lon(fp:Path):
    rows=json.load(fp.open())["timestamps"]
    lat,lon=[],[]
    for r in rows:
        try:
            lat.append(float(r["locationData"]["latitude"]))
            lon.append(float(r["locationData"]["longitude"]))
        except (KeyError,ValueError):
            continue
    return np.array(lat),np.array(lon)

def resample(lat,lon,n=N_POINTS):
    lat,lon=hampel(lat),hampel(lon)
    step=[0]+[haversine(lat[i-1],lon[i-1],lat[i],lon[i])
              for i in range(1,len(lat))]
    cum=np.cumsum(step)
    if cum[-1]==0: return None
    tgt=np.linspace(0,cum[-1],n)
    return interp1d(cum,lat)(tgt),interp1d(cum,lon)(tgt)

def rms(a_lat,a_lon,b_lat,b_lon):
    return np.mean((a_lat-b_lat)**2+(a_lon-b_lon)**2)

def best_shift(a_lat,a_lon,ref_lat,ref_lon):
    n=len(a_lat); best,idx=float("inf"),0
    for k in range(n):
        err=rms(np.roll(a_lat,k),np.roll(a_lon,k),ref_lat,ref_lon)
        if err<best: best,idx=err,k
    return np.roll(a_lat,idx),np.roll(a_lon,idx)

# ---------- centre-line refinement (iterative KNN) ---------------------
def refine_path(init_lat, init_lon, cloud_lat, cloud_lon,
                k_neigh, iterations):
    x, y = to_xy(cloud_lat, cloud_lon)
    tree = cKDTree(np.c_[x, y])
    lat,lon=init_lat.copy(),init_lon.copy()
    for _ in range(iterations):
        x,y=to_xy(lat,lon)
        new_lat,new_lon=[],[]
        for xi,yi in zip(x,y):
            _,idx=tree.query([xi,yi],k=k_neigh)
            new_lat.append(np.median(cloud_lat[idx]))
            new_lon.append(np.median(cloud_lon[idx]))
        lat,lon=np.array(new_lat),np.array(new_lon)
    return lat,lon

# ---------- MAD envelope ----------------------------------------------
def envelope(lat,lon,laps_lat,laps_lon):
    dev=np.vstack([np.sqrt((la-lat)**2+(lo-lon)**2)
                   for la,lo in zip(laps_lat,laps_lon)])
    mad=np.median(dev,axis=0)
    return lat+mad, lon+mad, lat-mad, lon-mad

# ---------- profile writer --------------------------------------------
def write_outputs(track, centre_lat, centre_lon,
                  env_lat_p, env_lon_p, env_lat_m, env_lon_m,
                  laps_lat, laps_lon):
    # geojson
    coords=[[float(lo),float(la)] for la,lo in zip(centre_lat,centre_lon)]
    gj={"type":"FeatureCollection",
        "features":[{"type":"Feature",
                     "properties":{"track":track,"points":N_POINTS},
                     "geometry":{"type":"LineString",
                                 "coordinates":coords}}]}
    out=BASE_DIR/f"profile-{track}.geojson"
    json.dump(gj,out.open("w"),indent=2)
    print(f"✓ wrote {pretty(out)}")

    # folium map
    m=folium.Map((centre_lat[0],centre_lon[0]),zoom_start=ZOOM_START)
    for la,lo in zip(laps_lat,laps_lon):
        folium.PolyLine(list(zip(la,lo)),
                        color="#888",weight=1,opacity=.35).add_to(m)
    folium.PolyLine(list(zip(env_lat_p,env_lon_p)),
                    color="#85C1E9",weight=2,opacity=.6,
                    dash_array="4").add_to(m)
    folium.PolyLine(list(zip(env_lat_m,env_lon_m)),
                    color="#85C1E9",weight=2,opacity=.6,
                    dash_array="4").add_to(m)
    folium.PolyLine(list(zip(centre_lat,centre_lon)),
                    color="blue",weight=4,opacity=.9).add_to(m)
    plot=BASE_DIR/"plot"; plot.mkdir(exist_ok=True)
    html=plot/f"{track}-profile.html"
    html.unlink(missing_ok=True); m.save(html)
    print(f"✓ wrote {pretty(html)}\n")

# ---------- builder ----------------------------------------------------
def build_track_a():         # original behaviour
    build_generic("Track-A", ITERATIONS_A, K_NEIGH_A,
                  reference="first")

def build_track_b():         # special: medoid reference, one-lap figure 8
    build_generic("Track-B", ITERATIONS_B, K_NEIGH_B,
                  reference="medoid")

def build_generic(track, iterations, k_neigh, reference="first"):
    laps_lat, laps_lon = [], []

    for day in BASE_DIR.glob("Day*"):
        root=day/track
        if not root.is_dir(): continue
        for pdir in root.glob("P*"):
            for jf in pdir.glob("*.json"):
                lat,lon=load_lat_lon(jf)
                if len(lat)<20: continue
                rs=resample(lat,lon)
                if rs:
                    laps_lat.append(rs[0]); laps_lon.append(rs[1])
    if not laps_lat:
        print(f"[WARN] no laps for {track}"); return

    # --- choose reference lap ---------------------------------------
    if reference=="first":
        ref_lat, ref_lon = laps_lat[0], laps_lon[0]
    else:                     # medoid
        vecs=[np.r_[la,lo] for la,lo in zip(laps_lat,laps_lon)]
        D=distance.squareform(distance.pdist(vecs,'euclidean'))
        idx=np.argmin(D.sum(axis=0))
        ref_lat, ref_lon = laps_lat[idx], laps_lon[idx]

    # --- orient + phase ---------------------------------------------
    al_lat, al_lon = [], []
    for la,lo in zip(laps_lat,laps_lon):
        if rms(la[::-1],lo[::-1],ref_lat,ref_lon) < rms(la,lo,ref_lat,ref_lon):
            la,lo=la[::-1],lo[::-1]
        la,lo = best_shift(la,lo,ref_lat,ref_lon)
        al_lat.append(la); al_lon.append(lo)

    # --- initial centre = median ------------------------------------
    centre_lat=np.median(np.vstack(al_lat),axis=0)
    centre_lon=np.median(np.vstack(al_lon),axis=0)

    # --- refine with KNN iterations ---------------------------------
    cloud_lat=np.concatenate(al_lat)
    cloud_lon=np.concatenate(al_lon)
    centre_lat,centre_lon = refine_path(centre_lat,centre_lon,
                                        cloud_lat,cloud_lon,
                                        k_neigh,iterations)

    # --- envelope + outputs -----------------------------------------
    lat_p,lon_p,lat_m,lon_m = envelope(centre_lat,centre_lon,
                                       al_lat,al_lon)
    write_outputs(track,centre_lat,centre_lon,
                  lat_p,lon_p,lat_m,lon_m,
                  al_lat,al_lon)

# ---------- main -------------------------------------------------------
def main():
    build_track_a()
    build_track_b()

if __name__=="__main__":
    main()
