#!/usr/bin/env python3
"""
plot_laps.py
============

Create a Folium HTML map for every ride logfile under

    ./Unsegmented/Day{1,2}/Track-{A,B}/P??/*.json

Each map is saved inside a sibling `plot/` directory:

    .../P01/plot/<basename>_map.html
"""

import os
import json
import folium
from pathlib import Path

# --------------------------------------------------------------------------- #
#                               core plotter                                  #
# --------------------------------------------------------------------------- #
def plot_lap(file_path: Path, output_file: Path) -> None:
    """Read one *.json logfile and write an HTML map."""
    with file_path.open() as fp:
        data = json.load(fp)

    lap_data = data.get("timestamps", [])
    if not lap_data:
        print(f"[WARN] {file_path}: no 'timestamps' array – skipped.")
        return

    try:
        latitudes  = [float(ts["locationData"]["latitude"])  for ts in lap_data]
        longitudes = [float(ts["locationData"]["longitude"]) for ts in lap_data]
    except (KeyError, ValueError, TypeError):
        print(f"[WARN] {file_path}: bad lat/lon fields – skipped.")
        return

    start_location = (latitudes[0], longitudes[0])
    fmap = folium.Map(location=start_location, zoom_start=16)

    # Draw points
    for lat, lon in zip(latitudes, longitudes):
        folium.CircleMarker(
            location=(lat, lon),
            radius=3,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.6,
        ).add_to(fmap)

    # Draw polyline, start & end markers
    folium.PolyLine(list(zip(latitudes, longitudes)),
                    color="red", weight=2.5, opacity=0.8).add_to(fmap)

    folium.Marker(start_location,
                  popup="Start",
                  icon=folium.Icon(color="green")).add_to(fmap)

    folium.Marker((latitudes[-1], longitudes[-1]),
                  popup="End",
                  icon=folium.Icon(color="red")).add_to(fmap)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(output_file)
    print(f"✓ map saved → {output_file.relative_to(Path.cwd())}")


# --------------------------------------------------------------------------- #
#                               batch runner                                  #
# --------------------------------------------------------------------------- #
def process_all_laps(base_dir: Path) -> None:
    """Walk Day?/Track-{A,B}/P??/ and plot every JSON logfile."""
    base_dir = base_dir.resolve()
    days    = ["Day1", "Day2"]
    tracks  = ["Track-A", "Track-B"]          #  ←  now handles both tracks

    for day in days:
        for track in tracks:
            track_path = base_dir / day / track
            if not track_path.exists():
                continue

            for participant_path in track_path.iterdir():
                if not participant_path.is_dir():
                    continue

                for json_file in participant_path.glob("*Cadence.json"):
                    plot_dir = participant_path / "plot"
                    output_file = plot_dir / f"{json_file.stem}_map.html"
                    plot_lap(json_file, output_file)


# --------------------------------------------------------------------------- #
#                                 main                                        #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    BASE_DIRECTORY = Path("./Unsegmented")   # change if your root differs
    process_all_laps(BASE_DIRECTORY)
