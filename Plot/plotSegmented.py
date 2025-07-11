#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import folium

# ────────────────────────────────────────────────────────────────────────────
# Configuration – adjust only if your folder names change
# ────────────────────────────────────────────────────────────────────────────
SEGMENTED_ROOT = Path(".././Segmented/Day1")  # where P01 … P15 live
PARTICIPANTS = [p for p in range(1, 17) if p != 8]  # P01 … P16
LAP_PATTERN = "lap_*.json"  # match all lap files


# ────────────────────────────────────────────────────────────────────────────
# Plotting helper (your original function with one print-line tweak)
# ────────────────────────────────────────────────────────────────────────────
def plot_lap(file_path: Path, output_file: Path) -> None:
    with file_path.open() as file:
        lap_data = json.load(file)

    # Extract latitude and longitude
    latitudes = [float(e["locationData"]["latitude"]) for e in lap_data]
    longitudes = [float(e["locationData"]["longitude"]) for e in lap_data]

    # Create a map centred around the starting point
    start_location = (latitudes[0], longitudes[0])
    m = folium.Map(location=start_location, zoom_start=16)

    # Add markers
    for lat, lon in zip(latitudes, longitudes):
        folium.CircleMarker(
            location=(lat, lon),
            radius=3,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.6,
        ).add_to(m)

    # Draw the route
    folium.PolyLine(
        list(zip(latitudes, longitudes)),
        color="red",
        weight=2.5,
        opacity=0.8,
    ).add_to(m)

    # Mark start / finish
    folium.Marker(
        start_location,
        popup="Start",
        icon=folium.Icon(color="green"),
    ).add_to(m)
    folium.Marker(
        (latitudes[-1], longitudes[-1]),
        popup="End",
        icon=folium.Icon(color="red"),
    ).add_to(m)

    m.save(str(output_file))

    # ——— fixed: always print a usable path on any OS ———
    print(f"    ✓ map saved → {output_file.resolve()}")


# ────────────────────────────────────────────────────────────────────────────
# Batch loop
# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    for n in PARTICIPANTS:
        pid = f"P{n:02d}"
        laps_dir = SEGMENTED_ROOT / pid
        plot_dir = laps_dir / "plot"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Find every lap_<n>.json in the participant folder
        lap_files: List[Path] = sorted(laps_dir.glob(LAP_PATTERN))
        if not lap_files:
            print(f"[!] {pid}: no lap files found, skipping.")
            continue

        print(f"{pid}: plotting {len(lap_files)} lap(s)…")
        for lap_path in lap_files:
            out_html = plot_dir / f"{lap_path.stem}_map.html"
            plot_lap(lap_path, out_html)

    print("\n✓ All laps plotted.")


if __name__ == "__main__":
    main()
