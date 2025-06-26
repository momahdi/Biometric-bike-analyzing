#!/usr/bin/env python3
"""
batch_plot.py
=============

Walk the folder tree

    ./Unsegmented/Day*/Track-*/P??/

For every logfile whose filename ends in "cadence.json"
(case-insensitive, e.g. 2024-12-10_115555-logfile-subject-P01_A.cadence.json),
create a multi-row PNG summary plot with timestamp on the X-axis.

The PNG is written right next to the source JSON with the same basename
but a .png extension.

Run:

    python batch_plot.py
"""

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # kept for future tweaks (e.g. smoothing); safe to remove if unused

# --------------------------------------------------------------------------- #
#                               configuration                                 #
# --------------------------------------------------------------------------- #
BASE_DIR = Path("./Unsegmented/")
PARTICIPANTS = [
    "P01", "P02", "P03", "P04", "P05", "P06", "P07",
    "P09", "P10", "P11", "P12", "P13", "P14", "P15", "P16",
]

# One subplot per dictionary entry (title : list-of-column-names)
GROUPS = {
    "Cadence (rpm)":                 ["cadence"],
    "Velocity (m·s⁻¹)":              ["locationData.velocity"],     # dedicated row
    "Brake state":                   ["brakeData"],
    "Pedal force balance":           ["pedalWeight.L", "pedalWeight.R"],
    "Acceleration (g)":              ["acceleration.x",
                                      "acceleration.y",
                                      "acceleration.z"],
    "User-accel (g, no gravity)":    ["userAccel.x",
                                      "userAccel.y",
                                      "userAccel.z"],
    "Gyro (rad/s)":                  ["rotationRate.x",
                                      "rotationRate.y",
                                      "rotationRate.z"],
    "Attitude (°)":                  ["roll", "pitch", "yaw"],
    "Magnetic field (µT)":           ["magneticField.x",
                                      "magneticField.y",
                                      "magneticField.z"],
    "Location altitude (m)":         ["locationData.altitude"],      # altitude only
}
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
#                               plot helper                                   #
# --------------------------------------------------------------------------- #
def plot_json(json_path: Path) -> None:
    """Parse one *.cadence.json file and save a PNG beside it."""
    with json_path.open() as f:
        js = json.load(f)

    # Flatten nested dicts into dot-separated columns
    df = pd.json_normalize(js.get("timestamps", []))

    if "timestamp" not in df.columns:
        print(f"[WARN] {json_path.name}: no 'timestamp' column – skipped.")
        return

    # quick numeric coercion helper
    to_num = lambda col: pd.to_numeric(df[col], errors="coerce")

    time = to_num("timestamp")

    nrows = len(GROUPS)
    fig, axes = plt.subplots(
        nrows, 1,
        figsize=(13, 2.6 * nrows),
        sharex=True
    )

    for ax, (title, cols) in zip(axes, GROUPS.items()):
        plotted = False
        for c in cols:
            if c in df.columns:
                ax.plot(time, to_num(c), label=c, linewidth=0.8)
                plotted = True
        if not plotted:
            ax.text(
                0.5, 0.5, "no data",
                ha="center", va="center",
                transform=ax.transAxes
            )
        ax.set_title(title, loc="left", fontsize=10)
        ax.legend(fontsize=7, ncol=4)
        ax.grid(True, alpha=.3)

    axes[-1].set_xlabel("timestamp  (s from session start)")
    plt.tight_layout()

    out_png = json_path.with_suffix(".png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    print(f"✓ {json_path.relative_to(BASE_DIR.parent)} → {out_png.name}")
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
#                                main driver                                  #
# --------------------------------------------------------------------------- #
def main() -> None:
    if not BASE_DIR.exists():
        print(f"BASE_DIR '{BASE_DIR}' not found.")
        return

    file_count = 0
    for day_dir in sorted(BASE_DIR.glob("Day*")):
        for track_dir in sorted(day_dir.glob("Track-*")):
            for pid in PARTICIPANTS:
                p_dir = track_dir / pid
                if not p_dir.is_dir():
                    continue

                # accept *any* filename that ends with cadence.json (case-insensitive)
                for jfile in p_dir.iterdir():
                    if jfile.is_file() and jfile.name.lower().endswith("cadence.json"):
                        plot_json(jfile)
                        file_count += 1

    print(f"All plots generated ({file_count} files).")


if __name__ == "__main__":
    main()
