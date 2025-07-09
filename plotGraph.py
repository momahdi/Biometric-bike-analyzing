#!/usr/bin/env python3
"""
batch_plot.py
=============

Walk the folder tree

    ./Unsegmented/Day*/Track-*/P??/

For every logfile whose filename ends in "cadence.json" or "removed.json"
(case‑insensitive), create a multi‑row PNG summary plot with timestamp on
X‑axis.

**Tweaks**
----------
* Cadence values are hard‑capped at **200 rpm** before plotting.
* Track duration (hh:mm:ss) is shown in the figure title.
* **NEW:** NaN‑värden droppas före plotten så linjerna blir obrutna och
  y‑skalor auto‑anpassar sig bättre.

Run:

    python batch_plot.py
"""

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

# --------------------------------------------------------------------------- #
#                               configuration                                 #
# --------------------------------------------------------------------------- #
BASE_DIR = Path("./Unsegmented/")
PARTICIPANTS = [
    "P01", "P02", "P03", "P04", "P05", "P06", "P07",
    "P09", "P10", "P11", "P12", "P13", "P14", "P15", "P16",
]

# One subplot per dictionary entry (title : list‑of‑column‑names)
GROUPS = {
    "Acceleration (g)":              ["acceleration.x", "acceleration.y", "acceleration.z"],
    "Garmin Cadence (rpm)":          ["cadence"],
    "Phone Cadence (rpm)":           ["phoneCadence"],
    "Velocity (m·s⁻¹)":              ["locationData.velocity"],
    "Brake state":                   ["brakeData"],
    "Pedal force balance":           ["pedalWeight.L", "pedalWeight.R"],
    "User‑accel (g, no gravity)":    ["userAccel.x", "userAccel.y", "userAccel.z"],
    "Gyro (rad/s)":                  ["rotationRate.x", "rotationRate.y", "rotationRate.z"],
    "Attitude (°)":                  ["roll", "pitch", "yaw"],
    "Magnetic field (µT)":           ["magneticField.x", "magneticField.y", "magneticField.z"],
    "Location altitude (m)":         ["locationData.altitude"],
}
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
#                               plot helper                                   #
# --------------------------------------------------------------------------- #

def _format_dur(seconds: float) -> str:
    """Float seconds → HH:MM:SS"""
    return str(timedelta(seconds=int(round(seconds))))


def plot_json(json_path: Path) -> None:
    """Parse one *.cadence.json or *.removed.json file and save a PNG beside it."""
    with json_path.open() as f:
        js = json.load(f)

    # Flatten nested dicts into dot‑separated columns
    df = pd.json_normalize(js.get("timestamps", []))

    if "timestamp" not in df.columns:
        print(f"[WARN] {json_path.name}: no 'timestamp' column – skipped.")
        return

    # shorthand numeric coercion
    to_num = lambda col: pd.to_numeric(df[col], errors="coerce")

    # Prep cadence column separately (apply cap)
    if "cadence" in df.columns:
        df["cadence"] = to_num("cadence").clip(upper=200)

    # Numeric time vector
    time = to_num("timestamp")
    dur_seconds = (time.max() - time.min()) if len(time.dropna()) > 1 else 0

    # --------------------------------------------------------------------
    # Plotting
    # --------------------------------------------------------------------
    nrows = len(GROUPS)
    fig, axes = plt.subplots(
        nrows, 1,
        figsize=(13, 2.6 * nrows),
        sharex=True,
    )

    for ax, (title, cols) in zip(axes, GROUPS.items()):
        plotted = False
        for c in cols:
            if c not in df.columns:
                continue

            y = to_num(c)
            if title.startswith("Cadence"):
                y = y.clip(upper=200)

            # ----- drop NaNs --------------------------------------------------
            mask = (~time.isna()) & (~y.isna())
            if mask.any():
                ax.plot(time[mask], y[mask], label=c, linewidth=0.8)
                plotted = True
        if not plotted:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, loc="left", fontsize=10)
        ax.legend(fontsize=7, ncol=4)
        ax.grid(True, alpha=.3)

    axes[-1].set_xlabel("timestamp  (s from session start)")
    plt.tight_layout()

    # Overall title with duration
    fig.subplots_adjust(top=0.92)
    fig.suptitle(f"{json_path.stem}   –   track time: {_format_dur(dur_seconds)}",
                 fontsize=12, y=0.99)

    # Save PNG
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

                # accept any filename that ends with cadence.json OR removed.json
                for jfile in p_dir.iterdir():
                    if not jfile.is_file():
                        continue
                    if jfile.name.lower().endswith("removed.json"):
                        plot_json(jfile)
                        file_count += 1

    print(f"All plots generated ({file_count} files).")


if __name__ == "__main__":
    main()
