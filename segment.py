#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from geopy.distance import geodesic

# ────────────────────────────────────────────────────────────────────────────
# Configuration ─ edit these values to match your environment
# ────────────────────────────────────────────────────────────────────────────
INPUT_FILE = Path(
    "./Unsegmented/P01/2024-12-10_123330-logfile-subject-P01_B.json"
)
OUTPUT_DIR = Path("./Segmented/P01")

# ─────────── NEW: batch-mode parameters (leave the two lines above untouched) ─
RUN_BATCH         = True                    # False ⇒ original single-file mode
PARTICIPANTS      = [p for p in range(1, 17) if p != 8]     # P01 … P15
UNSEGMENTED_ROOT  = Path("./Unsegmented/Day1/Track-B")
SEGMENTED_ROOT    = Path("./Segmented")
# ──────────────────────────────────────────────────────────────────────────────
# Lap-detection tuning
# • Keep these values in one place so the “figure-8” never splits (false finish)
#   and every genuine crossing of the start/finish line still registers.
# • The defaults below assume:
#       – modern phone/cycling computer (≈ ±3 m typical GNSS wander)
#       – 1 Hz logging rate
#       – lap time ≥ 15 s
#
#   ┌─────────────────────────┬───────────────────────────────────────────────┐
#   │ CONSTANT                │ RULE OF THUMB / WHEN TO TWEAK                 │
#   ├─────────────────────────┼───────────────────────────────────────────────┤
#   │ START_LAT / START_LON   │ Centre point you physically ride through      │
#   │                         │ each lap.  Move this if the crossing point    │
#   │                         │ changes (e.g. new course layout).             │
#   ├─────────────────────────┼───────────────────────────────────────────────┤
#   │ START_RADIUS_M          │ ≈ 1.5 × expected GPS error.                   │
#   │                         │   • Shrink (→ 4 m) if stray fixes still end   │
#   │                         │     the lap too early.                        │
#   │                         │   • Grow (→ 6–7 m) if an occasional lap is    │
#   │                         │     missed because you rode slightly wide.    │
#   ├─────────────────────────┼───────────────────────────────────────────────┤
#   │ OUTSIDE_POINTS_REQUIRED │ Samples that must remain *outside* the        │
#   │                         │ circle before the next crossing counts as a   │
#   │                         │ finish.  =  logging-rate [Hz] × seconds-clear │
#   │                         │   • Increase if a single noisy fix still      │
#   │                         │     closes the lap.                           │
#   ├─────────────────────────┼───────────────────────────────────────────────┤
#   │ MIN_LAP_POINTS          │ Rejects laps that are obviously too short     │
#   │                         │ to be real (e.g. GPS glitch).  Should be      │
#   │                         │ ≳ (expected-lap-time / sample-interval) ÷ 2   │
#   ├─────────────────────────┼───────────────────────────────────────────────┤
#   │ MAX_LAPS                │ 0 → keep every lap detected;                  │
#   │                         │ n → keep only the first *n* laps.             │
#   └─────────────────────────┴───────────────────────────────────────────────┘
# ──────────────────────────────────────────────────────────────────────────────

START_LAT  = 59.346485
START_LON  = 18.072987
START_RADIUS_M         = 7      # wider cone catches every crossing
OUTSIDE_POINTS_REQUIRED = 4      # ~4 s outside before a lap may finish  ← NEW
MIN_LAP_POINTS          = 30     # accept faster 8-shape laps
MIN_LAP_EXTENT_M        = 15     # farthest fix must be ≥15 m away        ← NEW
MAX_LAPS                = 5

# ────────────────────────────────────────────────────────────────────────────
# Type aliases
# ────────────────────────────────────────────────────────────────────────────
Timestamp = Dict[str, Any]
Lap       = List[Timestamp]
Point     = Tuple[float, float]  # (latitude, longitude)

# ────────────────────────────────────────────────────────────────────────────
# Core algorithm
# ────────────────────────────────────────────────────────────────────────────
def _inside_circle(point: Point, centre: Point, radius_m: float) -> bool:
    """Return *True* if *point* lies within *radius_m* metres of *centre*."""
    return geodesic(point, centre).meters < radius_m


def extract_laps(
    timestamps: Iterable[Timestamp],
    centre: Point,
    radius_m: float = START_RADIUS_M,
    outside_needed: int = OUTSIDE_POINTS_REQUIRED,
    min_points: int = MIN_LAP_POINTS,
) -> List[Lap]:
    """
    Recording starts only when the rider first enters the start circle.
    If the file ends while the rider is still outside, that partial lap is kept.
    """
    laps: List[Lap] = []
    current_lap: Lap = []

    recording = False
    outside_counter = 0

    for sample in timestamps:
        loc = sample["locationData"]
        lat, lon = float(loc["latitude"]), float(loc["longitude"])
        inside = _inside_circle((lat, lon), centre, radius_m)

        # ── state machine ────────────────────────────────────────────────
        if not recording:
            if inside:
                current_lap = [sample]
                recording = True
                outside_counter = 0
        else:
            current_lap.append(sample)

            if inside:
                if outside_counter >= outside_needed and len(current_lap) >= min_points:
                    # ── NEW: discard “micro-laps” that never leave a 15 m radius
                    if max(
                        geodesic(
                            (float(s["locationData"]["latitude"]),
                             float(s["locationData"]["longitude"])),
                            centre,
                        ).meters
                        for s in current_lap
                    ) >= MIN_LAP_EXTENT_M:
                        laps.append(current_lap)       # ← lap accepted
                    current_lap = []
                    recording = False
                    outside_counter = 0
            else:
                outside_counter += 1

    # file ended while still on course
    if recording and len(current_lap) >= min_points:
        if max(
            geodesic(
                (float(s["locationData"]["latitude"]),
                 float(s["locationData"]["longitude"])),
                centre,
            ).meters
            for s in current_lap
        ) >= MIN_LAP_EXTENT_M:                         # ← same extent check
            laps.append(current_lap)

    return laps

# ────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ────────────────────────────────────────────────────────────────────────────
def _load_timestamps(path: Path) -> List[Timestamp]:
    with path.open() as fp:
        data = json.load(fp)
    try:
        return data["timestamps"]
    except KeyError as err:
        raise SystemExit('No "timestamps" array found in the input file.') from err


def _save_laps(laps: List[Lap], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for idx, lap in enumerate(laps, start=1):
        (directory / f"lap_{idx}.json").write_text(json.dumps(lap, indent=4))
    print(f"    wrote {len(laps)} lap(s) → {directory}")


# helper unchanged
def _first_json(directory: Path) -> Path | None:
    for child in sorted(directory.iterdir()):
        if child.suffix.lower() == ".json":
            return child
    return None

# ────────────────────────────────────────────────────────────────────────────
# Entry point (unchanged except for banner tweaks)
# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    if not RUN_BATCH:
        input_path = INPUT_FILE.expanduser().resolve()
        if not input_path.is_file():
            raise SystemExit(f"Input file not found: {input_path}")

        laps = extract_laps(
            _load_timestamps(input_path),
            centre=(START_LAT, START_LON),
        )
        _save_laps(laps[:MAX_LAPS or None], OUTPUT_DIR.expanduser().resolve())
        print(f"Extracted {len(laps[:MAX_LAPS or None])} lap(s).")
    else:
        centre = (START_LAT, START_LON)
        for n in PARTICIPANTS:
            pid, in_dir = f"P{n:02d}", UNSEGMENTED_ROOT / f"P{n:02d}"
            out_dir = SEGMENTED_ROOT / pid
            if not in_dir.is_dir():
                print(f"[!] {pid}: directory not found, skipping.")
                continue
            json_file = _first_json(in_dir)
            if json_file is None:
                print(f"[!] {pid}: no JSON file found, skipping.")
                continue

            print(f"{pid}: processing {json_file.name}")
            laps = extract_laps(_load_timestamps(json_file), centre=centre)
            _save_laps(laps[:MAX_LAPS or None], out_dir)

        print("\n✓ Finished processing all participants.")


if __name__ == "__main__":
    main()
