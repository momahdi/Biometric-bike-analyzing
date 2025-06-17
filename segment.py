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
PARTICIPANTS      = range(1, 16)            # P01 … P15
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
START_RADIUS_M = 5          # metres

OUTSIDE_POINTS_REQUIRED = 12  # 12 samples @1 Hz ≈ 12 s outside
MIN_LAP_POINTS        = 50   # reject laps < 50 samples (≈ 50 s @1 Hz)
MAX_LAPS              = 5    # 0 ➜ no limit

# ────────────────────────────────────────────────────────────────────────────
# Type aliases
# ────────────────────────────────────────────────────────────────────────────
Timestamp = Dict[str, Any]
Lap = List[Timestamp]
Point = Tuple[float, float]  # (latitude, longitude)

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
    Segment the GPS stream into laps.

    Behaviour differences vs. original
    ───────────────────────────────────
    1. **Recording starts only when the rider first enters the start circle.**
    2. **If the file ends while the rider is still outside the circle, that
       partial lap is kept** (provided it meets *min_points*).
    """
    laps: List[Lap] = []
    current_lap: Lap = []

    recording = False
    outside_counter = 0

    for idx, sample in enumerate(timestamps):

        # ── 1. pull the position out of the record ───────────────────────────
        loc = sample["locationData"]
        lat = float(loc["latitude"])
        lon = float(loc["longitude"])
        inside = _inside_circle((lat, lon), centre, radius_m)

        # ── 2. debug print (comment out once happy) ──────────────────────────
        ts = sample.get("timestamp", idx)
        print(
            f"{idx:05d} {ts} "
            f"{'IN' if inside else 'out':>3} "
            f"lat={lat:.6f} lon={lon:.6f} "
            f"outside_counter={outside_counter:2d} "
            f"recording={recording}"
        )

        # ── 3. state machine ────────────────────────────────────────────────
        if not recording:
            if inside:                               # first hit = start lap
                current_lap = [sample]
                recording = True
                outside_counter = 0
        else:
            current_lap.append(sample)

            if inside:
                # potential normal finish
                if outside_counter >= outside_needed and len(current_lap) >= min_points:
                    laps.append(current_lap)
                    current_lap = []
                    recording = False
                    outside_counter = 0
            else:
                outside_counter += 1

    # ── 4. file ended while still recording outside the circle ──────────────
    if recording and len(current_lap) >= min_points:
        laps.append(current_lap)

    return laps

# ────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ────────────────────────────────────────────────────────────────────────────

def _load_timestamps(path: Path) -> List[Timestamp]:
    """Return the ``timestamps`` array from *path* or abort if absent."""
    with path.open() as fp:
        data = json.load(fp)
    try:
        return data["timestamps"]
    except KeyError as err:
        raise SystemExit('No "timestamps" array found in the input file.') from err


def _save_laps(laps: List[Lap], directory: Path) -> None:
    """Write each *lap* to ``lap_<n>.json`` inside *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    for idx, lap in enumerate(laps, start=1):
        target = directory / f"lap_{idx}.json"
        target.write_text(json.dumps(lap, indent=4))
        print(f"Wrote {target}")

# ────────────────────────────────────────────────────────────────────────────
# NEW: helper to grab the first *.json file in a participant folder
# ────────────────────────────────────────────────────────────────────────────
def _first_json(directory: Path) -> Path | None:
    """Return the first *.json file found in *directory* (or *None* if absent)."""
    for child in sorted(directory.iterdir()):
        if child.suffix.lower() == ".json":
            return child
    return None

# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────
def main() -> None:  # noqa: D401
    if not RUN_BATCH:
        # ── original single-file mode ───────────────────────────────────────
        input_path = INPUT_FILE.expanduser().resolve()
        if not input_path.is_file():
            raise SystemExit(f"Input file not found: {input_path}")

        timestamps = _load_timestamps(input_path)

        laps = extract_laps(
            timestamps,
            centre=(START_LAT, START_LON),
            radius_m=START_RADIUS_M,
            outside_needed=OUTSIDE_POINTS_REQUIRED,
            min_points=MIN_LAP_POINTS,
        )

        if MAX_LAPS > 0:
            laps = laps[:MAX_LAPS]

        _save_laps(laps, OUTPUT_DIR.expanduser().resolve())
        print(f"Extracted {len(laps)} lap(s).")

    else:
        # ── NEW: batch mode P01 … P15 ───────────────────────────────────────
        centre = (START_LAT, START_LON)

        for n in PARTICIPANTS:
            pid     = f"P{n:02d}"
            in_dir  = UNSEGMENTED_ROOT / pid
            out_dir = SEGMENTED_ROOT   / pid

            if not in_dir.is_dir():
                print(f"[!] {pid}: directory not found, skipping.")
                continue

            json_file = _first_json(in_dir)
            if json_file is None:
                print(f"[!] {pid}: no JSON file found, skipping.")
                continue

            print(f"{pid}: processing {json_file.name}")
            timestamps = _load_timestamps(json_file)

            laps = extract_laps(
                timestamps,
                centre=centre,
                radius_m=START_RADIUS_M,
                outside_needed=OUTSIDE_POINTS_REQUIRED,
                min_points=MIN_LAP_POINTS,
            )
            if MAX_LAPS > 0:
                laps = laps[:MAX_LAPS]

            _save_laps(laps, out_dir)

        print("\n✓ Finished processing all participants.")


if __name__ == "__main__":
    main()
