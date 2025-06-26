#!/usr/bin/env python3
"""
add_cadence.py  – write cadence as STRING and never create
                  *.cadence.cadence.json duplicates
"""

from pathlib import Path
import json, math

# ─── settings ────────────────────────────────────────────────────────────────
WHEEL_CIRCUMFERENCE_M = 2.3
BASE_DIR = Path("./Unsegmented")
PARTICIPANTS = [
    "P01", "P02", "P03", "P04", "P05", "P06", "P07",
    "P09", "P10", "P11", "P12", "P13", "P14", "P15", "P16",
]
DAYS   = ["Day1", "Day2"]
TRACKS = ["Track-A", "Track-B"]

DECIMAL_PLACES = 2
_FMT = f"{{:.{DECIMAL_PLACES}f}}".format   # e.g. _FMT(78.2559) → "78.26"

# ─── helpers ──────────────────────────────────────────────────────────────────
def cadence_as_str(vel) -> str:
    try:
        vel = float(vel)
    except (TypeError, ValueError):
        vel = -1
    if vel == -1:
        return "NaN"
    return _FMT((vel * 60) / WHEEL_CIRCUMFERENCE_M)


def output_path_for(src: Path) -> Path:
    """
    • logfile.json       → logfile.cadence.json
    • logfile.cadence.json (already processed)
                         → same path (overwrite in place)
    """
    if src.name.endswith(".cadence.json"):
        return src                       # already a cadence file – overwrite
    return src.with_name(f"{src.stem}.cadence.json")


def process_file(src: Path) -> None:
    """Inject/overwrite string-typed 'cadence' and save to correct file."""
    with src.open() as f:
        data = json.load(f)

    for ts in data.get("timestamps", []):
        vel = ts.get("locationData", {}).get("velocity", -1)
        ts["cadence"] = cadence_as_str(vel)

    dst = output_path_for(src)
    with dst.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"✓ {src.relative_to(BASE_DIR)}  →  {dst.name}")


# ─── main walk ────────────────────────────────────────────────────────────────
def main() -> None:
    total = 0
    for day in DAYS:
        for track in TRACKS:
            for pid in PARTICIPANTS:
                folder = BASE_DIR / day / track / pid
                if not folder.exists():
                    continue
                for src in folder.glob("*-logfile-subject-*.json"):
                    total += 1
                    process_file(src)
    print(f"Done – {total} files processed / overwritten as needed.")

if __name__ == "__main__":
    main()
