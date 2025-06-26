#!/usr/bin/env python3
"""
clean_cadence_files.py
----------------------

Recursively delete every file whose *basename* ends with
    • cadence.json
    • cadence.png
(case-insensitive).

Usage
-----
    # dry-run, just list what would be removed
    python clean_cadence_files.py --dry-run

    # actually delete (default root = current dir)
    python clean_cadence_files.py

    # clean another folder
    python clean_cadence_files.py /path/to/project
"""

from pathlib import Path
import argparse
import sys

PATTERNS = ("cadence.json", "cadence.png")   # lower-case for `.endswith`

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Delete *cadence.json / *cadence.png files recursively."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="directory to start from (default: current dir)",
    )
    parser.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="list files without deleting",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    if not root.is_dir():
        parser.error(f"Root path '{root}' is not a directory.")

    action = "Would delete" if args.dry_run else "Deleting"
    total = 0

    for file_path in root.rglob("*"):
        if file_path.is_file() and file_path.name.lower().endswith(PATTERNS):
            print(f"{action}: {file_path}")
            total += 1
            if not args.dry_run:
                try:
                    file_path.unlink()
                except Exception as exc:
                    print(f"    ERROR: {exc}")

    summary = "Found" if args.dry_run else "Deleted"
    print(f"{summary} {total} file(s).")

if __name__ == "__main__":
    main()
