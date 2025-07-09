from pathlib import Path
import shutil, os, stat

ROOT      = Path("Unsegmented")
SUFFIXES  = ("removed", "smoothed")
PNG_GLOB  = "*.png"

# Hjälpfunktion: gör skrivbar och försök igen
def _force_writeable(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

# --- 1. ta bort alla diag-mappar ----------------------------------------
for d in ROOT.rglob("diag"):
    if d.is_dir():
        print("Deleting dir:", d)
        shutil.rmtree(d, onerror=_force_writeable)

# --- 2. ta bort filer som slutar på removed / smoothed ------------------
for f in ROOT.rglob("*"):
    if f.is_file() and f.stem.endswith(SUFFIXES):
        try:
            print("Deleting file:", f)
            f.unlink()
        except PermissionError:
            os.chmod(f, stat.S_IWRITE)
            f.unlink()

# --- 3. ta bort ALLA PNG-filer ------------------------------------------
for png in ROOT.rglob(PNG_GLOB):
    try:
        print("Deleting PNG:", png)
        png.unlink()
    except PermissionError:
        os.chmod(png, stat.S_IWRITE)
        png.unlink()
