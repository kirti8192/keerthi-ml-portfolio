#!/usr/bin/env python3
"""
sync_data_to_drive.py

Copies local translation_models/data/ into Google Drive folder:
    ~/Library/CloudStorage/GoogleDrive*/MyDrive/colab_data/translation_models/data/

Adds a progress bar for file transfers.

Run:
    python sync_data_to_drive.py
"""

import shutil
from pathlib import Path
from tqdm import tqdm


# -------------------------------
# 1. Local and Google Drive paths
# -------------------------------

LOCAL_DATA = Path("translation_models/data")

DRIVE_BASE = Path.home() / "Library/CloudStorage"
drive_candidates = list(DRIVE_BASE.glob("GoogleDrive*"))

if not drive_candidates:
    raise RuntimeError("Could not find Google Drive folder under ~/Library/CloudStorage.\n"
                       "Make sure Google Drive for Desktop is installed.")

GOOGLE_DRIVE = drive_candidates[0]

# NEW (correct)
DEST = GOOGLE_DRIVE / "My Drive" / "colab_data" / "translation_models" / "data"


# -------------------------------
# 2. Utility: File generator
# -------------------------------

def iter_files(root: Path):
    """Yield all files inside a directory tree."""
    for path in root.rglob("*"):
        if path.is_file():
            yield path


# -------------------------------
# 3. Copy with progress bar
# -------------------------------

def copy_tree_with_progress(src: Path, dst: Path):
    """Copy entire folder tree with a progress bar."""
    if not src.exists():
        raise RuntimeError(f"Source directory does not exist: {src}")

    print(f"Copying:\n  {src}\n→ {dst}")

    # Only create the final 'data' directory.
    # We assume MyDrive/colab_data/translation_models already exist.
    dst.mkdir(exist_ok=True)

    files = list(iter_files(src))
    print(f"Total files to copy: {len(files)}")

    for file in tqdm(files, desc="Copying files", unit="file"):
        rel = file.relative_to(src)
        out_path = dst / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, out_path)

    print("Done.\n")


# -------------------------------
# 4. Main
# -------------------------------

def main():
    print("=== Syncing translation_models/data → Google Drive ===")
    print(f"Local source:      {LOCAL_DATA}")
    print(f"Google Drive dest: {DEST}\n")

    copy_tree_with_progress(LOCAL_DATA, DEST)

    print("Sync complete. Data available to use in Colab.\n")


if __name__ == "__main__":
    main()