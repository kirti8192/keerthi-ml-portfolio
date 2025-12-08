#!/usr/bin/env python3
"""
sync_outputs_to_local.py

Copies outputs/ and checkpoints/ from Google Drive into the local
translation_models/outputs/ and translation_models/checkpoints/ directories.

Source (Google Drive):
    ~/Library/CloudStorage/GoogleDrive*/My Drive/colab_data/translation_models/{outputs,checkpoints}

Destination (local):
    translation_models/{outputs,checkpoints}

Run:
    python translation_models/utilities/sync_outputs_to_local.py
"""

import shutil
from pathlib import Path
from tqdm import tqdm


# -------------------------------
# 1. Local and Google Drive paths
# -------------------------------

LOCAL_BASE = Path("translation_models")
LOCAL_OUTPUTS = LOCAL_BASE / "outputs"
LOCAL_CHECKPOINTS = LOCAL_BASE / "checkpoints"

DRIVE_BASE = Path.home() / "Library/CloudStorage"
drive_candidates = list(DRIVE_BASE.glob("GoogleDrive*"))

if not drive_candidates:
    raise RuntimeError(
        "Could not find Google Drive folder under ~/Library/CloudStorage.\n"
        "Make sure Google Drive for Desktop is installed."
    )

GOOGLE_DRIVE = drive_candidates[0]
DRIVE_ROOT = GOOGLE_DRIVE / "My Drive" / "colab_data" / "translation_models"
DRIVE_OUTPUTS = DRIVE_ROOT / "outputs"
DRIVE_CHECKPOINTS = DRIVE_ROOT / "checkpoints"


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
        print(f"Source directory does not exist, skipping: {src}")
        return

    print(f"Copying:\n  {src}\n→ {dst}")
    dst.mkdir(parents=True, exist_ok=True)

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
    print("=== Syncing outputs/checkpoints from Google Drive to local ===")
    print(f"Drive source (outputs):      {DRIVE_OUTPUTS}")
    print(f"Drive source (checkpoints):  {DRIVE_CHECKPOINTS}")
    print(f"Local dest (outputs):        {LOCAL_OUTPUTS}")
    print(f"Local dest (checkpoints):    {LOCAL_CHECKPOINTS}\n")

    copy_tree_with_progress(DRIVE_OUTPUTS, LOCAL_OUTPUTS)
    copy_tree_with_progress(DRIVE_CHECKPOINTS, LOCAL_CHECKPOINTS)

    print("Sync complete.\n")


if __name__ == "__main__":
    main()
