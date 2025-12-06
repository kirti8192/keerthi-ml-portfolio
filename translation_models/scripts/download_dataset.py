# %%
from pathlib import Path
import sys, os
from datasets import load_dataset

# %%
# Make project root importable when running as a script.
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW, LANG_TGT

# %%
def download_dataset():
    """Download the dataset to the configured raw data directory."""

    # download the dataset
    dataset = load_dataset("ai4bharat/samanantar", LANG_TGT)

    # save the dataset to disk
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(DATA_RAW))

# %%
if __name__ == "__main__":
    download_dataset()
    