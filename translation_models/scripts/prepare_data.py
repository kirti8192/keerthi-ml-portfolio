# %%
"""
prepare_data.py

Wrapper script that prepares a full NMT dataset for a given language pair.
Runs the entire pipeline:
    1. Download dataset
    2. Preprocess + tokenize
    3. Build vocabularies
    4. Numericalize dataset
    5. Save processed data to disk

This script should be run once per language pair after configuring LANG_TGT in config.py.
"""

# %%
from pathlib import Path
import sys, os
sys.path.append(str(Path(__file__).resolve().parent.parent))

# %%
import config
from download_dataset import download_dataset
from preprocess_dataset import preprocess_dataset
from construct_vocab import construct_vocab

# %%
print(f"Preparing data for language pair: {config.LANG_PAIR}")

# %% 
# download dataset if not already present
if not config.DATA_RAW.exists():
    print("\n--- Download stage ---")
    print(f"Downloading dataset for {config.LANG_PAIR} to {config.DATA_RAW}...")
    download_dataset()
else:
    print(f"Dataset already present at {config.DATA_RAW}, skipping download.")

# %%
# preprocess dataset 
if not config.DATA_PROCESSED.exists():
    print("\n--- Preprocess stage ---")
    print(f"Preprocessing dataset into {config.DATA_PROCESSED}...")
    preprocess_dataset()
else:
    print(f"Processed data already exists at {config.DATA_PROCESSED}, skipping preprocessing.")

# %%
# vocab construction
if not config.VOCAB_DIR.exists():
    print("\n--- Vocabulary stage ---")
    print(f"Constructing vocabulary at {config.VOCAB_DIR}...")
    construct_vocab()
else:
    print(f"Vocabulary already exists at {config.VOCAB_DIR}, skipping vocab construction.")

print("\nData preparation complete.")