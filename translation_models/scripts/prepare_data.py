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

# %% 
# download dataset if not already present
if not config.DATA_RAW.exists():
    download_dataset()

# %%
# preprocess dataset 
if not config.DATA_PROCESSED.exists():
    preprocess_dataset()

