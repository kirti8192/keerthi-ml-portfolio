"""Centralized filesystem paths for the translation project."""

from pathlib import Path

# seed
SEED = 42

# ROOT
PROJECT_ROOT = Path(__file__).resolve().parent

# Language configuration
# Assamese (as),
# Bengali (bn),
# Gujarati (gu),
# Hindi (hi),
# Kannada (kn),
# Malayalam (ml),
# Marathi (mr),
# Odia (or),
# Punjabi (pa),
# Tamil (ta) and
# Telugu (te).
LANG_SRC = "en"
LANG_TGT = "bn"
LANG_PAIR = f"{LANG_SRC}-{LANG_TGT}"

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw" / LANG_PAIR
DATA_PROCESSED = DATA_DIR / "processed" / LANG_PAIR
VOCAB_DIR = DATA_DIR / "vocab" / LANG_PAIR
DATA_NUM = DATA_DIR / "numericalized" / LANG_PAIR

# Pre-processing parameters
MIN_SEQ_LEN = 2
MAX_SEQ_LEN = 100

# split params
TRAIN_SPLIT = 0.9
VAL_SPLIT = 0.05

# vocab tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3

# vocab params
MIN_FREQ = 10

# dataloader params
BATCH_SIZE = 64
NUM_WORKERS = 2
PIN_MEMORY = True