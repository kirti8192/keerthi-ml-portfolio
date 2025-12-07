"""Centralized filesystem paths for the translation project."""

from pathlib import Path

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
LANG_TGT = "ta"
LANG_PAIR = f"{LANG_SRC}-{LANG_TGT}"

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw" / LANG_PAIR
DATA_PROCESSED = DATA_DIR / "processed" / LANG_PAIR
VOCAB_DIR = DATA_DIR / "vocab" / LANG_PAIR

# Pre-processing parameters
MIN_SEQ_LEN = 2
MAX_SEQ_LEN = 100