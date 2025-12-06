"""Centralized filesystem paths for the translation project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LANG_SRC = "en"
LANG_TGT = "bn"
LANG_PAIR = f"{LANG_SRC}-{LANG_TGT}"

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw" / LANG_PAIR
DATA_PROCESSED = DATA_DIR / "processed" / LANG_PAIR
VOCAB_DIR = DATA_DIR / "vocab" / LANG_PAIR
