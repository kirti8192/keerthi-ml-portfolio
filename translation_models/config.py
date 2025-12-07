"""Centralized filesystem paths for the translation project."""

from pathlib import Path
import os
import torch

# seed
SEED = 42

# ROOT
PROJECT_ROOT = Path(__file__).resolve().parent

# Base directories (override with env vars when running on Colab/remote)
# e.g., export NMT_DATA_ROOT="/content/drive/MyDrive/nmt_en_te"
#       export NMT_OUTPUTS_ROOT="/content/drive/MyDrive/nmt_en_te_outputs"
#       export NMT_CHECKPOINTS_ROOT="/content/drive/MyDrive/nmt_en_te_checkpoints"
DATA_ROOT = Path(os.environ.get("NMT_DATA_ROOT", PROJECT_ROOT / "data"))
OUTPUTS_ROOT = Path(os.environ.get("NMT_OUTPUTS_ROOT", PROJECT_ROOT / "outputs"))
CHECKPOINTS_ROOT = Path(os.environ.get("NMT_CHECKPOINTS_ROOT", PROJECT_ROOT / "checkpoints"))

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
DATA_RAW = DATA_ROOT / "raw" / LANG_PAIR
DATA_PROCESSED = DATA_ROOT / "processed" / LANG_PAIR
VOCAB_DIR = DATA_ROOT / "vocab" / LANG_PAIR
DATA_NUM = DATA_ROOT / "numericalized" / LANG_PAIR
CHECKPOINTS_DIR = CHECKPOINTS_ROOT / LANG_PAIR
METRICS_DIR = OUTPUTS_ROOT / "metrics" / LANG_PAIR
PLOTS_DIR = OUTPUTS_ROOT / "plots" / LANG_PAIR

# Pre-processing parameters
MIN_SEQ_LEN = 2
MAX_SEQ_LEN = 50

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

# pin_memory True only if CUDA
PIN_MEMORY = True if torch.cuda.is_available() else False

# debug params
DEBUG_MAX_SAMPLES = 64
DEBUG_MODE = True

# model hyperparams
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 1
DROPOUT = 0.1

# training hyperparams
NUM_EPOCHS = 2
LEARNING_RATE = 1e-3
TEACHER_FORCING = 1.0  # start with pure teacher forcing
