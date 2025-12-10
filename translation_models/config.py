"""Centralized filesystem paths for the translation project."""

from pathlib import Path
import os
import torch

# seed
SEED = 42

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

# model choice
MODEL_NAME = "seq2seq_attn"  # choose: "seq2seq", "seq2seq_attn"

# ROOT
PROJECT_ROOT = Path(__file__).resolve().parent

# Base directories (override with env vars when running on Colab/remote)
# e.g., export NMT_DATA_ROOT="/content/drive/MyDrive/nmt_en_te"
#       export NMT_OUTPUTS_ROOT="/content/drive/MyDrive/nmt_en_te_outputs"
#       export NMT_CHECKPOINTS_ROOT="/content/drive/MyDrive/nmt_en_te_checkpoints"
DATA_ROOT = Path(os.environ.get("NMT_DATA_ROOT", PROJECT_ROOT / "data"))
OUTPUTS_ROOT = Path(os.environ.get("NMT_OUTPUTS_ROOT", PROJECT_ROOT / "outputs"))
CHECKPOINTS_ROOT = Path(os.environ.get("NMT_CHECKPOINTS_ROOT", PROJECT_ROOT / "checkpoints"))

# Data directories
DATA_RAW = DATA_ROOT / "raw" / LANG_PAIR
DATA_PROCESSED = DATA_ROOT / "processed" / LANG_PAIR
VOCAB_DIR = DATA_ROOT / "vocab" / LANG_PAIR
DATA_NUM = DATA_ROOT / "numericalized" / LANG_PAIR
CHECKPOINTS_DIR = CHECKPOINTS_ROOT / LANG_PAIR
METRICS_DIR = OUTPUTS_ROOT / "metrics" / LANG_PAIR
PLOTS_DIR = OUTPUTS_ROOT / "plots" / LANG_PAIR

CHECKPOINT_FILENAME = f"{MODEL_NAME}_checkpoint.pth"
METRICS_FILENAME = f"{MODEL_NAME}_loss_history.csv"
PLOT_FILENAME = f"{MODEL_NAME}_loss_curve.png"

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
MAX_VOCAB_SIZE_SRC = 30000
MAX_VOCAB_SIZE_TGT = 30000

# dataloader params
BATCH_SIZE = 16
NUM_WORKERS = 2

# pin_memory True only if CUDA
PIN_MEMORY = True if torch.cuda.is_available() else False

# debug params
DEBUG_MAX_SAMPLES = 100000
DEBUG_MODE = True

# model hyperparams
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 1
DROPOUT = 0.1

# training hyperparams
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
TEACHER_FORCING_START = 1.0  # start with pure teacher forcing
TEACHER_FORCING_END = 0.1    # decay toward this ratio by final epoch
EARLY_STOP_PATIENCE = 3      # stop if no val improvement for these epochs
EARLY_STOP_MIN_DELTA = 0.0   # minimum improvement to reset patience
