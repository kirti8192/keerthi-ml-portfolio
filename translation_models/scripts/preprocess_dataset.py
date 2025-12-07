# %%
# Objectives

# 1. Load the raw dataset from disk
# 2. Normalize text for both languages
# 3. Tokenize sentences (whitespace tokenization)
# 4. Remove unusable examples
# 5. Save the tokenized dataset to disk

# %%
from pathlib import Path
import sys, os, pdb
sys.path.append(str(Path(__file__).resolve().parent.parent))

# %%
import config
from datasets import load_from_disk

# %%
# load dataset
def load_dataset():
    """
    Load the raw dataset from disk
    """
    return load_from_disk(str(config.DATA_RAW))

# %%
# preprocess + tokenize
def _normalize_tokenize(example):
    """
    Normalize text by lowercasing and stripping whitespace.
    Then, tokenize by splitting on whitespace.
    Language agnostic.

    Input: {"src": str, "tgt": str}
    Output: {"src_tokens": List[str], "tgt_tokens": List[str]}

    """
    return {
        "src_tokens": example["src"].lower().strip().split(),
        "tgt_tokens": example["tgt"].lower().strip().split(),
    }

# %% 
# drop unusable examples
def _filter_unusable(example):
    """
    Filter out 
    - source or target shorter than min length
    - source or target longer than max length

    Input: {"src_tokens": List[str], "tgt_tokens": List[str]}
    Output: bool
    """
    min_len = config.MIN_SEQ_LEN
    max_len = config.MAX_SEQ_LEN
    src_len = len(example["src_tokens"])
    tgt_len = len(example["tgt_tokens"])
    if src_len < min_len or tgt_len < min_len:
        return False
    if src_len > max_len or tgt_len > max_len:
        return False
    return True

# %%
# save processed dataset to disk
def _save_dataset(dataset_processed):
    """
    Save processed dataset to disk
    """
    dataset_processed_path = config.DATA_PROCESSED
    dataset_processed_path.mkdir(parents=True, exist_ok=True)
    for split in dataset_processed.keys():
        dataset_processed[split].save_to_disk(str(dataset_processed_path / split))

# %%
def preprocess_dataset():
    """
    Preprocess wrapper script
    """
    # load dataset
    dataset = load_dataset()

    # vectorize normalization and tokenization
    dataset_processed = {}
    for split in dataset.keys():
        print(f"Pre-processing {split} split...")

        dataset_processed[split] = dataset[split].map(_normalize_tokenize, remove_columns=["src", "tgt"])
        print(f"Total length before filtering: {len(dataset_processed[split])}")

        dataset_processed[split] = dataset_processed[split].filter(_filter_unusable)
        print(f"Total length after filtering: {len(dataset_processed[split])}")        

    # save dataset to disk
    _save_dataset(dataset_processed)

# %%
if __name__ == "__main__":
    preprocess_dataset()