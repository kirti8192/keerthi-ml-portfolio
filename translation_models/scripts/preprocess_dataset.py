# %%
# Objectives

# 1. Load the raw dataset from disk
# 2. Normalize text for both languages
# 3. Tokenize sentences (whitespace tokenization)
# 4. Remove unusable examples
# 5. Split into train/val/test splits
# 6. Save the tokenized dataset to disk

# %%
from pathlib import Path
import sys, os, pdb
sys.path.append(str(Path(__file__).resolve().parent.parent))

# %%
import config
from datasets import load_from_disk, DatasetDict

# %%
# load dataset
def _load_dataset():
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
# split into train/val/test
def _split_dataset(dataset):
    """
    Split dataset into train/val/test splits
    - Check if splits already exist in dataset
    """
    if all(split in dataset.keys() for split in ["train", "validation", "test"]):
        return dataset

    # otherwise, perform a random split on the DatasetDict
    dataset = dataset["train"]  # assume all data is in 'train' split
    dataset = dataset.shuffle(seed=config.SEED)
    total_size = len(dataset)
    train_size = int(config.TRAIN_SPLIT * total_size)
    val_size = int(config.VAL_SPLIT * total_size)

    # create splits and create DatasetDict
    dataset_split = DatasetDict({
        "train": dataset.select(range(0, train_size)),
        "validation": dataset.select(range(train_size, train_size + val_size)),
        "test": dataset.select(range(train_size + val_size, total_size)),       # put remaining examples in test
    })

    return dataset_split
    

# %%
# save processed dataset to disk
def _save_dataset(dataset_processed):
    """
    Save processed dataset to disk
    """
    dataset_processed_path = config.DATA_PROCESSED
    dataset_processed_path.mkdir(parents=True, exist_ok=True)

    # make a DatasetDict and save to disk
    dataset_dict = DatasetDict(dataset_processed)
    dataset_dict.save_to_disk(str(dataset_processed_path))

# %%
def preprocess_dataset():
    """
    Preprocess wrapper script
    """
    print(f"Loading raw dataset from {config.DATA_RAW}...")
    # load dataset
    dataset = _load_dataset()
    print(f"Loaded splits: {list(dataset.keys())}")

    # vectorize normalization and tokenization
    dataset_processed = {}
    for split in dataset.keys():
        print(f"Pre-processing {split} split...")

        dataset_processed[split] = dataset[split].map(_normalize_tokenize, remove_columns=["src", "tgt"])
        pre_filter_len = len(dataset_processed[split])

        dataset_processed[split] = dataset_processed[split].filter(_filter_unusable)
        post_filter_len = len(dataset_processed[split])
        
        # print stats of filtering
        print(f"  Examples before filtering: {pre_filter_len}")
        print(f"  Examples after filtering: {post_filter_len}")
        print(f"  Percentage removed: {100 * (pre_filter_len - post_filter_len) / pre_filter_len:.2f}%")

    # split dataset if needed
    dataset_processed = _split_dataset(dataset_processed)

    # save dataset to disk
    _save_dataset(dataset_processed)
    print(f"Processed dataset saved to {config.DATA_PROCESSED}")

# %%
if __name__ == "__main__":
    preprocess_dataset()
