# %%
# This file creates and returns PyTorch DataLoaders for the train/dev/test splits.

# ## What it does
# - wraps TranslationDataset objects inside DataLoader
# - sets batch size
# - sets shuffle=True for train, False for dev/test
# - applies num_workers for faster loading
# - keeps batching logic cleanly separated from dataset logic

# ## Summary
# `dataloader.py` is responsible for turning a dataset into
# efficient batches that the model can train on.

# %%
from pathlib import Path
import sys, os, pdb
sys.path.append(str(Path(__file__).resolve().parent.parent))

# %%
import pandas as pd
import torch
from pathlib import Path
import config
from datasets import load_from_disk
from dataloaders.dataset import TranslationDataset, collate_fn, make_translation_dataset
from torch.utils.data import DataLoader, Subset

# %%
# Create DataLoaders for train, dev, and test sets

def create_dataloaders(batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, debug=False):
    """
    Create dataloaders for train, dev and test
    """

    train_dataset, dev_dataset, test_dataset = make_translation_dataset()

    # to check for sanity
    if debug:

        max_samples = config.DEBUG_MAX_SAMPLES

        # use only a small subset of training data for speed
        n = min(max_samples, len(train_dataset))
        indices = torch.randperm(len(train_dataset))[:n].tolist()
        train_dataset = Subset(train_dataset, indices)
        print(f"[DEBUG] Using only {n} training examples for fast run.")

        # use only a small subset of dev data for speed
        n = min(max_samples, len(dev_dataset))
        indices = torch.randperm(len(dev_dataset))[:n].tolist()
        dev_dataset = Subset(dev_dataset, indices)
        print(f"[DEBUG] Using only {n} dev examples for fast run.")

        # use only a small subset of test data for speed
        n = min(max_samples, len(test_dataset))
        indices = torch.randperm(len(test_dataset))[:n].tolist()
        test_dataset = Subset(test_dataset, indices)
        print(f"[DEBUG] Using only {n} test examples for fast run.")


    # build dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=config.PIN_MEMORY,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=config.PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=config.PIN_MEMORY,
    )

    return train_loader, dev_loader, test_loader


if __name__ == "__main__":
    # Quick script check for dataloader wiring
    train_loader, dev_loader, test_loader = create_dataloaders(batch_size=4, num_workers=0, debug=True)
    print("Built loaders -> train: {}, dev: {}, test: {}".format(len(train_loader), len(dev_loader), len(test_loader)))

    try:
        first_batch = next(iter(train_loader))
        print("First batch shapes:")
        print("  src_padded:", first_batch["src_padded"].shape)
        print("  tgt_padded:", first_batch["tgt_padded"].shape)
        print("  src_lens:", first_batch["src_lens"].shape)
        print("  tgt_lens:", first_batch["tgt_lens"].shape)
    except StopIteration:
        print("Train loader is empty; no batches to preview.")