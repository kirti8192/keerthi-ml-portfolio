# %%
"""
construct_vocab.py

Objective is to construct a vocabulary file from processed dataset.
Input: processed dataset on disk
Output: vocabulary file on disk

"""

# %%
from pathlib import Path
import sys, os, pdb
sys.path.append(str(Path(__file__).resolve().parent.parent))

# %%
import config
from datasets import load_from_disk
from collections import Counter
import json

# %%
# load dataset
def load_dataset():
    """
    Load the raw dataset from disk
    """
    return load_from_disk(str(config.DATA_PROCESSED))

# %%
def build_vocab(tokens, vocab, max_size):
    """
    Build vocabulary from tokens
    - Filter by min frequency
    - Token to index mapping
    Input:
        tokens: List[List[str]] - list of tokenized sentences
        vocab: Dict[str, int] - existing vocabulary to update
        max_size: int - maximum size of vocabulary (including specials)
    """

    # Use Counter to count token frequencies in flattened list
    counter = Counter()
    flattened_tokens = [token for token_list in tokens for token in token_list]
    counter.update(flattened_tokens)

    # take top-K by frequency (excluding existing specials)
    current_idx = len(vocab)
    remaining = max_size - current_idx
    if remaining <= 0:
        return vocab

    for token, freq in counter.most_common():
        if freq < config.MIN_FREQ:
            break
        if token in vocab:
            continue
        vocab[token] = current_idx
        current_idx += 1
        remaining -= 1
        if remaining <= 0:
            break

    return vocab

# %%
# save vocab to disk
def _save_vocab(vocab, lang):
    """
    Save vocabulary to disk
    """

    # create vocab directory if not exists
    vocab_dir = config.VOCAB_DIR
    vocab_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = vocab_dir / f"vocab_{lang}.json"

    # save vocab to json file
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)

    print(f"Saved {lang} vocabulary of size {len(vocab)} to {vocab_path}")

# %%
def construct_vocab():
    """
    Construct vocabulary from processed dataset
    """
        
    print(f"Loading processed dataset from {config.DATA_PROCESSED}...")
    # load processed dataset
    dataset = load_dataset()
    print(f"Loaded splits: {list(dataset.keys())}")

    # initialize vocab with special tokens
    vocab_src = {
        config.PAD_TOKEN: config.PAD_ID,
        config.UNK_TOKEN: config.UNK_ID,
        config.SOS_TOKEN: config.SOS_ID,
        config.EOS_TOKEN: config.EOS_ID,
    }

    vocab_tgt = {
        config.PAD_TOKEN: config.PAD_ID,
        config.UNK_TOKEN: config.UNK_ID,
        config.SOS_TOKEN: config.SOS_ID,
        config.EOS_TOKEN: config.EOS_ID,
    }

    print("Building source vocabulary from train split...")
    # build vocab from training data only
    vocab_src = build_vocab(
        dataset['train']['src_tokens'],
        vocab_src,
        max_size=config.MAX_VOCAB_SIZE_SRC,
    )
    print(f"Source vocab size (including specials): {len(vocab_src)}")

    print("Building target vocabulary from train split...")
    vocab_tgt = build_vocab(
        dataset['train']['tgt_tokens'],
        vocab_tgt,
        max_size=config.MAX_VOCAB_SIZE_TGT,
    )
    print(f"Target vocab size (including specials): {len(vocab_tgt)}")

    # save vocab to disk
    _save_vocab(vocab_src, "src")
    _save_vocab(vocab_tgt, "tgt")
    print("Vocabulary construction complete.")

# %%
if __name__ == "__main__":
    construct_vocab()
