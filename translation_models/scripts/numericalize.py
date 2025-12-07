# # numericalize.py — concise objectives

# 1. Load tokenized + split dataset from DATA_PROCESSED.
# 2. Load vocab_src and vocab_tgt (token → id).
# 3. For each split:
#      - Convert src_tokens → src_ids (use <unk> for OOV).
#      - Convert tgt_tokens → tgt_ids (use <unk> for OOV).
#      - Add BOS/EOS to both sequences.
#      - Compute src_len and tgt_len.
# 4. Add new fields: src_ids, tgt_ids, src_len, tgt_len.
# 5. Save numericalized DatasetDict to DATA_NUM (train/val/test).

# %%
from pathlib import Path
import sys, os, pdb
sys.path.append(str(Path(__file__).resolve().parent.parent))

# %%
import config
from datasets import load_from_disk
import json

# %% 
# helper functions

def _numericalize_example(example, vocab_src, vocab_tgt,
                          unk_id_src, sos_id_src, eos_id_src,
                          unk_id_tgt, sos_id_tgt, eos_id_tgt):
    src_ids = [vocab_src.get(tok, unk_id_src) for tok in example["src_tokens"]]
    src_ids = [sos_id_src] + src_ids + [eos_id_src]

    tgt_ids = [vocab_tgt.get(tok, unk_id_tgt) for tok in example["tgt_tokens"]]
    tgt_ids = [sos_id_tgt] + tgt_ids + [eos_id_tgt]

    return {
        "src_ids": src_ids,
        "tgt_ids": tgt_ids,
        "src_len": len(src_ids),
        "tgt_len": len(tgt_ids),
    }

# %%
def numericalize():
    """
    Numericalize the tokenized dataset using the constructed vocabularies
    - Convert tokens to IDs
    - Add BOS/EOS tokens
    - Compute sequence lengths
    - Repeat for train/val/test splits

    Input:
        Processed dataset on disk at config.DATA_PROCESSED
        Vocabularies at config.VOCAB_SRC and config.VOCAB_TGT

    Output:
        Numericalized dataset saved to config.DATA_NUM
    """

    # load processed dataset
    dataset = load_from_disk(str(config.DATA_PROCESSED))
    print(f"Loaded tokenized dataset from {config.DATA_PROCESSED}")
    for split in dataset.keys():
        print(f"  Split '{split}': {dataset[split].num_rows} examples")

    # load vocabularies
    vocab_src_path = config.VOCAB_DIR / "vocab_src.json"
    vocab_tgt_path = config.VOCAB_DIR / "vocab_tgt.json"

    with open(vocab_src_path, 'r', encoding='utf-8') as f:
        vocab_src = json.load(f)
    with open(vocab_tgt_path, 'r', encoding='utf-8') as f:
        vocab_tgt = json.load(f)
    print(f"Loaded vocabularies: src={len(vocab_src)} tokens, tgt={len(vocab_tgt)} tokens")

    # get UNK/BOS/EOS IDs
    unk_id_src = vocab_src[config.UNK_TOKEN]
    sos_id_src = vocab_src[config.SOS_TOKEN]
    eos_id_src = vocab_src[config.EOS_TOKEN]
    unk_id_tgt = vocab_tgt[config.UNK_TOKEN]
    sos_id_tgt = vocab_tgt[config.SOS_TOKEN]
    eos_id_tgt = vocab_tgt[config.EOS_TOKEN]
    print(f"Source vocab special tokens: UNK={unk_id_src}, SOS={sos_id_src}, EOS={eos_id_src}")

    # numericalize each split
    for split in dataset.keys():
        print(f"Numericalizing split '{split}' ({dataset[split].num_rows} examples)...")
        dataset[split] = dataset[split].map(
            _numericalize_example,
            fn_kwargs={
                "vocab_src": vocab_src,
                "vocab_tgt": vocab_tgt,
                "unk_id_src": unk_id_src,
                "sos_id_src": sos_id_src,
                "eos_id_src": eos_id_src,
                "unk_id_tgt": unk_id_tgt,
                "sos_id_tgt": sos_id_tgt,
                "eos_id_tgt": eos_id_tgt,
            },
            remove_columns=["src_tokens", "tgt_tokens"],
        )
        print(f"Finished split '{split}'. Columns now: {dataset[split].column_names}")
        
    # save numericalized dataset to disk
    dataset.save_to_disk(str(config.DATA_NUM))
    print(f"Saved numericalized dataset to {config.DATA_NUM}")

# %%
if __name__ == "__main__":
    numericalize()
