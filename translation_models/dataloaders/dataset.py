# %%
# TranslationDataset + collate_fn — objectives
#
# 1. Wrap a single HuggingFace split (train/val/test) into a PyTorch Dataset.
#    - __getitem__ returns src_ids, tgt_ids, src_len, tgt_len.
#
# 2. Implement a collate_fn that:
#    - receives a list of examples,
#    - pads src_ids and tgt_ids to the max length in the batch (using PAD_ID),
#    - stacks lengths into tensors,
#    - returns:
#         src_padded  : LongTensor [B, T_src_max]
#         tgt_padded  : LongTensor [B, T_tgt_max]
#         src_lens    : LongTensor [B]
#         tgt_lens    : LongTensor [B]
#
# 3. Ensure the batch is ready for encoder–decoder models:
#    - BOS/EOS already exist (added during numericalization),
#    - padding handled cleanly,
#    - no tokens or metadata missing.

# %%
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

# %%
import torch
from pathlib import Path
import config
from datasets import load_from_disk

# %%
class TranslationDataset(torch.utils.data.Dataset):

    def __init__(self, hf_dataset_split):
        """
        Args:
            hf_dataset_split: HuggingFace dataset split (train/val/test)
        """
        self.dataset = hf_dataset_split

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Pick idx item from dataset and return a dict with:
        - src_ids: LongTensor 
        - tgt_ids: LongTensor 
        - src_len: LongTensor 
        - tgt_len: LongTensor 
        """

        item = self.dataset[idx]
        src_ids = torch.tensor(item['src_ids'], dtype=torch.long)
        tgt_ids = torch.tensor(item['tgt_ids'], dtype=torch.long)
        src_len = torch.tensor(len(src_ids), dtype=torch.long)
        tgt_len = torch.tensor(len(tgt_ids), dtype=torch.long)

        return {
            'src_ids': src_ids,
            'tgt_ids': tgt_ids,
            'src_len': src_len,
            'tgt_len': tgt_len
        }

# %%
def collate_fn(batch):
    """
    Collate function to pad sequences and stack lengths.
    
    Args:
        batch: List of dicts with keys src_ids, tgt_ids, src_len, tgt_len
    """

    # Pad src_ids and tgt_ids
    src_ids = [item['src_ids'] for item in batch]   # pick src_ids from each item
    src_padded = torch.nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=config.PAD_ID)

    tgt_ids = [item['tgt_ids'] for item in batch]   # pick tgt_ids from each item
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_ids, batch_first=True, padding_value=config.PAD_ID)

    # Stack lengths
    src_lens = torch.stack([item['src_len'] for item in batch])
    tgt_lens = torch.stack([item['tgt_len'] for item in batch])

    return {
        'src_padded': src_padded,
        'tgt_padded': tgt_padded,
        'src_lens': src_lens,
        'tgt_lens': tgt_lens
    }

# %%
# make train/dev/test datasets
def make_translation_dataset():
    """
    Create TranslationDataset instances for train/dev/test splits.
    """
    
    # load numericalized dataset from disk
    data_num_path = config.DATA_NUM
    hf_dataset = load_from_disk(data_num_path)

    # create TranslationDataset instances
    train_dataset = TranslationDataset(hf_dataset['train'])
    dev_dataset = TranslationDataset(hf_dataset['validation'])
    test_dataset = TranslationDataset(hf_dataset['test'])

    return train_dataset, dev_dataset, test_dataset
