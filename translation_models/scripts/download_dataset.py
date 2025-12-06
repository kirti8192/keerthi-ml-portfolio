# %%
from datasets import load_dataset
import sys, os
# %%
def download_dataset():

    # load huggingface dataset
    dataset = load_dataset("ai4bharat/samanantar", "ta-en")

    # create data directory if not exists
    