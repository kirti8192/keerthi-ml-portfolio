# Neural Machine Translation (source → target)

Generic NMT scaffold intended for source→target language pairs, with English→Tamil as the showcase instance. Includes slots for RNN+attention, CNN+RNN encoders, and a tiny transformer, all wired for reproducible PyTorch workflows.

Data source: [ai4bharat/samanantar](https://huggingface.co/datasets/ai4bharat/samanantar)

## First-time setup for a new target language
- Set `LANG_TGT` (and optionally `LANG_SRC`) in `config.py`.
- Run `python scripts/prepare_data.py` once to download raw data, preprocess it, and build vocabularies for the new language pair.
