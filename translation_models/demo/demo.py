"""
Run all saved checkpoints and print translations for a given English sentence.

Usage:
    python3 -m demo.demo "your sentence here"
or simply:
    python3 translation_models/demo/demo.py
    (and then type the sentence when prompted)
"""

from pathlib import Path
import sys
import json
from typing import Dict, List, Tuple

import torch

# allow imports from project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config 
from models import seq2seq, seq2seq_attn  


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_vocabs() -> Tuple[Dict[str, int], Dict[str, int]]:
    vocab_src_path = config.VOCAB_DIR / "vocab_src.json"
    vocab_tgt_path = config.VOCAB_DIR / "vocab_tgt.json"

    with open(vocab_src_path, "r", encoding="utf-8") as f:
        vocab_src = json.load(f)
    with open(vocab_tgt_path, "r", encoding="utf-8") as f:
        vocab_tgt = json.load(f)
    return vocab_src, vocab_tgt


def sentence_to_ids(sentence: str, vocab_src: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = sentence.lower().strip().split()
    ids = [config.SOS_ID] + [vocab_src.get(tok, config.UNK_ID) for tok in tokens] + [config.EOS_ID]
    src_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # [1, T_src]
    src_len = torch.tensor([len(ids)], dtype=torch.long)  # [1]
    return src_tensor, src_len


def ids_to_sentence(token_ids: List[int], id_to_token: Dict[int, str]) -> str:
    tokens = [id_to_token.get(idx, config.UNK_TOKEN) for idx in token_ids if idx not in (config.SOS_ID, config.EOS_ID, config.PAD_ID)]
    return " ".join(tokens)


def greedy_decode_seq2seq(model, src_padded, src_lens, max_len: int) -> List[int]:
    device = next(model.parameters()).device
    src_padded = src_padded.to(device)
    src_lens = src_lens.to(device)

    with torch.no_grad():
        _, hidden = model.encoder(src_padded, src_lens)
        input_token = torch.full((1,), config.SOS_ID, dtype=torch.long, device=device)

        decoded = []
        for _ in range(max_len):
            logits, hidden = model.decoder(tgt_step=input_token, hidden=hidden)
            next_token = logits.argmax(1)
            token_id = next_token.item()
            if token_id == config.EOS_ID:
                break
            decoded.append(token_id)
            input_token = next_token

    return decoded


def greedy_decode_seq2seq_attn(model, src_padded, src_lens, max_len: int) -> List[int]:
    device = next(model.parameters()).device
    src_padded = src_padded.to(device)
    src_lens = src_lens.to(device)

    with torch.no_grad():
        enc_out, hidden = model.encoder(src_padded, src_lens)
        input_token = torch.full((1,), config.SOS_ID, dtype=torch.long, device=device)

        decoded = []
        for _ in range(max_len):
            logits, hidden, _ = model.decoder(enc_out=enc_out, tgt_step=input_token, hidden=hidden)
            next_token = logits.argmax(1)
            token_id = next_token.item()
            if token_id == config.EOS_ID:
                break
            decoded.append(token_id)
            input_token = next_token

    return decoded


def build_model(model_name: str, device: torch.device):
    if model_name == "seq2seq":
        encoder = seq2seq.Encoder(
            vocab_size_src=len(vocab_src_global),
            embed_dim=config.EMBED_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            pad_id=config.PAD_ID,
        )
        decoder = seq2seq.Decoder(
            vocab_size_tgt=len(vocab_tgt_global),
            embed_dim=config.EMBED_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            pad_id=config.PAD_ID,
        )
        model = seq2seq.Seq2Seq(encoder=encoder, decoder=decoder, sos_id=config.SOS_ID)
    elif model_name == "seq2seq_attn":
        encoder = seq2seq_attn.Encoder(
            vocab_size_src=len(vocab_src_global),
            embed_dim=config.EMBED_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            pad_id=config.PAD_ID,
        )
        decoder = seq2seq_attn.Decoder(
            vocab_size_tgt=len(vocab_tgt_global),
            embed_dim=config.EMBED_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            pad_id=config.PAD_ID,
        )
        model = seq2seq_attn.Seq2SeqAttn(encoder=encoder, decoder=decoder, sos_id=config.SOS_ID)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model.to(device)


def load_model_from_checkpoint(model_name: str, ckpt_path: Path, device: torch.device):
    model = build_model(model_name, device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def find_checkpoints() -> List[Tuple[str, Path]]:
    base = config.CHECKPOINTS_DIR
    checkpoints = []
    for path in base.rglob("*_checkpoint.pth"):
        stem = path.stem
        model_name = stem.replace("_checkpoint", "")
        checkpoints.append((model_name, path))
    return sorted(checkpoints, key=lambda x: x[0])


def main():
    device = get_device()
    print(f"Using device: {device}")

    checkpoints = find_checkpoints()
    if not checkpoints:
        print(f"No checkpoints found under {config.CHECKPOINTS_DIR}")
        return

    sentence = " ".join(sys.argv[1:]).strip()
    if not sentence:
        sentence = input("Enter an English sentence to translate: ").strip()
    if not sentence:
        print("No sentence provided; exiting.")
        return

    src_padded, src_lens = sentence_to_ids(sentence, vocab_src_global)
    id_to_token = {idx: tok for tok, idx in vocab_tgt_global.items()}

    print(f"\nTranslating: \"{sentence}\"")
    max_decode_len = config.MAX_SEQ_LEN

    for model_name, ckpt_path in checkpoints:
        try:
            model = load_model_from_checkpoint(model_name, ckpt_path, device)
        except Exception as exc:
            print(f"[{model_name}] Failed to load from {ckpt_path}: {exc}")
            continue

        if model_name == "seq2seq":
            token_ids = greedy_decode_seq2seq(model, src_padded, src_lens, max_decode_len)
        else:
            token_ids = greedy_decode_seq2seq_attn(model, src_padded, src_lens, max_decode_len)

        translation = ids_to_sentence(token_ids, id_to_token)
        print(f"{model_name:15s} -> {translation}")


if __name__ == "__main__":
    vocab_src_global, vocab_tgt_global = load_vocabs()
    main()
