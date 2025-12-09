""" 
Train a sequence-to-sequence model for machine translation using PyTorch.

This script
- Loads preprocessed data and vocabularies.
- Instantiates the Seq2Seq model architecture.
- Trains the model with teacher forcing.
- Evaluates the model on a validation set.
- Saves the trained model as a checkpoint.

"""

# %%
from pathlib import Path
import sys, os
sys.path.append(str(Path(__file__).resolve().parent.parent))

# %%
import torch
import config
import json
import csv
from dataloaders.dataloader import create_dataloaders
from training.loop import train_one_epoch, evaluate
from models import seq2seq, seq2seq_attn

# %%
def get_device():
    """Get the available device (GPU or MPS if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
# %%
def load_vocabs():
    """
    Load source and target vocabularies from disk.
    """

    vocab_src_path = config.VOCAB_DIR / "vocab_src.json"
    vocab_tgt_path = config.VOCAB_DIR / "vocab_tgt.json"

    with open(vocab_src_path, "r", encoding="utf-8") as f:
        vocab_src = json.load(f)
    with open(vocab_tgt_path, "r", encoding="utf-8") as f:
        vocab_tgt = json.load(f)

    return vocab_src, vocab_tgt

# %%
def main():
    """
    Main training loop for Seq2Seq model.
    """

    device = get_device()
    print(f"Using device: {device}")

    # Load vocabularies
    vocab_src, vocab_tgt = load_vocabs()
    vocab_size_src = len(vocab_src)
    vocab_size_tgt = len(vocab_tgt)
    print(f"Source vocab size: {vocab_size_src}, Target vocab size: {vocab_size_tgt}")

    # Get data loaders
    train_loader, val_loader, test_loader = create_dataloaders(debug=config.DEBUG_MODE)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # define model based on config.MODEL_NAME
    print(f"Defining model: {config.MODEL_NAME}...")
    if config.MODEL_NAME == "seq2seq":
        EncoderCls, DecoderCls, Seq2SeqCls = seq2seq.Encoder, seq2seq.Decoder, seq2seq.Seq2Seq
    elif config.MODEL_NAME == "seq2seq_attn":
        EncoderCls, DecoderCls, Seq2SeqCls = seq2seq_attn.Encoder, seq2seq_attn.Decoder, seq2seq_attn.Seq2SeqAttn
    else:
        raise ValueError(f"Unsupported MODEL_NAME: {config.MODEL_NAME}")

    encoder = EncoderCls(
        vocab_size_src=vocab_size_src,
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        pad_id=config.PAD_ID,
    )

    decoder = DecoderCls(
        vocab_size_tgt=vocab_size_tgt,
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        pad_id=config.PAD_ID,
    )

    model = Seq2SeqCls(
        encoder=encoder,
        decoder=decoder,
        sos_id=config.SOS_ID,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model defined with {num_params} trainable parameters.")

    # define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=config.PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # checkpoint directory
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    checkpoint_path = config.CHECKPOINTS_DIR / config.CHECKPOINT_FILENAME

    # training loop
    print("Starting training...")

    best_dev_loss = float("inf")
    epochs_since_improve = 0
    train_history = []
    dev_history = []
    
    # linearly decay teacher forcing from start to end across epochs
    tf_schedule = torch.linspace(
        config.TEACHER_FORCING_START,
        config.TEACHER_FORCING_END,
        steps=config.NUM_EPOCHS,
    ).tolist()

    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        teacher_forcing_ratio = tf_schedule[epoch]

        # train one epoch
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        print(f"  Train Loss: {train_loss:.4f} | TF ratio: {teacher_forcing_ratio:.3f}")
        train_history.append(train_loss)

        # evaluate on validation set
        dev_loss = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(f"  Val Loss: {dev_loss:.4f}")
        dev_history.append(dev_loss)

        # save best model / early stopping
        if dev_loss < (best_dev_loss - config.EARLY_STOP_MIN_DELTA):
            best_dev_loss = dev_loss
            epochs_since_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved best model with Val Loss: {best_dev_loss:.4f}")
        else:
            epochs_since_improve += 1
            print(f"  No improvement for {epochs_since_improve} epoch(s).")
            if epochs_since_improve >= config.EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered after {epochs_since_improve} epochs without improvement.")
                break

    # persist loss history
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    metrics_path = config.METRICS_DIR / config.METRICS_FILENAME
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "dev_loss"])
        for idx, (tr, dv) in enumerate(zip(train_history, dev_history), start=1):
            writer.writerow([idx, tr, dv])
    print(f"Saved loss history to {metrics_path}")

if __name__ == "__main__":
    main()
