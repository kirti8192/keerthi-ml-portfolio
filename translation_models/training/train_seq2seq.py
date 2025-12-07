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
from dataloaders.dataloader import create_dataloaders
from models.seq2seq import Encoder, Decoder, Seq2Seq
from training.loop import train_one_epoch, evaluate

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

    # define model 
    print("Defining model...")

    encoder = Encoder(
        vocab_size_src=vocab_size_src,
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        pad_id=config.PAD_ID,
    )

    decoder = Decoder(
        vocab_size_tgt=vocab_size_tgt,
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        pad_id=config.PAD_ID,
    )

    model = Seq2Seq(
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
    checkpoint_path = config.CHECKPOINTS_DIR / "seq2seq_checkpoint.pth"

    # training loop
    print("Starting training...")

    best_dev_loss = float("inf")
    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")

        # train one epoch
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            teacher_forcing_ratio=config.TEACHER_FORCING,
        )

        print(f"  Train Loss: {train_loss:.4f}")

        # evaluate on validation set
        dev_loss = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(f"  Val Loss: {dev_loss:.4f}")

        # save best model
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved best model with Val Loss: {best_dev_loss:.4f}")

if __name__ == "__main__":
    main()