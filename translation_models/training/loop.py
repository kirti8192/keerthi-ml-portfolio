"""
Training utilities for seq2seq models.

Provides:
- train_one_epoch(): runs one training epoch with teacher forcing.
- evaluate(): computes validation/test loss with full teacher forcing.

Expects batches with: src_padded, tgt_padded, src_lens, tgt_lens.
Loss is averaged over non-PAD tokens.
"""

import torch
import torch.nn as nn

def train_one_epoch(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        teacher_forcing_ratio: float,
    ) -> float:
    """
    Train the model for one epoch.
    """

    # put model in train mode
    model.train()

    # loss accumulator
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        src_padded = batch['src_padded'].to(device)
        tgt_padded = batch['tgt_padded'].to(device)
        src_lens = batch['src_lens'].to(device)

        optimizer.zero_grad()

        model_output = model(
            src_padded,
            src_lens,
            tgt_padded,
            teacher_forcing_ratio=teacher_forcing_ratio
        )
        logits = model_output[0] if isinstance(model_output, tuple) else model_output

        # Shift targets to align with outputs
        tgt_input = tgt_padded[:, 1:]  # exclude <SOS>

        B, T, V = logits.size() # Batch, Time, Vocab size

        # compute loss by comparing logits with tgt_input. Note logits are for each token in the vocab
        loss = criterion(
            logits.reshape(B * T, V),
            tgt_input.reshape(B * T)
        )

        # gradient descent step
        loss.backward()
        optimizer.step()

        # track loss
        with torch.no_grad():
            num_tokens = (tgt_input != criterion.ignore_index).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
        
    return total_loss / max(total_tokens, 1)

def evaluate(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> float:
    """
    Evaluate the model on validation/test set.
    """

    # put model in eval mode
    model.eval()

    # loss accumulator
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            src_padded = batch['src_padded'].to(device)
            tgt_padded = batch['tgt_padded'].to(device)
            src_lens = batch['src_lens'].to(device)

            model_output = model(
                src_padded,
                src_lens,
                tgt_padded,
                teacher_forcing_ratio=1.0  # teacher forcing during eval
            )
            logits = model_output[0] if isinstance(model_output, tuple) else model_output

            # Shift targets to align with outputs
            tgt_input = tgt_padded[:, 1:]

            B, T, V = logits.size() # Batch, Time, Vocab size

            # compute loss by comparing logits with tgt_input. Note logits are for each token in the vocab
            loss = criterion(
                logits.reshape(B * T, V),
                tgt_input.reshape(B * T)
            )

            num_tokens = (tgt_input != criterion.ignore_index).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / max(total_tokens, 1)
