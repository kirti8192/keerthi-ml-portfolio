# %%
# Seq2Seq + Attention (GRU) Architecture Flow
#
# SOURCE TOKENS → Src Embedding → Encoder GRU → All Encoder Hidden States (H_enc)
#
# DECODING (step t):
#   Previous Token (y_{t-1}) → Tgt Embedding → Decoder GRU (with previous hidden h_{t-1})
#                                                    ↓
#                                Attention(h_t, H_enc) → Attention Weights → Context Vector c_t
#                                                    ↓
#                           [h_t ; c_t] → Linear → Softmax → Predict y_t (next token)
#                                                    ↓
#                                   Choose next input token (teacher forcing or argmax)

# %%
# Seq2Seq (GRU) Architecture Flow
# SOURCE TOKENS → Embedding → Encoder GRU → Final Hidden → Decoder Initial Hidden
#                                                              ↓
#                                      <SOS> → Embedding → Decoder GRU → Linear → Softmax → next token
#                                                              ↓
#                                                        feed next input token

# %%
import torch
import torch.nn as nn

# %%
class Encoder(nn.Module):
    """
    Encoder module using GRU.
    Inputs: 
    - src_padded : Tensor [B, T_src]
    - src_lens: Tensor [B]

    Outputs:
    - gru_out: Tensor [B, T_src, hidden_dim]
    - gru_hidden: Tensor [num_layers, B, hidden_dim]
    """

    def __init__(
            self, 
            vocab_size_src: int,
            embed_dim: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float,
            pad_id: int,
    ):

        super().__init__()

        # Embedding layer: [B, T_src] -> [B, T_src, embed_dim]
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size_src,
            embedding_dim=embed_dim,
            padding_idx=pad_id,
        )

        # GRU layer: [B, T_src, embed_dim] -> [num_layers, B, hidden_dim]
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(
            self,
            src_padded: torch.Tensor,
            src_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the Encoder.
        Inputs:
        - src_padded : Tensor [B, T_src]
        - src_lens: Tensor [B]

        Outputs:
        - output: Tensor [B, T_src, hidden_dim]
        - hidden: Tensor [num_layers, B, hidden_dim]
        """

        # Embed the source sequences
        embedded = self.embedding(src_padded)  # [B, T_src, embed_dim]

        # Pack the embedded sequences for variable length processing
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths=src_lens.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # Pass through GRU
        gru_out, gru_hidden = self.gru(packed)

        # unpack the GRU outputs
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(
            gru_out,
            batch_first=True,
        )

        return gru_out, gru_hidden  # [B, T_src, hidden_dim], [num_layers, B, hidden_dim]
    
class Decoder(nn.Module):
    """
    Decoder module using GRU with attention
    Inputs:
    - input_tokens: Tensor [B] (current timestep token ids)
    - hidden: Tensor [num_layers, B, hidden_dim]

    Outputs:
    - logits: Tensor [B, vocab_size_tgt]
    - hidden: Tensor [num_layers, B, hidden_dim]
    """

    def __init__(
            self,
            vocab_size_tgt: int,
            embed_dim: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float,
            pad_id: int,
    ):

        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size_tgt,
            embedding_dim=embed_dim,
            padding_idx=pad_id,
        )

        # GRU layer
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Attention layer and score
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

        # Linear layer to project GRU outputs to vocabulary size
        self.fc_out = nn.Linear(hidden_dim, vocab_size_tgt)

    def forward(
            self,
            enc_out: torch.Tensor,
            tgt_step: torch.Tensor,
            hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the Decoder with attention
        Inputs:
        - encoder_outputs: Tensor [B, T_src, hidden_dim]
        - tgt_step: Tensor [B] (current timestep token ids)
        - hidden: Tensor [num_layers, B, hidden_dim] 

        Outputs:
        - logits: Tensor [B, vocab_size_tgt]
        - hidden: Tensor [num_layers, B, hidden_dim]

        """

        # Embed target token
        # tgt_step: [B] -> [B, 1]
        embedded = self.embedding(tgt_step.unsqueeze(1))  # [B, 1, embed_dim], unsqueeze needed for time step dimension

        # Pass through GRU for a single timestep
        gru_out, gru_hidden = self.gru(embedded, hidden)      # gru_out: [B, 1, hidden_dim], gru_hidden: [num_layers, B, hidden_dim]

        # Compute attention scores
        decoder_hidden = gru_hidden[-1].unsqueeze(1)        # [B, 1, hidden_dim]
        repeated_decoder_hidden = decoder_hidden.repeat(1, enc_out.size(1), 1)  # [B, T_src, hidden_dim]

        # Concatenate encoder outputs and decoder hidden state
        attn_input = torch.cat((enc_out, repeated_decoder_hidden), dim=2)  # [B, T_src, hidden_dim*2]

        # Compute energy scores
        energy = torch.tanh(self.attn(attn_input))          # [B, T_src, hidden_dim]
        scores = torch.matmul(energy, self.v)            # [B, T_src]

        # Compute attention weights
        attn_weights = torch.softmax(scores, dim=1)         # [B, T_src]

        # context vector as weighted sum of encoder outputs
        context = torch.bmm(attn_weights.unsqueeze(1), enc_out)  # [B, 1, hidden_dim]

        # combine GRU output and context vector
        combined = torch.cat((gru_out, context), dim=2)  # [B, 1, hidden_dim*2]

        # Project GRU outputs to vocabulary size
        logits = self.fc_out(combined.squeeze(1))  # [B, vocab_size_tgt]

        return logits, gru_hidden, attn_weights
    
class Seq2SeqAttn(nn.Module):
    """
    Seq2Seq model combining Encoder and Decoder with attention.

    Inputs:
    - src_padded: Tensor [B, T_src]
        Padded source sequences (input to the encoder).
    - src_lens: Tensor [B]
        Lengths of the source sequences (for packing/unpacking).
    - tgt_padded: Tensor [B, T_tgt]
        Padded target sequences (input to the decoder during training).
    - teacher_forcing_ratio: float
        Probability of using the ground truth token as the next input during training.

    Outputs:
    - outputs: Tensor [B, T_tgt-1, V_tgt]
        Logits for each timestep (excluding the <SOS> token).
    - attn_weights_all: Tensor [B, T_tgt-1, T_src] (optional)
        Attention weights for each timestep (useful for visualization).
    """

    def __init__(
            self,
            encoder: Encoder,
            decoder: Decoder,
            sos_id: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_id = sos_id

    def forward(
            self,
            src_padded: torch.Tensor,
            src_lens: torch.Tensor,
            tgt_padded: torch.Tensor,
            teacher_forcing_ratio: float = 0.5,
    ) -> tuple:
        """
        Forward pass for the Seq2Seq model with attention.

        Args:
        - src_padded: Tensor [B, T_src]
            Padded source sequences.
        - src_lens: Tensor [B]
            Lengths of the source sequences.
        - tgt_padded: Tensor [B, T_tgt]
            Padded target sequences.
        - teacher_forcing_ratio: float
            Probability of using the ground truth token as the next input.

        Returns:
        - outputs: Tensor [B, T_tgt-1, V_tgt]
            Logits for each timestep (excluding the <SOS> token).
        - attn_weights_all: Tensor [B, T_tgt-1, T_src]
            Attention weights for each timestep.
        """

        # Get batch size and device
        B = src_padded.size(0)
        device = src_padded.device

        # Target sequence length and vocabulary size
        T_tgt = tgt_padded.size(1)  # Target sequence length
        V_tgt = self.decoder.embedding.num_embeddings  # Target vocabulary size

        # Encode the source sequences
        encoder_output, encoder_hidden = self.encoder(src_padded, src_lens)  # encoder_hidden: [num_layers, B, hidden_dim]
        hidden = encoder_hidden  # Initialize decoder hidden state with encoder's final hidden state

        # Prepare output tensor
        outputs = torch.zeros(B, T_tgt - 1, V_tgt).to(device)  # Exclude <SOS> token time step

        # Prepare tensor to store attention weights for all timesteps
        attn_weights_all = torch.zeros(B, T_tgt - 1, encoder_output.size(1)).to(device)  # [B, T_tgt-1, T_src]

        # First input to the decoder is the <SOS> tokens
        input_decoder = torch.full((B,), self.sos_id, dtype=torch.long).to(device)

        for t in range(1, T_tgt):  # t=1 to T_tgt-1, since t=0 is <SOS>

            # Decode one time step based on current input and hidden state
            logits, hidden, attn_weights = self.decoder(
                enc_out=encoder_output,
                tgt_step=input_decoder,
                hidden=hidden,
            )  # logits: [B, V_tgt], attn_weights: [B, T_src]

            # Store the output logits for this time step
            outputs[:, t - 1, :] = logits

            # Store the attention weights for this time step
            attn_weights_all[:, t - 1, :] = attn_weights

            # Decide whether to use teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio

            if teacher_force:
                # Use actual next token as next input
                input_decoder = tgt_padded[:, t]
            else:
                # Use predicted token as next input
                input_decoder = logits.argmax(1)  # argmax along vocab dimension

        # Return outputs and attention weights
        return outputs, attn_weights_all
