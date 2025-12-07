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
    Decoder module using GRU.
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

        # Linear layer to project GRU outputs to vocabulary size
        self.fc_out = nn.Linear(hidden_dim, vocab_size_tgt)

    def forward(
            self,
            tgt_step: torch.Tensor,
            hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the Decoder.
        Inputs:
        - tgt_step: Tensor [B] (current timestep token ids)
        - hidden: Tensor [num_layers, B, hidden_dim] 

        Outputs:
        - logits: Tensor [B, vocab_size_tgt]
        - hidden: Tensor [num_layers, B, hidden_dim]

        """

        # tgt_step: [B] -> [B, 1]
        embedded = self.embedding(tgt_step.unsqueeze(1))  # [B, 1, embed_dim], unsqueeze needed for time step dimension

        # Pass through GRU for a single timestep
        gru_out, gru_hidden = self.gru(embedded, hidden)      # gru_out: [B, 1, hidden_dim], gru_hidden: [num_layers, B, hidden_dim]

        # Project GRU outputs to vocabulary size
        logits = self.fc_out(gru_out.squeeze(1))          # [B, vocab_size_tgt], squeeze to remove time step dimension

        return logits, gru_hidden
    
class Seq2Seq(nn.Module):
    """
    Seq2Seq model combining Encoder and Decoder.
    
    Inputs:
    - src_padded: Tensor [B, T_src]
    - src_lens: Tensor [B]
    - tgt_padded: Tensor [B, T_tgt]

    Outputs:
    - logits: Tensor [B, T_tgt-1, V_tgt]    --> because we do not predict <SOS> token

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
            teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        
        device = src_padded.device  # Get the device (CPU or GPU)
        B = src_padded.size(0)      # Batch size
        T_tgt = tgt_padded.size(1)  # Target sequence length
        V_tgt = self.decoder.embedding.num_embeddings   # Target vocabulary size

        # Encode the source sequences
        _, encoder_hidden = self.encoder(src_padded, src_lens)      # encoder_hidden: [num_layers, B, hidden_dim]
        hidden = encoder_hidden  # Initialize decoder hidden state with encoder's final hidden state

        # prepare output tensor
        outputs = torch.zeros(B, T_tgt-1, V_tgt).to(device)         # Exclude <SOS> token time step

        # First input to the decoder is the <SOS> tokens
        input_decoder = torch.full((B,), self.sos_id, dtype=torch.long).to(device)

        for t in range(1, T_tgt):   # t=1 to T_tgt-1, since t=0 is <SOS>
            
            # Decode one time step based on current input and hidden state
            logits, hidden = self.decoder(
                tgt_step=input_decoder,
                hidden=hidden,
            )  # logits: [B, 1, V_tgt], note that returned hidden is updated decoder hidden state

            outputs[:, t-1, :] = logits  # Store the output logits for this time step

            # Decide whether to use teacher forcing --> that is, decide between using GOLD target token or PREDICTED token as next input
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio

            if teacher_force:
                # Use actual next token as next input
                input_decoder = tgt_padded[:, t]
            
            else:
                # Use predicted token as next input
                input_decoder = logits.argmax(1)     # argmax along vocab dimension
            
        return outputs
        
