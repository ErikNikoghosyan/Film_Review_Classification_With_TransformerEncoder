"""
simple_transformer_encoder
--------------------------

Lightweight Transformer encoder-based classifier intended for short-text
sentiment classification tasks.

Key features
- Token embedding plus sinusoidal positional encodings.
- Stacked nn.TransformerEncoder layers with configurable model dimension,
  number of heads and depth.
- Padding-aware attention via `attention_mask` (format: 1=token, 0=pad).
- Configurable MLP classification head with dropout and activation options.
- Returns raw logits suitable for BCEWithLogitsLoss (apply sigmoid for probs).

Typical usage
- Instantiate with vocabulary size (input_dim), model_dim, num_heads, num_layers,
  and optional FC head parameters.
- Call forward with `src` (LongTensor of shape (batch_size, seq_len)) and an
  optional `attention_mask` (tensor of same shape).
- The forward method returns a 1-D tensor of logits with shape (batch_size,).

Notes
- Positional encodings are generated and added during forward; ensure the model
  is moved to the desired device (model.to(device)) before inference/training.
- The implementation internally permutes between (batch, seq, embed) and
  (seq, batch, embed) as required by PyTorch's Transformer when batch_first=False.
"""

import torch
import torch.nn as nn


class SimpleTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        model_dim,
        num_heads,
        num_layers,
        fc_params=[128, 64, 32],
        fc_activation=nn.ReLU,
        fc_dropout=0.0,
        dropout=0.1,
        max_seq_length=512,
        device="cpu",
    ):
        super().__init__()

        self.device = device
        self.model_dim = model_dim

        # Embedding & Positional Encoding
        self.embedding = nn.Embedding(input_dim, model_dim)

        self.positional_encoding = self._generate_positional_encoding(
            max_seq_length, model_dim
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=False,  # PyTorch expects (seq, batch, embed)
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.dropout_layer = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

        # Fully Connected Head
        layers = []
        in_features = model_dim

        # Hidden layers except last
        for out_features in fc_params[:-1]:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(fc_activation())
            in_features = out_features

        # Last hidden layer → dropout → activation
        last_hidden = fc_params[-1]
        layers.append(nn.Linear(in_features, last_hidden))
        layers.append(fc_activation())
        layers.append(nn.Dropout(fc_dropout))

        # Final output
        layers.append(nn.Linear(last_hidden, 1))

        self.fc_out = nn.Sequential(*layers)

    # Positional Encoding
    def _generate_positional_encoding(self, max_seq_length, model_dim):
        position = torch.arange(0, max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2) * -(torch.log(torch.tensor(10000.0)) / model_dim)
        )
        pe = torch.zeros(max_seq_length, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, seq, model_dim)

    # Forward Pass
    def forward(self, src, attention_mask=None):
        """
        src: (batch, seq)
        attention_mask: (batch, seq) with 1=keep, 0=pad
        """

        seq_length = src.size(1)

        # Embed 
        embedded = self.embedding(src) * torch.sqrt(torch.tensor(self.model_dim, dtype=torch.float32))
        embedded += self.positional_encoding[:, :seq_length, :].to(src.device)

        # (batch, seq, embed) → (seq, batch, embed)
        embedded = self.dropout_layer(embedded).permute(1, 0, 2)

        # Attention mask
        if attention_mask is not None:
            # src_key_padding_mask: True = ignore, False = keep
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None

        # Transformer
        transformer_output = self.transformer_encoder(
            embedded,
            src_key_padding_mask=src_key_padding_mask
        )

        # (seq, batch, embed) → (batch, seq, embed)
        transformer_output = transformer_output.permute(1, 0, 2)

        # Pooling
        pooled_output = torch.mean(transformer_output, dim=1)
        pooled_output = self.layer_norm(pooled_output)

        # Feed-forward 
        output = self.fc_out(pooled_output)

        return output.squeeze(1)  # (batch,)
