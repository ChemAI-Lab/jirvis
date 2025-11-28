import torch
import torch.nn as nn
import math
from typing import Optional, List

Tensor = torch.Tensor


class NMRTransformerRegressor(nn.Module):
    """Transformer regressor for NMR data (numerical targets)"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 300,
        num_classes: int = 19,
        use_scalar: bool = True,  # Use scalar embeddings
        n_scalar_features: int = 1,  # Only 1 scalar feature as requested
        scalar_num_layers: int = 2,
        **kwargs,
    ):
        super().__init__()

        # Token embeddings only (positional embeddings handled by attention layers)
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # Transformer encoder with scalar embeddings
        self.transformer = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            norm_first=True,
            batch_first=True,
            num_layers=num_layers,
            max_len=max_len,
            use_scalar=use_scalar,
            n_scalar_features=n_scalar_features,  # Just 1 scalar feature as requested
            scalar_num_layers=scalar_num_layers,
            scalar_hidden_dim=d_model,
        )

        # Regression head (no sigmoid/softmax for numerical outputs)
        self.regressor = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, scalar_values, scalar_mask):
        # Token embeddings only
        x = self.token_embed(input_ids)

        # Create padding mask (True for padding tokens)
        padding_mask = ~attention_mask.bool()

        # Reshape scalar values to have 1 feature dimension
        scalars = scalar_values.unsqueeze(-1)  # [batch, seq_len, 1]

        # Transformer (positional embeddings handled internally)
        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask,
            scalars=scalars,
            scalar_mask=scalar_mask,
        )

        # Global average pooling (ignoring padding)
        mask = attention_mask.unsqueeze(-1).float()
        x = (x * mask).sum(dim=1) / mask.sum(dim=1)

        # Regression output (no activation function)
        x = self.dropout(x)
        outputs = self.regressor(x)

        return outputs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor [batch_size, seq_len, d_model]
        Returns:
            Tensor with positional encodings added: [batch_size, seq_len, d_model]
        """

        x = x + self.pe[:, : x.size(1)]
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder with optional scalar embeddings.

    Args:
        d_model: hidden size
        nhead: number of attention heads
        dim_feedforward: feedforward hidden size
        dropout: dropout rate
        activation: activation fn name for encoder layer
        norm_first: whether to apply norm before attention/ff
        batch_first: whether input is batch-first
        num_layers: number of encoder layers
        max_len: positional embedding length
        bias: whether attention uses bias
        attn: which attention type
        use_scalar: if True, adds scalar features via ScalarEmbedding
        n_scalar_features: number of scalar features
        scalar_num_layers: number of MLP layers in ScalarEmbedding
        scalar_hidden_dim: hidden dim of MLP in ScalarEmbedding
        scalar_activation: activation class for MLP
        scalar_layer_norm: whether to layer-norm scalar projection
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        norm_first: bool,
        batch_first: bool,
        num_layers: int,
        max_len: int = 5000,
        # bias: bool = True,
        # attn: str = 'learned',
        use_scalar: bool = False,
        n_scalar_features: int = 0,
        scalar_num_layers: int = 1,
        scalar_hidden_dim: Optional[int] = None,
        scalar_activation: nn.Module = nn.ReLU,
        scalar_layer_norm: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.use_scalar = use_scalar
        if use_scalar:
            self.scalar_embed = ScalarEmbedding(
                n_scalar_features,
                d_model,
                num_layers=scalar_num_layers,
                hidden_dim=scalar_hidden_dim,
                activation=scalar_activation,
                use_layer_norm=scalar_layer_norm,
            )

        self.positional_encoding = PositionalEncoding(d_model, max_len)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=batch_first,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        scalars: Optional[torch.Tensor] = None,
        scalar_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            scalars: [batch, seq_len, n_scalar_features]
            src_key_padding_mask: [batch, seq_len]
        """
        if self.use_scalar:
            if scalars is None:
                raise ValueError("scalars must be provided when use_scalar=True")
            x = self.scalar_embed(x, scalars, scalar_mask=scalar_mask)
        x = self.positional_encoding(x)

        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


class ScalarEmbedding(nn.Module):
    """
    MLP-based projector for scalar features, then adds to token embeddings.

    Args:
        n_scalar_features: number of scalar input channels per token
        d_model: hidden dimension of transformer
        num_layers: number of linear layers in MLP (>=1)
        hidden_dim: intermediate hidden dim (if None, uses d_model)
        activation: activation class (default nn.ReLU)
        use_layer_norm: whether to apply LayerNorm on projected output
    """

    def __init__(
        self,
        n_scalar_features: int,
        d_model: int,
        num_layers: int = 1,
        hidden_dim: Optional[int] = None,
        activation: nn.Module = nn.ReLU,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be at least 1"
        self.use_layer_norm = use_layer_norm
        hidden_dim = hidden_dim or d_model

        # build layers
        dims: List[int] = (
            [n_scalar_features] + [hidden_dim] * (num_layers - 1) + [d_model]
        )
        []

        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.layers.append(nn.Linear(in_dim, out_dim, bias=True))
        self.activation = activation()
        if use_layer_norm:
            self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        scalars: torch.Tensor,
        scalar_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: token embeddings [batch, seq_len, d_model]
            scalars: scalar features [batch, seq_len, n_scalar_features]
        Returns:
            combined embeddings [batch, seq_len, d_model]
        """
        out = scalars
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        # final layer (no activation)
        out = self.layers[-1](out)

        if self.use_layer_norm:
            out = self.norm(out)

        if scalar_mask is not None:
            scalar_mask = scalar_mask.unsqueeze(-1).float()  # [B, S, 1]
            out = out * scalar_mask  # zero out padded positions

        # add to token embeddings
        return x + out
