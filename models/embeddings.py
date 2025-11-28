import math
import torch
import torch.nn as nn
from typing import Optional


def get_attention(
    d_model: int,
    nhead: int,
    dropout: float = 0.1,
    max_position_embeddings: int = 4096,
    bias: bool = True,
    attn: str = "learned",
    batch_first: bool = True,
):
    if attn == "learned":
        return LearnedMultiHeadAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            bias=bias,
            max_position_embeddings=max_position_embeddings,
            batch_first=batch_first,
        )
    elif attn == "sinusoidal":
        return SinusoidalMultiHeadAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            bias=bias,
            max_position_embeddings=max_position_embeddings,
            batch_first=batch_first,
        )
    else:
        raise ValueError(f"Unknown attn type {attn}")


class SinusoidalMultiHeadAttention(nn.Module):
    """Multi-head attention with sinusoidal positional embeddings"""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        bias: bool = True,
        max_position_embeddings: int = 10000,
        batch_first: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.max_position_embeddings = max_position_embeddings
        self.batch_first = batch_first

        # wrap built-in MHA
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
        )
        self.register_buffer(
            "pos_emb",
            self._create_sinusoidal_embeddings(max_position_embeddings, d_model),
        )

    def _create_sinusoidal_embeddings(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> torch.Tensor:

        bsz, seq_len, _ = query.size()
        kv_len = key.size(1)

        # check lengths
        if (
            seq_len > self.max_position_embeddings
            or kv_len > self.max_position_embeddings
        ):
            raise ValueError(
                f"Sequence length exceeds max_position_embeddings={self.max_position_embeddings}"
            )

        # add pos emb
        q = query + self.pos_emb[:seq_len].unsqueeze(0)
        k = key + self.pos_emb[:kv_len].unsqueeze(0)

        # causal mask
        if is_causal:
            causal = torch.triu(
                torch.ones(seq_len, kv_len, device=query.device), diagonal=1
            ).bool()
            attn_mask = attn_mask | causal if attn_mask is not None else causal

        # run MHA
        out, weights = self.mha(
            query=q,
            key=k,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )
        return out, weights


class LearnedMultiHeadAttention(nn.Module):
    """Multi-head attention with learned positional embeddings"""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        bias: bool = True,
        max_position_embeddings: int = 2048,
        batch_first: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.max_position_embeddings = max_position_embeddings
        self.batch_first = batch_first

        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
        )
        self.q_pos_emb = nn.Embedding(max_position_embeddings, d_model)
        self.k_pos_emb = nn.Embedding(max_position_embeddings, d_model)
        nn.init.normal_(self.q_pos_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.k_pos_emb.weight, mean=0.0, std=0.02)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> torch.Tensor:

        bsz, seq_len, _ = query.size()
        kv_len = key.size(1)
        if (
            seq_len > self.max_position_embeddings
            or kv_len > self.max_position_embeddings
        ):
            raise ValueError(
                f"Sequence length exceeds max_position_embeddings={self.max_position_embeddings}"
            )

        device = query.device
        positions_q = torch.arange(seq_len, device=device)
        positions_k = torch.arange(kv_len, device=device)

        q = query + self.q_pos_emb(positions_q).unsqueeze(0)
        k = key + self.k_pos_emb(positions_k).unsqueeze(0)

        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, kv_len, device=query.device), diagonal=1
            ).bool()
            if attn_mask is not None:
                # ensure attn_mask is bool before OR
                attn_mask = torch.logical_or(attn_mask.bool(), causal_mask)
            else:
                attn_mask = causal_mask

        out, weights = self.mha(
            query=q,
            key=k,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )
        return out, weights
