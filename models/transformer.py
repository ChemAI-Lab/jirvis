import torch
import torch.nn as nn
import math
from typing import Optional, Callable

from .embeddings import get_attention
from hydra.utils import instantiate


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_first: bool = False,
        batch_first: bool = True,
        max_position_embeddings: int = 10000,
        bias: bool = True,
        attn: str = "learned",
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=batch_first,
        )
        # overwrite the default attention
        self.self_attn = get_attention(
            d_model, nhead, dropout, max_position_embeddings, bias, attn
        )


class TransformerEncoder(nn.Module):
    """
    encoder:
        _target_: models.encoder_decoder.TransformerEncoder
        d_model:
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        activation:
        norm_first: bool,
        batch_first: bool,
        num_layers: int,
        max_len: int,
        use_learned_positional_encoding: bool

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
        bias: bool = True,
        attn: str = "learned",
    ):

        super().__init__()

        layer = CustomTransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            norm_first,
            batch_first,
            max_len,
            bias,
            attn,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x, src_key_padding_mask=None):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
            src_key_padding_mask: Optional mask for padded tokens [batch_size, seq_len]
        Returns:
            Encoded output of shape [batch_size, seq_len, d_model]
        """

        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_first: bool = False,
        batch_first: bool = True,
        max_position_embeddings: int = 5000,
        bias: bool = True,
        attn: str = "learned",
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=batch_first,
        )
        # replace the two MHA modules:
        # self.self_attn = get_attention(
        #     d_model, nhead, dropout, max_position_embeddings, bias, attn, batch_first
        # )
        # self.multihead_attn = get_attention(
        #     d_model, nhead, dropout, max_position_embeddings, bias, attn, batch_first
        # )


class TransformerDecoder(nn.Module):
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
        vocab_size: int,
        start_token_id: int,
        eos_token_id: Optional[int] = None,
        max_len: int = 77,
        attn: str = "learned",
    ):
        super().__init__()

        # build your custom layer
        layer = CustomTransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=batch_first,
            max_position_embeddings=max_len,
            bias=True,
            attn=attn,
        )
        # stack them
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.start_token_id = start_token_id
        self.eos_token_id = eos_token_id

        self.max_len = max_len
        self.jirvis_embedding_projection = nn.Linear(2048, d_model // 2)

    def generate_causal_mask(self, tgt_len: int, device: torch.device):
        # [tgt_len, tgt_len] with -inf above diagonal
        return torch.triu(
            torch.full((tgt_len, tgt_len), float("-inf"), device=device), diagonal=1
        )

    def predict(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        tgt:  [B, D]
        memory: [B, S, D]
        """
        B, T, _ = tgt.size()
        device = tgt.device

        # causal mask for decoder self-attn
        causal = self.generate_causal_mask(T, device)

        return self.decoder(
            tgt,
            memory,
            tgt_mask=causal,
            memory_mask=None,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
    
    def forward(
            self,
            jirvis_embedding: torch.Tensor,   # [B, 2048] before proj
            nmr_embedding: torch.Tensor,      # [B, d_model//2] if you concat
            decoder_in: torch.Tensor,         # [B, T] ground-truth tokens, including <sos> at t=0
            fusion: Callable = torch.cat,
        ):  
        # 1) Build memory once
        jirvis_proj = self.jirvis_embedding_projection(jirvis_embedding)  # [B, d_model//2]
        memory = fusion(jirvis_proj, nmr_embedding)          # [B, d_model]
        memory = memory.unsqueeze(1)                                      # [B, 1, d_model]

        # 2) Embed all inputs
        tgt_emb = self.embedding(decoder_in)                              # [B, T_in, d_model]

        # 3) Causal mask over T_in
        T = tgt_emb.size(1)
        causal = self.generate_causal_mask(T, tgt_emb.device)             # [T, T]

        # 4) Decode (be careful with batch_first)
        dec_out = self.decoder(
            tgt=tgt_emb, memory=memory, tgt_mask=causal,
            memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None
        )                                                                 # [B, T, d_model] if batch_first

        # 5) Project to vocab
        logits = self.output_proj(dec_out)                                # [B, T, V]
        return logits

    # def forward(
    #     self,
    #     jirvis_embedding: torch.Tensor,
    #     nmr_embedding: torch.Tensor,
    #     fusion: Callable = torch.cat,
    # ):
    #     """
    #     memory: [B, D] — encoder output (compressed)
    #     Returns: [B, max_len] — generated token IDs
    #     """

    #     # Project Jirvis embeddings to match d_model
    #     jirvis_embedding = self.jirvis_embedding_projection(jirvis_embedding)  # [B, D]

    #     # Fuse embeddings
    #     memory = fusion(jirvis_embedding, nmr_embedding)

    #     B, D = memory.shape
    #     device = memory.device

    #     # Prepare memory shape for decoder: [B, 1, D]
    #     memory = memory.unsqueeze(1)

    #     all_logits = []
    #     # Initialize target sequence with start token
    #     generated_ids = torch.full(
    #         (B, 1), self.start_token_id, dtype=torch.long, device=device
    #     )

    #     for _ in range(self.max_len):
    #         tgt_emb = self.embedding(generated_ids)  # [B, T, D]

    #         decoder_out = self.predict(tgt_emb, memory)  # [B, T, D]
    #         last_hidden = decoder_out[:, -1, :]  # [B, D]

    #         logits = self.output_proj(last_hidden)  # [B, vocab_size]
    #         all_logits.append(logits.unsqueeze(1))

    #         next_token = logits.argmax(dim=-1, keepdim=True)  # [B, 1]

    #         generated_ids = torch.cat([generated_ids, next_token], dim=1)  # [B, T+1]

    #         # Optional: early stopping if all sequences end
    #         if self.eos_token_id is not None:
    #             if (next_token == self.eos_token_id).all():
    #                 break
    #     return torch.cat(all_logits, dim=1)  # [B, T, vocab_size]
    #     return generated_ids[:, 1:]  # remove the initial <sos> token
