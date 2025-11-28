"""
This code is provided solely for the personal and private use of students
taking the CSC401H/2511H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Author: Arvid Frydenlund <arvie@cs.toronto.edu>

All of the files in this directory and all subdirectories are:
Copyright (c) 2024 University of Toronto
"""

import math
import torch
import torch.nn as nn

from torch import Tensor
from typing import Tuple

# import a2_utils


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x: Tensor):
        """
        Compute layer normalization
            y = gamma * (x - mu) / (sigma + eps) + beta where mu and sigma are computed over the feature dimension

        x: torch.Tensor, shape [batch_size, seq_len, d_model]
        return: torch.Tensor, shape [batch_size, seq_len, d_model]
        """
        # Get sd and mean
        mu, sigma = torch.mean(x, dim=2, keepdim=True), torch.std(x, dim=2, keepdim=True, unbiased=False)  # Shape [batch_size, seq_len, 1]
        
        
        # x - mu
        x_mu = x - mu 

        # Divide by (sigma + eps)
        x_norm = x_mu / (sigma + self.eps)

        # Apply gamma beta
        return x_norm * self.gamma + self.beta

        



class MultiHeadAttention(nn.Module):
    """
    Multi-head attention for both self-attention and cross-attention
    """

    def __init__(
        self,
        num_heads: int,
        d_model: int,
        dropout: float = 0.0,
        atten_dropout: float = 0.0,
        store_attention_scores: bool = False,
    ):
        """
        num_heads: int, the number of heads
        d_model: int, the dimension of the model
        dropout: float, the dropout rate
        atten_dropout: float, the dropout rate for the attention i.e. drops out full tokens
        store_attention_scores: bool, whether to store the attention scores for visualization
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        # Assume values and keys are the same size
        self.d_head = d_model // num_heads
        self.num_heads = num_heads

        # Note for students, for self-attention, it is more efficient to treat q, k, and v as one matrix
        # but this way allows us to use the same attention function for cross-attention
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.atten_dropout = nn.Dropout(p=atten_dropout)  # applied after softmax

        # applied at the end
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

        # used for visualization
        self.store_attention_scores = store_attention_scores
        self.attention_scores = None  # set by set_attention_scores

    def set_attention_scores(self, scores: Tensor) -> None:
        """
        A helper function for visualization of attention scores.
        These are stored as attributes so that students do not need to deal with passing them around.

        The attention scores should be given after masking but before the softmax.
        scores: torch.Tensor, shape [batch_size, num_heads, query_seq_len, key_seq_len]
        return: None
        """
        if scores is None:  # for clean up
            self.attention_scores = None
        if self.store_attention_scores and not self.training:
            self.attention_scores = scores.cpu().detach().numpy()

    def attention(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        """
        Scaled dot product attention
        Hint: the mask is applied before the softmax.
        Hint: attention dropout `self.atten_dropout` is applied to the attention weights after the softmax.

        You are required to make comments about the shapes of the tensors at each step of the way
        in order to assist the markers.  Does a tensor change shape?  Make a comment.

        You are required to call set_attention_scores with the correct tensor before returning from this function.
        The attention scores should be given after masking but before the softmax.
        This is for testing purposes and potentially for other uses.

        query: torch.Tensor, shape [batch_size, num_heads, query_seq_len, d_head]
        key: torch.Tensor, shape [batch_size, num_heads, key_seq_len, d_head]
        value: torch.Tensor, shape [batch_size, num_heads, key_seq_len, d_head]
        mask:  torch.Tensor, shape [batch_size, query_seq_len, key_seq_len,], True, where masked or None

        return torch.Tensor, shape [batch_size, num_heads, query_seq_len, d_head]
        """
        # Compute logits: Q * K^T / sqrt(d_k)
        key_transpose = key.transpose(-1, -2)                               # Shape: [batch_size, num_heads, d_head, key_seq_len]
        logits = query @ key_transpose                                      # Shape: [batch_size, num_heads, query_seq_len, key_seq_len]
        logits /= math.sqrt(self.d_head)

        # If mask is not none, replace maked positios with -inf
        if mask is not None:    
            
            mask = mask.unsqueeze(1)                                        # Shape: [batch_size, 1, query_seq_len, key_seq_len]
            # print(mask.shape)
            # print(logits.shape) 
            logits = logits.masked_fill(mask, float('-inf'))                # No shape change

        # Store logits for visualization
        self.set_attention_scores(logits)  

        # Apply softmax and dropout to logits
        # No shape change
        preds = self.atten_dropout(nn.functional.softmax(logits, dim=-1))    # Shape: [batch_size, num_heads, query_seq_len, key_seq_len]

        # Multply by V to get output
        output = preds @ value                                              # Shape: [batch_size, num_heads, query_seq_len, d_head]

        return output

    def forward(self, query: Tensor, key: Tensor = None, value: Tensor = None, mask: Tensor = None) -> Tensor:
        """
        If the key and values are None, assume self-attention is being applied.  Otherwise, assume cross-attention.

        Note we only need one mask, which will work for either causal self-attention or cross-attention as long as
        the mask is set up properly beforehand.

        You are required to make comments about the shapes of the tensors at each step of the way
        in order to assist the markers.  Does a tensor change shape?  Make a comment.

        query: torch.Tensor, shape [batch_size, query_seq_len, d_model]
        key: torch.Tensor, shape [batch_size, key_seq_len, d_model] or None
        value: torch.Tensor, shape [batch_size, key_seq_len, d_model] or None
        mask: torch.Tensor, shape [batch_size, query_seq_len, key_seq_len,], True where masked or None

        return: torch.Tensor, shape [batch_size, query_seq_len, d_model]
        """

        if key == None and value == None:
            key = query                                     # Shape: [batch_size, query_seq_len, d_model]
            value = query                                  # Shape: [batch_size, query_seq_len, d_model]

        # Tuple of self.num_head tensors of shape [batch_size, {type}_seq_len, self.d_head]
        Q = query.split(self.d_head, dim=-1)
        K = key.split(self.d_head, dim=-1) 
        V = value.split(self.d_head, dim=-1) 

        # Tensors of shape [batch_size, num_heads, {type}_seq_len, d_head]
        Q = torch.stack(Q, dim = 1)
        K = torch.stack(K, dim = 1)
        V = torch.stack(V, dim = 1)

        # Get attention output
        output = self.attention(query=Q, key=K, value=V, mask=mask)    # Shape:  [batch_size, num_heads, query_seq_len, d_head]
        
        # Cast back to original dims
        output = output.transpose(1, 2)                     # Shape: [batch_size, query_seq_len, num_heads, d_head]
        W_O_shape = (output.shape[0], output.shape[1], -1)
        W_O = output.reshape(W_O_shape)                     # Shape: [batch_size, query_seq_len, d_model]

        # Apply linear projection and dropout
        W = self.dropout(self.out_linear(W_O))                            # Shape: [batch_size, query_seq_len, d_model]

        return W



class FeedForwardLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForwardLayer, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.f = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the feedforward sublayer.
        Dropout is applied after the activation function and after the second linear layer

        x: torch.Tensor, shape [batch_size, seq_len, d_model]
        return: torch.Tensor, shape [batch_size, seq_len, d_model]
        """
        # First linear layer w relu then dropout
        h1 = self.dropout1(self.f(self.w_1(x)))

        # Second linear with dropout
        h2 = self.dropout2(self.w_2(h1))
        
        return h2

class TransformerDecoderLayer(nn.Module):
    """
    Performs multi-head self attention, multi-head cross attention, and FFN,
    with the desired pre- or post-layer norm and residual connections.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float = 0.1,
        atten_dropout: float = 0.0,
        is_pre_layer_norm: bool = True,
    ):
        """
        d_model: int, the dimension of the model
        d_ff: int, the dimension of the feedforward network interior projection
        num_heads: int, the number of heads for the multi-head attention
        dropout: float, the dropout rate
        atten_dropout: float, the dropout rate for the attention i.e. drops out full tokens
        is_pre_layer_norm: bool, whether to apply layer norm before or after each sublayer

        Please use the following attribute names 'self_attn', 'cross_attn', and 'ff' and any others you think you need.
        """
        super(TransformerDecoderLayer, self).__init__()
        
        self.is_pre_layer_norm = is_pre_layer_norm
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_head = num_heads

        # Layer norms
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.ln3 = LayerNorm(d_model)

        # Multi-head attention for self-attention and cross-attention
        self.self_attn = MultiHeadAttention(
            num_heads, d_model, dropout=dropout, atten_dropout=atten_dropout
        )
        self.cross_attn = MultiHeadAttention(
            num_heads, d_model, dropout=dropout, atten_dropout=atten_dropout
        )

        # Feed forward 
        self.ff = FeedForwardLayer(d_model, d_ff, dropout=dropout)

        # Dropout
        # self.dropout = nn.Dropout(dropout)

    def pre_layer_norm_forward(self, x: Tensor, mask: Tensor, src_x: Tensor, src_mask: Tensor) -> Tensor:
        """
        x: torch.Tensor, the input to the layer
        mask: torch.Tensor, the mask to apply to the attention
        """
        # Get ln1 and self attn. Add residual x
        x_1 = x + self.self_attn(query=self.ln1(x), mask=mask)

        # Apply ln2 and cross attentin. add residual x_1. USe src_x for k v and src_mask for mask
        x_2 = x_1 + self.cross_attn(query=self.ln2(x_1), key=src_x, value=src_x, mask=src_mask)

        # Apply ln3 and ff. Add residual x_2
        output = x_2 + self.ff(self.ln3(x_2))

        # Apply dropout
        # output = self.dropout(output)

        return output

    def post_layer_norm_forward(self, x: Tensor, mask: Tensor, src_x: Tensor, src_mask: Tensor) -> Tensor:
        """
        x: torch.Tensor, the input to the layer
        mask: torch.Tensor, the mask to apply to the attention
        """
        # Self attention then add residual x then apply ln1
        x_1 = self.ln1(self.self_attn(query=x, mask=mask) + x)

        # Cross attention. add residual x_1
        x_2 = self.cross_attn(query=x_1, key=src_x, value=src_x, mask=src_mask) + x_1 

        # ln2. store residual
        x_3 = self.ln2(x_2)

        # apply ff to x_3, add residual and finally apply ln3 
        output = self.ln3(self.ff(x_3) + x_3)

        return output
        
    def forward(self, x: Tensor, mask: Tensor, src_x: Tensor, src_mask: Tensor) -> Tensor:
        if self.is_pre_layer_norm:
            return self.pre_layer_norm_forward(x, mask, src_x, src_mask)
        else:
            return self.post_layer_norm_forward(x, mask, src_x, src_mask)

    def store_attention_scores(self, should_store: bool = True) -> None:
        self.self_attn.store_attention_scores = should_store
        self.cross_attn.store_attention_scores = should_store

    def get_attention_scores(self):
        return self.self_attn.attention_scores, self.cross_attn.attention_scores


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float = 0.1,
        atten_dropout: float = 0.0,
        is_pre_layer_norm: bool = True,
    ):
        super(TransformerDecoder, self).__init__()
        self.is_pre_layer_norm = is_pre_layer_norm
        self.layers = torch.nn.ModuleList()
        for l in range(num_layers):
            self.layers.append(
                TransformerDecoderLayer(
                    d_model, d_ff, num_heads, dropout, atten_dropout, is_pre_layer_norm
                )
            )
        self.norm = LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)  # logit projection

    def forward(self, x: Tensor, mask: Tensor, src_x: Tensor, src_mask: Tensor,
                normalize_logits: bool = False) -> Tensor:
        """
        x: torch.Tensor, the input to the decoder
        mask: torch.Tensor, the mask to apply to the attention
        src_x: torch.Tensor, the output of the encoder
        src_mask: torch.Tensor, the mask to apply to the attention
        normalize_logits: bool, whether to apply log_softmax to the logits

        Returns the logits or log probabilities if normalize_logits is True

        Hint: look at the encoder for how pre/post layer norm is handled
        """
        # Apply initial norm if using post norm
        if not self.is_pre_layer_norm:
            x = self.norm(x)

        for layer in self.layers:
            x = layer(x, mask, src_x, src_mask)

        # Apply final norm if using pre norm
        if self.is_pre_layer_norm:
            x = self.norm(x)

        # Project logits
        x = self.proj(x)

        # Apply log-softmax if normalize_logits is True
        if normalize_logits:
            x = nn.functional.log_softmax(x, dim=-1)

        return x
    def store_attention_scores(self, should_store: bool = True) -> None:
        for layer in self.layers:
            layer.store_attention_scores(should_store)

    def get_attention_scores(self):
        """
        Return the attention scores (self-attention, cross-attention) from all layers
        """
        scores = []
        for layer in self.layers:
            scores.append(layer.get_attention_scores())
        return scores

class TransformerEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super(TransformerEmbeddings, self).__init__()
        self.lookup = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: Tensor) -> Tensor:
        """
        x: torch.Tensor, shape [batch_size, seq_len] of int64 in range [0, vocab_size)
        return torch.Tensor, shape [batch_size, seq_len, d_model]
        """
        return self.lookup(x) * math.sqrt(self.d_model)
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectroTFDecoder(nn.Module):
    """
    Cross-attention conditioning:
      - Take jirvis (B, d_jirvis) and nmr (B, d_nmr) embeddings
      - Project each to common dim, then fuse -> (B, d_embeddings)
      - Project -> m context tokens (B, m, d_model)  [memory]
      - Decoder attends to memory via cross-attention to generate SELFIES tokens

    Forward (training):
      logits = model.forward(jirvis_emb, nmr_emb, tgt_ids)  # (B, T, vocab_size)

    Generate (inference):
      out_ids = model.generate(jirvis_emb, nmr_emb, bos_id, eos_id, max_new_tokens=..., top_p=..., temperature=...)
    """


    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        d_jirvis: int = 2048,
        d_nmr: int = 256,
        d_common: int = 512,
        m: int = 16,
        max_len: int = 77,
        use_positional_embeddings: bool = True,
    ):  
        super(SpectroTFDecoder, self).__init__()
        self.d_model = d_model
        self.d_jirvis = d_jirvis
        self.d_nmr = d_nmr
        self.d_common = d_common
        self.d_embeddings = 2 * d_common  # Concatenated common dims
        self.m = m
        self.max_len = max_len
        self.max_pos = max_len + m
        self.vocab_size = vocab_size
        self.use_positional_embeddings = use_positional_embeddings

        # Project embeddings to common dimension
        self.jirvis_proj = nn.Sequential(
            nn.Linear(d_jirvis, d_common),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.nmr_proj = nn.Sequential(
            nn.Linear(d_nmr, d_common),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Your provided decoder stack (self + cross attn inside)
        self.model = TransformerDecoder(
            vocab_size,
            num_layers,
            d_model,
            d_ff,
            num_heads,
            dropout=0.1,
            atten_dropout=0.1,
            is_pre_layer_norm=True,
        )

        # Token embeddings for SELFIES ids
        self.tok_emb = TransformerEmbeddings(vocab_size, d_model)

        # Project fused embeddings -> m memory tokens
        self.cond_to_mem = nn.Sequential(nn.Linear(self.d_embeddings, m * d_model), nn.Tanh())

        # (Optional) learned positions shared by targets + memory
        if self.use_positional_embeddings:
            self.pos_emb = nn.Embedding(self.max_pos, d_model)

    # ---------- Utilities ----------

    def fuse_embeddings(self, jirvis_emb: torch.Tensor, nmr_emb: torch.Tensor) -> torch.Tensor:
        """
        Project and fuse embeddings.
        jirvis_emb: (B, d_jirvis)  
        nmr_emb: (B, d_nmr)
        return: (B, d_embeddings)
        """
        jirvis_proj = self.jirvis_proj(jirvis_emb)  # (B, d_common)
        nmr_proj = self.nmr_proj(nmr_emb)           # (B, d_common)
        fused = torch.cat([jirvis_proj, nmr_proj], dim=-1)  # (B, 2*d_common)
        return fused

    def make_memory(self, cond: torch.Tensor) -> torch.Tensor:
        """
        cond: (B, d_embeddings)  ->  memory: (B, m, d_model)
        """
        B = cond.size(0)
        mem = self.cond_to_mem(cond).view(B, self.m, self.d_model)
        return mem

    @staticmethod
    def _causal_mask(B: int, T: int, device: torch.device) -> torch.Tensor:
        """
        Boolean mask True where masked; shape [B, T, T]
        """
        base = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
        return base.unsqueeze(0).expand(B, -1, -1)

    @staticmethod
    def _all_false_mask(B: int, T: int, M: int, device: torch.device) -> torch.Tensor:
        """
        Cross-attn mask; True = masked. Usually all False. Shape [B, T, M]
        """
        return torch.zeros(B, T, M, dtype=torch.bool, device=device)

    # ---------- Positional helpers (optional) ----------

    def _add_pos_target(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positions to target tokens.
        We offset target positions by m so memory & target positions don't collide.
        x: (B, T, d_model)
        """
        if not self.use_positional_embeddings:
            return x
        B, T, _ = x.shape
        device = x.device
        idx = torch.arange(self.m, self.m + T, device=device) % self.max_pos  # (T,)
        pos = self.pos_emb(idx)[None, :, :]  # (1, T, d_model)
        return x + pos

    def _add_pos_memory(self, mem: torch.Tensor) -> torch.Tensor:
        """
        Add learned positions to memory tokens.
        mem: (B, m, d_model)
        """
        if not self.use_positional_embeddings:
            return mem
        device = mem.device
        idx = torch.arange(0, self.m, device=device) % self.max_pos  # (m,)
        pos = self.pos_emb(idx)[None, :, :]  # (1, m, d_model)
        return mem + pos

    # ---------- Training forward ----------

    def forward(
        self,
        jirvis_emb: torch.Tensor,   # (B, d_jirvis)
        nmr_emb: torch.Tensor,      # (B, d_nmr)
        tgt_ids: torch.Tensor,      # (B, T) gold tokens (NOT shifted)
        bos_id: int = 0,
        normalize_logits: bool = False,
    ) -> torch.Tensor:
        """
        Teacher-forced forward pass. Returns logits over vocabulary for each position in tgt_ids.
        """
        device = tgt_ids.device
        B, T = tgt_ids.shape

        # Fuse embeddings
        cond = self.fuse_embeddings(jirvis_emb, nmr_emb)  # (B, d_embeddings)

        # Shift-right inputs: prepend BOS, drop last token
        bos = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        inp_ids = torch.cat([bos, tgt_ids[:, :-1]], dim=1)  # (B, T)

        # Token + (optional) positional embeddings for targets
        x = self.tok_emb(inp_ids)                # (B, T, d_model)
        x = self._add_pos_target(x)              # (B, T, d_model)

        # Memory from cond (+ positions)
        mem = self._add_pos_memory(self.make_memory(cond))  # (B, m, d_model)

        # Masks
        tgt_mask = self._causal_mask(B, T, device)          # (B, T, T), True=masked
        src_mask = self._all_false_mask(B, T, self.m, device)  # (B, T, m)

        # Decode
        logits = self.model(x, tgt_mask, mem, src_mask, normalize_logits=normalize_logits)  # (B, T, V)
        return logits

    # ---------- Inference (autoregressive generate) ----------

    @torch.no_grad()
    def generate(
        self,
        jirvis_emb: torch.Tensor,  # (B, d_jirvis)
        nmr_emb: torch.Tensor,     # (B, d_nmr)
        bos_id: int,
        eos_id: int,
        max_new_tokens: int = 128,
        top_p: float = 0.9,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Returns generated token ids including the initial BOS and (if reached) EOS.
        """
        device = jirvis_emb.device
        B = jirvis_emb.size(0)

        # Fuse embeddings and prepare memory once
        cond = self.fuse_embeddings(jirvis_emb, nmr_emb)  # (B, d_embeddings)
        mem = self._add_pos_memory(self.make_memory(cond))  # (B, m, d_model)

        # Init sequence with BOS
        seq = torch.full((B, 1), bos_id, dtype=torch.long, device=device)  # (B, 1)

        for _ in range(max_new_tokens):
            T = seq.size(1)

            # Embed current seq (we predict next token)
            x = self.tok_emb(seq)          # (B, T, d_model)
            x = self._add_pos_target(x)    # (B, T, d_model)

            # Masks for current length
            tgt_mask = self._causal_mask(B, T, device)          # (B, T, T)
            src_mask = self._all_false_mask(B, T, self.m, device)  # (B, T, m)

            # Decode and take last-step logits
            logits = self.model(x, tgt_mask, mem, src_mask, normalize_logits=False)  # (B, T, V)
            logits_last = logits[:, -1, :]  # (B, V)

            # Sample next token (nucleus/top-p)
            if temperature <= 0:
                temperature = 1e-6
            probs = F.softmax(logits_last / temperature, dim=-1)

            # Top-p filtering
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = (cumsum > top_p).float().argmax(dim=-1)  # (B,)
            # build a mask that zeros out everything after cutoff for each batch
            arange = torch.arange(probs.size(-1), device=device).unsqueeze(0).expand(B, -1)
            mask = arange > cutoff.unsqueeze(1)
            filtered = sorted_probs.masked_fill(mask, 0.0)
            filtered = filtered / filtered.sum(dim=-1, keepdim=True)
            next_in_sorted = torch.multinomial(filtered, num_samples=1).squeeze(1)  # (B,)
            next_token = sorted_idx.gather(1, next_in_sorted.unsqueeze(1)).squeeze(1)  # (B,)

            # Append
            seq = torch.cat([seq, next_token[:, None]], dim=1)

            # Stop early if everyone hit EOS (optional: could also track per-row)
            if (next_token == eos_id).all():
                break

        return seq
