import math
from typing import Dict, Optional, Tuple

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .._utils import INTERNAL_ERROR_MESSAGE, _all_or_none


class MultiheadAttention(nn.Module):
    """Multihead Attention (self-/cross-) with optional 'linear' (fast) attention.

    To learn more about Multihead Attention, see [devlin2018bert]. See the implementation
    of `Transformer` and the examples below to learn how to use the compression technique
    from [wang2020linformer] to speed up the module when the number of tokens is large.

    Examples:
        .. testcode::

            n_objects, n_tokens, d_embedding = 2, 3, 12
            n_heads = 6
            a = torch.randn(n_objects, n_tokens, d_embedding)
            b = torch.randn(n_objects, n_tokens * 2, d_embedding)
            module = MultiheadAttention(
                d_embedding=d_embedding, n_heads=n_heads, dropout=0.2, bias=True, initialization='kaiming'
            )

            # self-attention
            x, attention_stats = module(a, a, None, None)
            assert x.shape == a.shape
            assert attention_stats['attention_probs'].shape == (n_objects * n_heads, n_tokens, n_tokens)
            assert attention_stats['attention_logits'].shape == (n_objects * n_heads, n_tokens, n_tokens)

            # cross-attention
            assert module(a, b, None, None)

            # Linformer self-attention with the 'headwise' sharing policy
            k_compression = torch.nn.Linear(n_tokens, n_tokens // 4)
            v_compression = torch.nn.Linear(n_tokens, n_tokens // 4)
            assert module(a, a, k_compression, v_compression)

            # Linformer self-attention with the 'key-value' sharing policy
            kv_compression = torch.nn.Linear(n_tokens, n_tokens // 4)
            assert module(a, a, kv_compression, kv_compression)

    References:
        * [devlin2018bert] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" 2018
        * [wang2020linformer] Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma "Linformer: Self-Attention with Linear Complexity", 2020
    """

    def __init__(
        self,
        *,
        d_embedding: int,
        n_heads: int,
        dropout: float,
        bias: bool,
        initialization: str,
    ) -> None:
        """
        Args:
            d_embedding: the embedding size. Must be a multiple of :code:`n_heads`.
            n_heads: the number of heads. If greater than 1, then the module will have
                an addition output layer (so called "mixing" layer).
            dropout: dropout rate for the attention map. The dropout is applied to
                *probabilities* and do not affect logits.
            bias: if `True`, then input (and output, if presented) layers also have bias.
                `True` is a reasonable default choice.
            initialization: initialization for input projection layers. Must be one of
                :code:`['kaiming', 'xavier']`. `kaiming` is a reasonable default choice.
        Raises:
            AssertionError: if requirements for the inputs are not met.
        """
        super().__init__()
        if n_heads > 1:
            assert (
                d_embedding % n_heads == 0
            ), 'd_embedding must be a multiple of n_heads'
        assert initialization in ['kaiming', 'xavier']

        self.W_q = nn.Linear(d_embedding, d_embedding, bias)
        self.W_k = nn.Linear(d_embedding, d_embedding, bias)
        self.W_v = nn.Linear(d_embedding, d_embedding, bias)
        self.W_out = nn.Linear(d_embedding, d_embedding, bias) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            # the "xavier" branch tries to follow torch.nn.MultiheadAttention;
            # the second condition checks if W_v plays the role of W_out; the latter one
            # is initialized with Kaiming in torch
            if initialization == 'xavier' and (
                m is not self.W_v or self.W_out is not None
            ):
                # gain is needed since W_qkv is represented with 3 separate layers (it
                # implies different fan_out)
                nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.W_out is not None:
            nn.init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        key_compression: Optional[nn.Linear],
        value_compression: Optional[nn.Linear],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Perform the forward pass.

        Args:
            x_q: query token embeddings
            x_kv: key-value token embeddings
            key_compression: Linformer-style compression for keys
            value_compression: Linformer-style compression for values
        Returns:
            (new_token_embeddings, attention_stats)
        """
        assert _all_or_none(
            [key_compression, value_compression]
        ), 'key_compression and value_compression must be both None or both not-None.'
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0, INTERNAL_ERROR_MESSAGE
        if key_compression is not None:
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)  # type: ignore

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention_logits = q @ k.transpose(1, 2) / math.sqrt(d_head_key)
        attention_probs = F.softmax(attention_logits, dim=-1)
        if self.dropout is not None:
            attention_probs = self.dropout(attention_probs)
        x = attention_probs @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x, {
            'attention_logits': attention_logits,
            'attention_probs': attention_probs,
        }
