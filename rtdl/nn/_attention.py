import math
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .._utils import INTERNAL_ERROR_MESSAGE, all_or_none


class MultiheadAttention(nn.Module):
    """Multihead Attention (self-/cross-) with optional 'linear' (fast) attention.

    To learn more about Multihead Attention, see [1]. See the implementation
    of `Transformer` and the examples below to learn how to use the compression technique
    from [2] to speed up the module when the number of tokens is large.

    Examples:
        .. testcode::

            batch_size, n_tokens, d = 2, 3, 12
            n_heads = 6
            a = torch.randn(batch_size, n_tokens, d)
            b = torch.randn(batch_size, n_tokens * 2, d)
            module = MultiheadAttention(d_embedding=d, n_heads=n_heads, dropout=0.2)

            # self-attention
            x, attention_stats = module(a, a)
            assert x.shape == a.shape
            assert attention_stats['attention_probs'].shape == (batch_size, n_heads, n_tokens, n_tokens)
            assert attention_stats['attention_logits'].shape == (batch_size, n_heads, n_tokens, n_tokens)

            # cross-attention
            module(a, b)

            # Linformer (fast) self-attention
            module = MultiheadAttention(
                **kwargs,
                linformer_compression_ratio=0.25,
                linformer_sharing_policy='headwise',
                n_tokens=n_tokens,
            )
            module(a, a)

    References:
        * [1] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" 2018
        * [2] Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma "Linformer: Self-Attention with Linear Complexity", 2020
    """

    def __init__(
        self,
        *,
        d_embedding: int,
        n_heads: int,
        dropout: float,
        d_key: Optional[int] = None,
        d_value: Optional[int] = None,
        share_key_query_projection: bool = False,
        bias: bool = True,
        initialization: str = 'kaiming',
        linformer_compression_ratio: Optional[float] = None,
        linformer_sharing_policy: Optional[str] = None,
        n_tokens: Optional[int] = None,
    ) -> None:
        """
        Args:
            d_embedding: the input embedding size. Must be a multiple of ``n_heads```
            n_heads: the number of heads. If greater than 1, then the module will have
                an addition output layer (so called "mixing" layer).
            dropout: dropout rate for the attention map. The dropout is applied to
                *probabilities* and do not affect logits.
            d_key: the key projection size. Must be a multiple of ``n_heads``. If `None`,
                then ``d_embedding`` is used instead.
            d_value: the value (output) projection size. Must be a multiple of ``n_heads``.
                  If `None`, then ``d_embedding`` is used instead.
            share_key_query_projection: if `True`, then the projections for keys and
                queries are shared.
            bias: if `True`, then input (and output, if presented) layers also have bias.
                `True` is a reasonable default choice.
            initialization: initialization for input projection layers. Must be one of
                ``['kaiming', 'xavier']``. ``'kaiming'`` is a reasonable default choice.
            linformer_compression_ratio: apply the technique from [1] to speed
                up the attention operation when the number of tokens is large. Can
                actually slow things down if the number of tokens is too low. This
                option can affect task metrics in an unpredictable way, use it with caution.
            linformer_sharing_policy: weight sharing policy for the Linformer compression.
                Must be `None` if ``linformer_compression_ratio`` is None. Otherwise,
                must either ``'headwise'`` or ``'key-value'`` (both policies are
                described in [1]). The first one leads to more parameters. The effect
                on the task performance depends on the task.
            n_tokens: the number of tokens (features). Must be provided if
                ``linformer_compression_ratio`` is not `None`.
        Raises:
            ValueError: if input arguments are not valid.
        """
        super().__init__()
        if d_key is None:
            d_key = d_embedding
        if d_value is None:
            d_value = d_embedding
        if n_heads > 1 and any(d % n_heads != 0 for d in [d_embedding, d_key, d_value]):
            raise ValueError(
                'd_embedding, d_key and d_value must be multiples of n_heads'
            )
        if initialization not in ['kaiming', 'xavier']:
            raise ValueError('initialization must be "kaiming" or "xavier"')
        if not all_or_none(
            [n_tokens, linformer_compression_ratio, linformer_sharing_policy]
        ):
            raise ValueError(
                'The arguments n_tokens, linformer_compression_ratio and'
                ' linformer_sharing_policy must be either all None or all not-None'
            )
        linformer_sharing_policy_valid_values = [None, 'headwise', 'key-value']
        if linformer_sharing_policy not in linformer_sharing_policy_valid_values:
            raise ValueError(
                f'linformer_sharing_policy must be one of: {linformer_sharing_policy_valid_values}'
            )

        self.W_k = nn.Linear(d_embedding, d_key, bias)
        self.W_q = (
            None if share_key_query_projection else nn.Linear(d_embedding, d_key, bias)
        )
        self.W_v = nn.Linear(d_embedding, d_value, bias)
        self.W_out = nn.Linear(d_value, d_value, bias) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None
        self._initialization = initialization
        # the following modules enables hook-based introspection
        self.logits_handler = nn.Identity()
        self.probs_handler = nn.Identity()

        def make_linformer_compression():
            assert (
                n_tokens and linformer_compression_ratio
            ), INTERNAL_ERROR_MESSAGE  # for mypy
            # https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/examples/linformer/linformer_src/modules/multihead_linear_attention.py#L83
            return nn.Linear(
                n_tokens, int(n_tokens * linformer_compression_ratio), bias=False
            )

        if linformer_compression_ratio is not None:
            self.linformer_key_compression = make_linformer_compression()
            self.linformer_value_compression = (
                None
                if linformer_sharing_policy == 'key-value'
                else make_linformer_compression()
            )
        else:
            self.linformer_key_compression = None
            self.linformer_value_compression = None

        self.reset_parameters()

    def reset_parameters(self):
        for m in [self.W_q, self.W_k, self.W_v]:
            if m is None:
                continue
            # the "xavier" branch tries to follow torch.nn.MultiheadAttention;
            # the second condition checks if W_v plays the role of W_out; the latter one
            # is initialized with Kaiming in torch
            if self._initialization == 'xavier' and (
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

    def forward(self, x_q: Tensor, x_kv: Tensor) -> Tensor:
        """Perform the forward pass.

        Args:
            x_q: query token embeddings. Shape: ``(batch_size, n_q_tokens, d_embedding)``.
            x_kv: key-value token embeddings. Shape: ``(batch_size, n_kv_tokens, d_embedding)``.
        Returns:
            (new_token_embeddings, attention_stats)
        """
        W_q = self.W_k if self.W_q is None else self.W_q
        q, k, v = W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0, INTERNAL_ERROR_MESSAGE

        if self.linformer_key_compression is not None:
            k = self.linformer_key_compression(k.transpose(1, 2)).transpose(1, 2)
            value_compression = (
                self.linformer_key_compression
                if self.linformer_value_compression is None
                else self.linformer_value_compression
            )
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]
        n_k_tokens = k.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention_logits = q @ k.transpose(1, 2) / math.sqrt(d_head_key)
        attention_probs = F.softmax(attention_logits, dim=-1)

        _attention_shape = (batch_size, self.n_heads, n_k_tokens, n_q_tokens)
        _ = self.logits_handler(attention_logits.reshape(*_attention_shape))
        _ = self.probs_handler(attention_probs.reshape(*_attention_shape))

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

        return x
