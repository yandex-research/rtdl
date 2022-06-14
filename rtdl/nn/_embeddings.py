import math

import torch
import torch.nn as nn
from torch import Tensor


def _initialize_embeddings(weight: Tensor, initialization: str, d: int) -> None:
    d_sqrt_inv = 1 / math.sqrt(d)
    if initialization == 'uniform':
        # used in the paper "Revisiting Deep Learning Models for Tabular Data";
        # is equivalent to `nn.init.kaiming_uniform_(x, a=math.sqrt(5))` (which is
        # used by torch to initialize nn.Linear.weight, for example)
        nn.init.uniform_(weight, a=-d_sqrt_inv, b=d_sqrt_inv)
    elif initialization == 'normal':
        nn.init.normal_(weight, std=d_sqrt_inv)
    else:
        raise ValueError('initialization must be one of: ["uniform", "normal"]')


class CLSEmbedding(nn.Module):
    """Embedding of the [CLS]-token for BERT-like inference.

    To learn about the [CLS]-based inference, see [devlin2018bert].

    When used as a module, the [CLS]-embedding is appended **to the beginning** of each
    item in the batch.

    Examples:
        .. testcode::

            batch_size = 2
            n_tokens = 3
            d = 4
            cls_embedding = CLSEmbedding(d, 'uniform')
            x = torch.randn(batch_size, n_tokens, d)
            x = cls_embedding(x)
            assert x.shape == (batch_size, n_tokens + 1, d)
            assert (x[:, 0, :] == cls_embedding.expand(len(x))).all()

    References:
        * [devlin2018bert] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" 2018
    """

    def __init__(self, d: int, initialization: str) -> None:
        """
        Args:
            d: the size of the embedding
            initialization: initialization policy for parameters. Must be one of
                :code:`['uniform', 'normal']`. Let :code:`s = d ** -0.5`. Then, the
                corresponding distributions are :code:`Uniform(-s, s)` and
                :code:`Normal(0, s)`. `'uniform'` is a good starting point.
        """
        super().__init__()
        self.weight = nn.Parameter(Tensor(d))
        _initialize_embeddings(self.weight, initialization, d)

    def expand(self, *d_leading: int) -> Tensor:
        """Repeat the [CLS]-embedding (e.g. to make a batch).

        Namely, this::

            cls_batch = cls_embedding.expand(d1, d2, ..., dN)

        is equivalent to this::

            new_dimensions = (1,) * N
            cls_batch = cls_embedding.weight.view(*new_dimensions, -1).expand(
                d1, d2, ..., dN, len(cls_embedding.weight)
            )

        Examples:
            .. testcode::

                batch_size = 2
                n_tokens = 3
                d = 4
                x = torch.randn(batch_size, n_tokens, d)
                cls_embedding = CLSEmbedding(d, 'uniform')
                assert cls_embedding.expand(len(x)).shape == (len(x), d)
                assert cls_embedding.expand(len(x), 1).shape == (len(x), 1, d)

        Note:
            Under the hood, the `torch.Tensor.expand` method is applied to the
            underlying :code:`weight` parameter, so gradients will be propagated as
            expected.

        Args:
            d_leading: the additional new dimensions

        Returns:
            tensor of the shape :code:`(*d_leading, len(self.weight))`
        """
        if not d_leading:
            return self.weight
        new_dims = (1,) * (len(d_leading) - 1)
        return self.weight.view(*new_dims, -1).expand(*d_leading, -1)

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([self.expand(len(x), 1), x], dim=1)
