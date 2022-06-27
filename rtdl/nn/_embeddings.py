import math
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..data import compute_piecewise_linear_encoding, piecewise_linear_encoding


def _initialize_embeddings(weight: Tensor, d: int) -> None:
    d_sqrt_inv = 1 / math.sqrt(d)
    # This initialization is taken from torch.nn.Linear and is equivalent to:
    # nn.init.kaiming_uniform_(..., a=math.sqrt(5))
    # Also, this initialization was used in the paper "Revisiting Deep Learning Models
    # for Tabular Data".
    nn.init.uniform_(weight, a=-d_sqrt_inv, b=d_sqrt_inv)


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

    def __init__(self, d_embedding: int) -> None:
        """
        Args:
            d_embedding: the size of the embedding
        """
        super().__init__()
        self.weight = nn.Parameter(Tensor(d_embedding))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _initialize_embeddings(self.weight, self.weight.shape[-1])

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
        if x.ndim != 3:
            raise ValueError('The input must have three dimensions')
        return torch.cat([self.expand(len(x), 1), x], dim=1)


class OneHotEncoder(nn.Module):
    cardinalities: Tensor

    def __init__(self, cardinalities: List[int]) -> None:
        self.register_buffer('cardinalities', torch.tensor(cardinalities))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError('The input must have two dimensions')
        encoded_columns = [
            F.one_hot(column, cardinality)
            for column, cardinality in zip(x.T, self.cardinalities)
        ]
        return torch.cat(encoded_columns, 1)


class CatEmbeddings(nn.Module):
    """Embeddings for categorical features."""

    category_offsets: Tensor

    def __init__(self, cardinalities: List[int], d_embedding: int, bias: bool) -> None:
        super().__init__()
        if not cardinalities:
            raise ValueError('cardinalities must be non-empty')
        if d_embedding < 1:
            raise ValueError('d_embedding must be positive')

        category_offsets = torch.tensor([0] + cardinalities[:-1]).cumsum(0)
        self.register_buffer('category_offsets', category_offsets)
        self.embeddings = nn.Embedding(sum(cardinalities), d_embedding)
        self.bias = (
            nn.Parameter(Tensor(len(cardinalities), d_embedding)) if bias else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in [self.embeddings.weight, self.bias]:
            if parameter is not None:
                _initialize_embeddings(parameter, parameter.shape[-1])

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError('The input must have two dimensions')
        x = self.embeddings(x + self.category_offsets[None])
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class LinearEmbeddings(nn.Module):
    """Linear embeddings for numerical features."""

    def __init__(self, n_features: int, d_embedding: int, bias: bool = True):
        self.weight = nn.Parameter(Tensor(n_features, d_embedding))
        self.bias = nn.Parameter(Tensor(n_features, d_embedding)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in [self.weight, self.bias]:
            if parameter is not None:
                _initialize_embeddings(parameter, parameter.shape[-1])

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError('The input must have two dimensions')
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class PiecewiseLinearEncoder(nn.Module):
    bin_edges: Tensor
    d_encoding: Union[int, List[int]]

    def __init__(self, bin_edges: List[Tensor], optimize_shape: bool) -> None:
        super().__init__()
        self.register_buffer('bin_edges', torch.cat(bin_edges), False)
        self.edge_counts = [len(x) for x in bin_edges]
        self.optimize_shape = optimize_shape
        self.d_encoding = (
            [x - 1 for x in self.edge_counts]
            if self.optimize_shape
            else max(self.edge_counts) - 1
        )

    def forward(self, x: Tensor, indices: Optional[Tensor]) -> Tensor:
        if indices is None:
            # x represents raw values
            bin_edges = self.bin_edges.split(self.edge_counts)
            return compute_piecewise_linear_encoding(
                x, bin_edges, optimize_shape=self.optimize_shape
            )
        else:
            # x represents ratios
            return piecewise_linear_encoding(x, indices, self.d_encoding)


class PeriodicEmbeddings(nn.Module):
    # Source: https://github.com/Yura52/tabular-dl-num-embeddings/blob/e49e95c52f829ad0ab7d653e0776c2a84c03e261/lib/deep.py#L28
    def __init__(self, n_features: int, d_embedding: int, sigma: float) -> None:
        if d_embedding % 2:
            raise ValueError('d_embedding must be even')
        self.sigma = sigma
        self.coefficients = nn.Parameter(Tensor(n_features, d_embedding // 2))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.coefficients, 0.0, self.sigma)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError('The input must have two dimensions')
        x = 2 * math.pi * self.coefficients[None] * x[..., None]
        return torch.cat([torch.cos(x), torch.sin(x)], -1)


class ELinear(nn.Module):
    def __init__(self, d_in: int, d_out: int, n_tokens: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(Tensor(n_tokens, d_in, d_out))
        self.bias = nn.Parameter(Tensor(n_tokens, d_out)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        # This initialization is equivalent to that of torch.nn.Linear
        d_in = self.weight.shape[1]
        bound = 1 / math.sqrt(d_in)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                'The input must have three dimensions (batch_size, n_tokens, d_embedding)'
            )
        x = x[..., None] * self.weight[None]
        x = x.sum(-2)
        if self.bias is not None:
            x = x + self.bias[None]
        return x
