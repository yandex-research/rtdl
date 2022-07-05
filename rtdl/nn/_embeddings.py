import collections.abc
import itertools
import math
from typing import List, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

from .._utils import INTERNAL_ERROR_MESSAGE
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
        self.weight = Parameter(Tensor(d_embedding))
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
    def __init__(
        self,
        _cardinalities_and_maybe_dimensions: Union[List[int], List[Tuple[int, int]]],
        d_embedding: Optional[int] = None,
        *,
        stack: bool = False,
        bias: bool = False,
    ) -> None:
        spec = _cardinalities_and_maybe_dimensions
        if not spec:
            raise ValueError('The first argument must be non-empty')
        if (d_embedding is None) ^ isinstance(spec[0], int):
            raise ValueError(
                'Invalid arguments. Valid combinations are:'
                ' (1) spec is a list of tuples (cardinality, d_embedding) AND d_embedding is None'
                ' (2) spec is a list of cardinalities AND d_embedding is an integer'
            )
        if stack and d_embedding is None:
            raise ValueError('stack can be True only when d_embedding is not None')

        spec_ = cast(
            List[Tuple[int, int]],
            spec if d_embedding is None else [(x, d_embedding) for x in spec],
        )
        self._embeddings = nn.ModuleList()
        for cardinality, d_embedding in spec_:
            self._embeddings.append(nn.Embedding(cardinality, d_embedding))
        self._biases = (
            nn.ParameterList(Parameter(Tensor(d)) for _, d in spec_) if bias else None
        )
        self.stack = stack
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self._embeddings:
            _initialize_embeddings(module.weight, module.weight.shape[-1])
        if self._biases is not None:
            for x in self._biases:
                _initialize_embeddings(x, x.shape[-1])

    def get_embeddings(self, feature_idx: int) -> Tensor:
        if feature_idx < 0 or feature_idx >= len(self._embeddings):
            raise ValueError(
                f'feature_idx must be in the range(0, {len(self._embeddings)}).'
                f' The provided value is {feature_idx}.'
            )
        x = self._embeddings[feature_idx].weight
        if self._biases is not None:
            x = x + self._biases[feature_idx][None]
        return x

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError('x must have two dimensions')
        if x.shape[1] != len(self._embeddings):
            raise ValueError(
                f'x has {x.shape[1]} columns, but it must have {len(self._embeddings)} columns.'
            )
        out = []
        biases = itertools.repeat(None) if self._biases is None else self._biases
        assert isinstance(biases, collections.abc.Iterable)  # hint for mypy
        for module, bias, column in zip(self._embeddings, biases, x.T):
            x = module(column)
            if bias is not None:
                x = x + bias[None]
            out.append(x)
        return torch.stack(out, 1) if self.stack else torch.cat(out, 1)


class LinearEmbeddings(nn.Module):
    """Linear embeddings for numerical features."""

    def __init__(self, n_features: int, d_embedding: int, bias: bool = True):
        self.weight = Parameter(Tensor(n_features, d_embedding))
        self.bias = Parameter(Tensor(n_features, d_embedding)) if bias else None
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

    def __init__(self, bin_edges: List[Tensor], *, stack: bool) -> None:
        super().__init__()
        self.register_buffer('bin_edges', torch.cat(bin_edges), False)
        self.edge_counts = [len(x) for x in bin_edges]
        self.stack = stack
        self.d_encoding = (
            max(self.edge_counts) - 1
            if self.stack
            else [x - 1 for x in self.edge_counts]
        )

    def forward(self, x: Tensor, indices: Optional[Tensor]) -> Tensor:
        if indices is None:
            # x represents raw values
            bin_edges = self.bin_edges.split(self.edge_counts)
            return compute_piecewise_linear_encoding(x, bin_edges, stack=self.stack)
        else:
            # x represents ratios
            return piecewise_linear_encoding(x, indices, self.d_encoding)


class PeriodicEmbeddings(nn.Module):
    # Source: https://github.com/Yura52/tabular-dl-num-embeddings/blob/e49e95c52f829ad0ab7d653e0776c2a84c03e261/lib/deep.py#L28
    def __init__(self, n_features: int, d_embedding: int, sigma: float) -> None:
        if d_embedding % 2:
            raise ValueError('d_embedding must be even')
        self.sigma = sigma
        self.coefficients = Parameter(Tensor(n_features, d_embedding // 2))
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
        self.weight = Parameter(Tensor(n_tokens, d_in, d_out))
        self.bias = Parameter(Tensor(n_tokens, d_out)) if bias else None
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


def make_lr_embeddings(n_features: int, d_embedding: int) -> nn.Module:
    """The LR embeddings for numerical features from [1].

    This embedding module is easy to use, and it usually provides performance gain over
    the embeddings-free approach (given the same backbone architecture). Depending on a
    task, however, one may try to achieve better performance with more advanced modules.

    This embedding consists of two parts:

    * (L) Linear embeddings (via `LinearEmbeddings`)
    * (R) ReLU activation

    In [1], the following models used these embeddings: MLP-LR, ResNet-LR, Transformer-LR.

    Args:
        n_features: the number of features
        d_embedding: the embedding size
    Returns:
        embeddings: the embedding module

    See also:
        `make_ple_lr_embeddings`
        `make_plr_embeddings`

    References:
        * [1] Yury Gorishniy, Ivan Rubachev, Artem Babenko, "On Embeddings for Numerical Features in Tabular Deep Learning", 2022

    Examples:
        .. testcode::

            batch_size = 2
            n_features = 3
            d_embedding = 4
            x = torch.randn(batch_size, n_features)
            m = make_lr_embeddings(n_features, d_embedding)
            assert m(x).shape == (batch_size, n_features, d_embedding)
    """
    return nn.Sequential(
        LinearEmbeddings(n_features, d_embedding),
        nn.ReLU(),
    )


def make_ple_lr_embeddings(bin_edges: List[Tensor], d_embedding: int) -> nn.Module:
    n_features = len(bin_edges)
    embeddings = PiecewiseLinearEncoder(bin_edges, stack=True)
    assert isinstance(embeddings.d_encoding, int), INTERNAL_ERROR_MESSAGE
    return nn.Sequential(
        embeddings,
        ELinear(embeddings.d_encoding, d_embedding, n_features),
        nn.ReLU(),
    )


def make_plr_embeddings(n_features: int, d_embedding: int, sigma: float) -> nn.Module:
    return nn.Sequential(
        PeriodicEmbeddings(n_features, d_embedding, sigma),
        ELinear(d_embedding, d_embedding, n_features),
        nn.ReLU(),
    )
