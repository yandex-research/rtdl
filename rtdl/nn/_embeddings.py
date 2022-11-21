import collections.abc
import itertools
import math
from typing import List, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

from .._utils import INTERNAL_ERROR_MESSAGE, experimental
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

    In the forward pass, the module appends [CLS]-embedding **to the beginning** of each
    item in the batch.

    Examples:
        .. testcode::

            batch_size = 2
            n_tokens = 3
            d = 4
            cls_embedding = CLSEmbedding(d)
            x = torch.randn(batch_size, n_tokens, d)
            x = cls_embedding(x)
            assert x.shape == (batch_size, n_tokens + 1, d)
            assert (x[:, 0, :] == cls_embedding.weight.expand(len(x), -1)).all()

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

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError('The input must have three dimensions')
        if x.shape[-1] != len(self.weight):
            raise ValueError(
                'The last dimension of x must be equal to the embedding size'
            )
        return torch.cat([self.weight.expand(len(x), 1, -1), x], dim=1)


class OneHotEncoder(nn.Module):
    """One hot encoding for categorical features.

    * **Input shape**: ``(batch_size, n_categorical_features)``
    * **Input data type**: ``integer``

    Examples::

        # three categorical features
        cardinalities = [3, 4, 5]
        ohe = OneHotEncoder(cardinalities)
        batch_size = 2
        x_cat = torch.stack([torch.randint(0, c, (batch_size,)) for c in cardinalities], 1)
        assert ohe(x_cat).shape == (batch_size, sum(cardinalities))
        assert (x_cat.sum(1) == len(cardinalities)).all()
    """

    cardinalities: Tensor

    def __init__(self, cardinalities: List[int]) -> None:
        """
        Args:
            cardinalities: ``cardinalities[i]`` is the number of unique values for the
                i-th categorical feature.
        """
        super().__init__()
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
    """Embeddings for categorical features.

    * **Input shape**: ``(batch_size, n_categorical_features)``
    * **Input data type**: ``integer``

    To obtain embeddings for the i-th feature, use `get_embeddings`.

    Examples:
        .. testcode::

            # three categorical features
            cardinalities = [3, 4, 5]
            embedding_sizes = [6, 7, 8]
            m_cat = CatEmbeddings(list(zip(cardinalities, embedding_sizes)))
            batch_size = 2
            x_cat = torch.stack([
                torch.randint(0, c, (batch_size,))
                for c in cardinalities
            ], 1)
            assert m_cat(x_cat).shape == (batch_size, sum(embedding_sizes))
            i = 1
            assert m_cat.get_embeddings(i).shape == (cardinalities[i], embedding_sizes[i])

            d_embedding = 9
            m_cat = CatEmbeddings(cardinalities, d_embedding, stack=True)
            m_cat(x_cat).shape == (batch_size, len(cardinalities), d_embedding)
    """

    def __init__(
        self,
        _cardinalities_and_maybe_dimensions: Union[List[int], List[Tuple[int, int]]],
        d_embedding: Optional[int] = None,
        *,
        stack: bool = False,
        bias: bool = False,
    ) -> None:
        """
        Args:
            _cardinalities_and_maybe_dimensions: (positional-only argument!) either a
                list of cardinalities or a list of ``(cardinality, embedding_size)`` pairs.
            d_embedding: if not `None`, then (1) the first argument must be a list of
                cardinalities, (2) all the features will have the same embedding size,
                (3) ``stack=True`` becomes allowed.
            stack: if `True`, then ``d_embedding`` must be provided, and the module will
                produce outputs of the shape ``(batch_size, n_cat_features, d_embedding)``.
            bias: this argument is presented for historical reasons, just keep it `False`
                (when it is `True`, then for a each feature one more trainable vector is
                allocated, and it is added to the main embedding regardless of the
                feature values).
        """
        spec = _cardinalities_and_maybe_dimensions
        if not spec:
            raise ValueError('The first argument must be non-empty')
        if not (
            (isinstance(spec[0], tuple) and d_embedding is None)
            or (isinstance(spec[0], int) and d_embedding is not None)
        ):
            raise ValueError(
                'Invalid arguments. Valid combinations are:'
                ' (1) the first argument is a list of (cardinality, embedding)-tuples AND d_embedding is None'
                ' (2) the first argument is a list of cardinalities AND d_embedding is an integer'
            )
        if stack and d_embedding is None:
            raise ValueError('stack can be True only when d_embedding is not None')

        super().__init__()
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
        """Get embeddings for the i-th feature.

        This method is needed because of the ``bias`` option (when it is set to `True`,
        the embeddings provided by the underlying `torch.nn.Embedding` are not "complete").

        Args:
            feature_idx: the feature index
        Return:
            embeddings for the feature ``feature_idx``
        Raises:
            ValueError: for invalid inputs
        """
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
    """Linear embeddings for numerical features.

    * **Input shape**: ``(batch_size, n_features)``
    * **Output shape**: ``(batch_size, n_features, d_embedding)``

    For each feature, a separate linear layer is allocated (``n_features`` layers in total).
    One such layer can be represented as ``torch.nn.Linear(1, d_embedding)``

    The embedding process is illustrated in the following pseudocode::

        layers = [nn.Linear(1, d_embedding) for _ in range(n_features)]
        x = torch.randn(batch_size, n_features)
        x_embeddings = torch.stack(
            [layers[i](x[:, i:i+1]) for i in range(n_features)],
            1,
        )

    Examples:
        .. testcode::

            batch_size = 2
            n_features = 3
            d_embedding = 4
            x = torch.randn(batch_size, n_features)
            m = LinearEmbeddings(n_features, d_embedding)
            assert m(x).shape == (batch_size, n_features, d_embedding)
    """

    def __init__(self, n_features: int, d_embedding: int, bias: bool = True):
        super().__init__()
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
    """Piecewise linear encoding for numerical features described in [1].

    See `rtdl.nn.compute_piecewise_linear_encoding` for details.

    Examples:
        .. testcode::

            train_size = 100
            n_features = 4
            X = torch.randn(train_size, n_features)
            n_bins = 3
            bin_edges = compute_quantile_bin_edges(X, n_bins)
            bin_counts = [len(x) - 1 for x in bin_edges]
            batch_size = 3
            x = X[:batch_size]

            m_ple = PiecewiseLinearEncoder(bin_edges, stack=False)
            assert m_ple(x).shape == (n_objects, sum(bin_counts))

            m_ple = PiecewiseLinearEncoder(bin_edges, stack=True)
            assert m_ple(x).shape == (n_objects, n_features, max(bin_counts))

            x_bin_indices = compute_bin_indices(x, bin_edges)
            x_bin_ratios = compute_bin_linear_ratios(x, x_bin_indices, bin_edges)
            m_ple = PiecewiseLinearEncoder(
                bin_edges, stack=True, expect_ratios_and_indices=True
            )
            assert m_ple(x_bin_ratios, x_bin_indices).shape == (
                n_objects, n_features, max(bin_counts)
            )

    References:
        * [1] Yury Gorishniy, Ivan Rubachev, Artem Babenko, "On Embeddings for Numerical Features in Tabular Deep Learning", 2022
    """

    bin_edges: Tensor
    d_encoding: Union[int, List[int]]

    def __init__(
        self,
        bin_edges: List[Tensor],
        *,
        stack: bool,
        expect_ratios_and_indices: bool = False,
    ) -> None:
        """
        Args:
            bin_edges: the bin edges. Can be obtained via
                `rtdl.data.compute_quantile_bin_edges` or `rtdl.data.compute_decision_tree_bin_edges`
            stack: the argument for `rtdl.data.compute_piecewise_linear_encoding`.
            expect_ratios_and_indices: if `True`, then the module will expect two arguments
                in its forward pass: bin ratios (produced by `rtdl.data.compute_bin_linear_ratios`)
                and indices (produced by `rtdl.data.compute_bin_indices`). Otherwise,
                the modules will expect one argument (raw numerical feature values).
                This option can be usefull if computing ratios and indices on-the-fly
                is a bottleneck and you want to use precomputed values.
        """
        super().__init__()
        self.register_buffer('bin_edges', torch.cat(bin_edges), False)
        self.edge_counts = [len(x) for x in bin_edges]
        self.stack = stack
        self.d_encoding = (
            max(self.edge_counts) - 1
            if self.stack
            else [x - 1 for x in self.edge_counts]
        )
        self.expect_ratios_and_indices = expect_ratios_and_indices

    def forward(self, x: Tensor, indices: Optional[Tensor] = None) -> Tensor:
        bin_edges = self.bin_edges.split(self.edge_counts)
        if indices is None:
            if self.expect_ratios_and_indices:
                raise ValueError(
                    'The module expects two arguments (ratios and indices),'
                    ' because the argument expect_ratios_and_indices was set to `True` in the constructor'
                )
            # x represents raw values
            return compute_piecewise_linear_encoding(x, bin_edges, stack=self.stack)
        else:
            if not self.expect_ratios_and_indices:
                raise ValueError(
                    'The module expects one arguments (raw numerical feature values),'
                    ' because the argument expect_ratios_and_indices was set to `False` in the constructor'
                )
            ratios = x
            return piecewise_linear_encoding(
                bin_edges, ratios, indices, self.d_encoding, stack=self.stack
            )


class PeriodicEmbeddings(nn.Module):
    """Periodic embeddings for numerical features described in [1].

    Warning:
        For better performance and to avoid some failure modes, it is recommended
        to insert `NLinear` after this module (even if the next module after that is the
        first linear layer of the model's backbone). Alternatively, you can use
        `make_plr_embeddings`.

    Examples:
        .. testcode::

            batch_size = 2
            n_features = 3
            d_embedding = 4
            x = torch.randn(batch_size, n_features)
            sigma = 0.1  # THIS HYPERPARAMETER MUST BE TUNED CAREFULLY
            m = PeriodicEmbeddings(n_features, d_embedding, sigma)
            # for better performance: m = nn.Sequantial(PeriodicEmbeddings(...), NLinear(...))
            assert m(x).shape == (batch_size, n_features, d_embedding)

    References:
        * [1] Yury Gorishniy, Ivan Rubachev, Artem Babenko, "On Embeddings for Numerical Features in Tabular Deep Learning", 2022
    """

    # Source: https://github.com/Yura52/tabular-dl-num-embeddings/blob/e49e95c52f829ad0ab7d653e0776c2a84c03e261/lib/deep.py#L28
    def __init__(self, n_features: int, d_embedding: int, sigma: float) -> None:
        """
        Args:
            n_features: the number of numerical features
            d_embedding: the embedding size, must be an even positive integer.
            sigma: the scale of the weight initialization.
                **This is a super important parameter which significantly affects performance**.
                Its optimal value can be dramatically different for different datasets, so
                no "default value" can exist for this parameter, and it must be tuned for
                each dataset. In the original paper, during hyperparameter tuning, this
                parameter was sampled from the distribution ``LogUniform[1e-2, 1e2]``.
                A similar grid would be ``[1e-2, 1e-1, 1e0, 1e1, 1e2]``.
                If possible, add more intermidiate values to this grid.
        """
        if d_embedding % 2:
            raise ValueError('d_embedding must be even')

        super().__init__()
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


class NLinear(nn.Module):
    """N linear layers for N token (feature) embeddings.

    To understand this module, let's revise `torch.nn.Linear`. When `torch.nn.Linear` is
    applied to three-dimensional inputs of the shape
    ``(batch_size, n_tokens, d_embedding)``, then the same linear transformation is
    applied to each of ``n_tokens`` token (feature) embeddings.

    By contrast, `NLinear` allocates one linear layer per token (``n_tokens`` layers in total).
    One such layer can be represented as ``torch.nn.Linear(d_in, d_out)``.
    So, the i-th linear transformation is applied to the i-th token embedding, as
    illustrated in the following pseudocode::

        layers = [nn.Linear(d_in, d_out) for _ in range(n_tokens)]
        x = torch.randn(batch_size, n_tokens, d_in)
        result = torch.stack([layers[i](x[:, i]) for i in range(n_tokens)], 1)

    Examples:
        .. testcode::

            batch_size = 2
            n_features = 3
            d_embedding_in = 4
            d_embedding_out = 5
            x = torch.randn(batch_size, n_features, d_embedding_in)
            m = NLinear(n_features, d_embedding_in, d_embedding_out)
            assert m(x).shape == (batch_size, n_features, d_embedding_out)
    """

    def __init__(self, n_tokens: int, d_in: int, d_out: int, bias: bool = True) -> None:
        """
        Args:
            n_tokens: the number of tokens (features)
            d_in: the input dimension
            d_out: the output dimension
            bias: indicates if the underlying linear layers have biases
        """
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


@experimental
def make_lr_embeddings(n_features: int, d_embedding: int) -> nn.Module:
    """**[EXPERIMENTAL]** The LR embeddings for numerical features described in [1].

    This embedding module is easy to use, and it usually provides performance gain over
    the embeddings-free approach (given the same backbone architecture). Depending on a
    task, however, one may try to achieve better performance with more advanced embedding modules.

    This embedding sequantially applies two transformations:

    * (L) Linear transformation (`LinearEmbeddings`)
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


@experimental
def make_ple_lr_embeddings(bin_edges: List[Tensor], d_embedding: int) -> nn.Module:
    """**[EXPERIMENTAL]** The PLE-LR embeddings for numerical features described in [1].

    Specifically, the T-LR and Q-LR embeddings were described in the paper, both of which
    are special cases of the PLE-LR embeddings. The hyphen in all the names highlights
    the fact that the bin edges of the piecewise linear encoding are not trained
    end-to-end with the rest of the model.

    This embedding sequantially applies three transformations:

    * (PLE) Piecewise linear encoding (`PiecewiseLinearEncoder`)
    * (L) Linear transformation (`NLinear`)
    * (R) ReLU activation

    In [1], the following models used these embeddings:

        * MLP-Q-LR, ResNet-Q-LR, Transformer-Q-LR
        * MLP-T-LR, ResNet-T-LR, Transformer-T-LR

    Args:
        bin_edges: the bin edges for `PiecewiseLinearEncoder` (the size of this list
            is interpreted as the number of features).
        d_embedding: the embedding size.
    Returns:
        embeddings: the embedding module

    See also:
        `make_lr_embeddings`
        `make_plr_embeddings`

    References:
        * [1] Yury Gorishniy, Ivan Rubachev, Artem Babenko, "On Embeddings for Numerical Features in Tabular Deep Learning", 2022

    Examples:
        .. testcode::

            train_size = 100
            n_features = 4
            X = torch.randn(train_size, n_features)
            n_bins = 3
            bin_edges = compute_quantile_bin_edges(X, n_bins)  # Q-LR
            # bin_edges = compute_decision_tree_bin_edges(X, n_bins, ...)  # T-LR

            batch_size = 2
            n_features = 3
            d_embedding = 4
            x = X[:batch_size]
            m = make_ple_lr_embeddings(bin_edges, d_embedding)
            assert m(x).shape == (batch_size, n_features, d_embedding)
    """
    n_features = len(bin_edges)
    embeddings = PiecewiseLinearEncoder(bin_edges, stack=True)
    assert isinstance(embeddings.d_encoding, int), INTERNAL_ERROR_MESSAGE
    return nn.Sequential(
        embeddings,
        NLinear(n_features, embeddings.d_encoding, d_embedding),
        nn.ReLU(),
    )


@experimental
def make_plr_embeddings(
    n_features: int, d_embedding: int, d_periodic_embedding: int, sigma: float
) -> nn.Module:
    """**[EXPERIMENTAL]** The PLR embeddings for numerical features described in [1].

    This embedding sequantially applies three transformations:

    * (P) `Periodic`
    * (L) Linear transformation (`NLinear`)
    * (R) ReLU activation

    In [1], the following models used these embeddings: MLP-PLR, ResNet-PLR, Transformer-PLR

    Args:
        n_features: the number of features
        d_embedding: the embedding size
        d_periodic_embedding: the embedding size produced by the `Periodic` module
        sigma: the **super important** paramerer for `Periodic`
    Returns:
        embeddings: the embedding module

    See also:
        `make_lr_embeddings`
        `make_ple_lr_embeddings`

    References:
        * [1] Yury Gorishniy, Ivan Rubachev, Artem Babenko, "On Embeddings for Numerical Features in Tabular Deep Learning", 2022

    Examples:
        .. testcode::

            batch_size = 2
            n_features = 3
            d_periodic_embedding = 4
            sigma = 0.1  # THIS HYPERPARAMETER MUST BE TUNED CAREFULLY
            d_embedding = 6
            x = torch.randn(batch_size, n_features)
            m = make_plr_embeddings(n_features, d_embedding, d_periodic_embedding, sigma)
            assert m(x).shape == (batch_size, n_features, d_embedding)
    """
    return nn.Sequential(
        PeriodicEmbeddings(n_features, d_periodic_embedding, sigma),
        NLinear(n_features, d_periodic_embedding, d_embedding),
        nn.ReLU(),
    )
