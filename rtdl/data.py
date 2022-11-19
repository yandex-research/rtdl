"""Tools for data (pre)processing."""

__all__ = [
    'compute_quantile_bin_edges',
    'compute_decision_tree_bin_edges',
    'compute_bin_indices',
    'compute_bin_linear_ratios',
    'piecewise_linear_encoding',
    'PiecewiseLinearEncoder',
    'get_category_sizes',
    'NoisyQuantileTransformer',
]

import math
import warnings
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast, overload

import numpy as np
import scipy.sparse
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_random_state
from torch import Tensor, as_tensor
from typing_extensions import Literal

from ._utils import experimental

Number = TypeVar('Number', int, float)


def _adjust_bin_counts(X: Union[Tensor, np.ndarray], n_bins: int) -> List[int]:
    if n_bins < 2:
        raise ValueError('n_bins must be greater than 1')
    unique_fn = torch.unique if isinstance(X, Tensor) else np.unique
    adjusted_bin_counts = []
    for i, column in enumerate(X.T):
        n_unique = len(unique_fn(column))
        if n_unique < 2:
            raise ValueError(f'All elements in the column {i} are the same')
        if n_unique < n_bins:
            warnings.warn(
                f'For the feature {i}, the number of bins will be set to the number of'
                ' distinct values, becuase the provided n_bins is greater than this number.'
            )
        adjusted_bin_counts.append(min(n_bins, n_unique))
    return adjusted_bin_counts


@overload
def compute_quantile_bin_edges(X: Tensor, n_bins: int) -> List[Tensor]:
    ...


@overload
def compute_quantile_bin_edges(X: np.ndarray, n_bins: int) -> List[np.ndarray]:
    ...


def compute_quantile_bin_edges(X, n_bins: int):
    """Compute bin edges using decision trees as described in [1].

    The output of this function can be passed as input to:

        * `compute_bin_indices`
        * `compute_bin_linear_ratios`
        * `compute_piecewise_linear_encoding`
        * `PiecewiseLinearEncoder`
        * `rtdl.nn.PiecewiseLinearEncoder`

    For each column, the bin edges are computed as ``n_bins + 1`` quantiles (including 0.0 and 1.0).

    Args:
        X: the feature matrix. Shape: ``(n_objects, n_features)``.
        n_bins: the number of bins to compute
    Returns:
        bin edges: a list of size ``n_features``;
            the i-th entry contains the bin edges for i-th feature. The edges are returned
            in sorted order with duplicates removed (i.e. for a feature with less then
            ``n_bins`` unique values, the number of edges will be less than ``n_bins + 1``).
    Raises:
        ValueError: for invalid inputs

    References:
        * [1] Yury Gorishniy, Ivan Rubachev, Artem Babenko, "On Embeddings for Numerical Features in Tabular Deep Learning", 2022

    Examples:
        .. testcode::

            n_objects = 100
            n_features = 4
            X = torch.randn(n_objects, n_features)
            n_bins = 3
            bin_edges = compute_quantile_bin_edges(X, n_bins)
            assert len(bin_edges) == n_features
            for x in bin_edges:
                assert len(x) == n_bins + 1
    """
    is_torch = isinstance(X, Tensor)
    X = as_tensor(X)

    if X.ndim != 2:
        raise ValueError('X must have two dimensions')
    adjusted_bin_counts = _adjust_bin_counts(X, n_bins)
    edges = []
    for column, adjusted_n_bins in zip(X.T, adjusted_bin_counts):
        # quantiles include 0.0 and 1.0
        quantiles = torch.linspace(0.0, 1.0, adjusted_n_bins + 1).to(column)
        edges.append(torch.sort(torch.unique(torch.quantile(column, quantiles))).values)
    return edges if is_torch else [x.numpy() for x in edges]


@overload
def compute_decision_tree_bin_edges(
    X: Tensor,
    n_bins: int,
    *,
    y: Tensor,
    regression: bool,
    tree_kwargs: Dict[str, Any],
) -> List[Tensor]:
    ...


@overload
def compute_decision_tree_bin_edges(
    X: np.ndarray,
    n_bins: int,
    *,
    y: np.ndarray,
    regression: bool,
    tree_kwargs: Dict[str, Any],
) -> List[np.ndarray]:
    ...


def compute_decision_tree_bin_edges(
    X,
    n_bins: int,
    *,
    y,
    regression: bool,
    tree_kwargs: Dict[str, Any],
):
    """Compute bin edges using decision trees as described in [1].

    The output of this function can be passed as input to:

        * `compute_bin_indices`
        * `compute_bin_linear_ratios`
        * `compute_piecewise_linear_encoding`
        * `PiecewiseLinearEncoder`
        * `rtdl.nn.PiecewiseLinearEncoder`

    For each column, a decision tree is built, which uses for growing only this one
    feature and the provided target. The regions corresponding to the leaves form
    the bin edges (see the illustration below). Additionally, the leftmost and the
    rightmost bin edges are computed as the minimum and maximum values, respectively.

    .. image:: ../images/decision_tree_bins.png
        :scale: 25%
        :alt: obtaining bins from decision trees

    Warning:
        This function performs the computations in terms of numpy arrays. For
        PyTorch-based inputs located on non-CPU devices, data transfer happens.

    Args:
        X: the feature matrix. Shape: ``(n_objects, n_features)``.
        n_bins: the number of bins to compute
        y: the classification or regression target for building the decision trees
        regression: if True, `sklearn.tree.DecisionTreeRegressor` is used for building trees.
            otherwise, `sklearn.tree.DecisionTreeClassifier` is used.
        tree_kwargs: keyword arguments for the corresponding Scikit-learn decision tree class.
            It must not contain "max_leaf_nodes", since this parameter is set to be ``n_bins``.
    Returns:
        bin edges: a list of size ``n_features``;
            the i-th entry contains the bin edges for i-th feature. The edges are returned
            in sorted order with duplicates removed (i.e. for a feature with less then
            ``n_bins`` unique values, the number of edges will be less than ``n_bins + 1``).
    Raises:
        ValueError: for invalid inputs

    References:
        * [1] Yury Gorishniy, Ivan Rubachev, Artem Babenko, "On Embeddings for Numerical Features in Tabular Deep Learning", 2022

    Examples:
        .. testcode::

            n_objects = 100
            n_features = 4
            X = torch.randn(n_objects, n_features)
            y = torch.randn(n_objects)
            n_bins = 3
            bin_edges = compute_decision_tree_bin_edges(
                X, n_bins, y=y, regression=True, tree_kwargs={'min_samples_leaf': 16}
            )
            assert len(bin_edges) == n_features
            for x in bin_edges:
                assert len(x) == n_bins + 1

    """
    # The implementation relies on scikit-learn, so all the computations are performed
    # in terms of numpy arrays.
    if isinstance(X, Tensor):
        is_torch = True
        X_device = X.device
        if X_device.type != 'cpu':
            warnings.warn(
                'One of the input tensors is not located on CPU.'
                ' This will cause data movements between devices.'
            )
        X = X.cpu().numpy()
        y = y.cpu().numpy()
    else:
        is_torch = False
        X_device = None
    X = cast(np.ndarray, X)
    y = cast(np.ndarray, y)

    if X.ndim != 2:
        raise ValueError('X must have two dimensions')
    if len(X) != len(y):
        raise ValueError('X and y have different first dimensions')
    if 'max_leaf_nodes' in tree_kwargs:
        raise ValueError(
            'Do not include max_leaf_nodes in tree_kwargs (it will be set equal to n_bins automatically).'
        )

    adjusted_bin_counts = _adjust_bin_counts(X, n_bins)
    edges = []
    for column, adjusted_n_bins in zip(X.T, adjusted_bin_counts):
        tree = (
            (DecisionTreeRegressor if regression else DecisionTreeClassifier)(
                max_leaf_nodes=adjusted_n_bins, **tree_kwargs
            )
            .fit(column.reshape(-1, 1), y)
            .tree_
        )
        tree_thresholds = []
        for node_id in range(tree.node_count):
            # the following condition is True only for split nodes
            # See https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
            if tree.children_left[node_id] != tree.children_right[node_id]:
                tree_thresholds.append(tree.threshold[node_id])
        tree_thresholds.append(column.min())
        tree_thresholds.append(column.max())
        edges.append(np.array(sorted(set(tree_thresholds)), dtype=X.dtype))
    return [as_tensor(x, device=X_device) for x in edges] if is_torch else edges


@overload
def compute_bin_indices(X: Tensor, bin_edges: List[Tensor]) -> Tensor:
    ...


@overload
def compute_bin_indices(X: np.ndarray, bin_edges: List[np.ndarray]) -> np.ndarray:
    ...


def compute_bin_indices(X, bin_edges):
    """Compute bin indices for the given feature values.

    The output of this function can be passed as input to:

        * `compute_bin_linear_ratios`
        * `piecewise_linear_encoding`
        * `rtdl.nn.PiecewiseLinearEncoder` (to the forward method)

    For ``X[i][j]``, compute the index ``k`` of the bin in ``bin_edges[j]`` such that
    ``bin_edges[j][k] <= X[i][j] < bin_edges[j][k + 1]``. If the value is less than the
    leftmost bin edge, ``0`` is returned. If the value is greater or equal than the rightmost
    bin edge, ``len(bin_edges[j]) - 1`` is returned.

    Args:
        X: the feature matrix. Shape: ``(n_objects, n_features)``.
        bin_edges: the bin edges for each features. Can be obtained from
            `compute_quantile_bin_edges` or `compute_decision_tree_bin_edges`.
    Return:
        bin indices: Shape: ``(n_objects, n_features)``.

    Examples:
        .. testcode::

            n_objects = 100
            n_features = 4
            X = torch.randn(n_objects, n_features)
            n_bins = 3
            bin_edges = compute_quantile_bin_edges(X, n_bins)
            bin_indices = compute_bin_indices(X, bin_edges)
    """
    is_torch = isinstance(X, Tensor)
    X = as_tensor(X)
    bin_edges = [as_tensor(x) for x in bin_edges]

    if X.ndim != 2:
        raise ValueError('X must have two dimensions')
    if X.shape[1] != len(bin_edges):
        raise ValueError(
            'The number of columns in X must be equal to the size of the `bin_edges` list'
        )

    inf = torch.tensor([math.inf], dtype=X.dtype, device=X.device)
    # NOTE: torch.bucketize(..., right=True) is consistent with np.digitize(..., right=False)
    bin_indices_list = [
        torch.bucketize(
            column, torch.cat((-inf, column_bin_edges[1:-1], inf)), right=True
        )
        - 1
        for column, column_bin_edges in zip(X.T, bin_edges)
    ]
    bin_indices = torch.stack(bin_indices_list, 1)
    return bin_indices if is_torch else bin_indices.numpy()


@overload
def compute_bin_linear_ratios(
    X: np.ndarray, bin_edges: List[np.ndarray], bin_indices: np.ndarray
) -> np.ndarray:
    ...


@overload
def compute_bin_linear_ratios(
    X: Tensor, bin_edges: List[Tensor], bin_indices: Tensor
) -> Tensor:
    ...


def compute_bin_linear_ratios(X, bin_edges, bin_indices):
    """Compute the ratios for piecewise linear encoding as described in [1].

    The output of this function can be passed as input to:

        * `piecewise_linear_encoding`
        * `rtdl.nn.PiecewiseLinearEncoder` (to the forward method)

    For details, see the section "Piecewise linear encoding" in [1].

    Args:
        X: the feature matrix. Shape: ``(n_objects, n_features)``.
        bin_edges: the bin edges for each features. Size: ``n_features``. Can be obtained from
            `compute_quantile_bin_edges` or `compute_decision_tree_bin_edges`.
        bin_indices: the bin indices (can be computed via `compute_bin_indices`)
    Return:
        ratios

    References:
        * [1] Yury Gorishniy, Ivan Rubachev, Artem Babenko, "On Embeddings for Numerical Features in Tabular Deep Learning", 2022

    Examples:
        .. testcode::

            n_objects = 100
            n_features = 4
            X = torch.randn(n_objects, n_features)
            n_bins = 3
            bin_edges = compute_quantile_bin_edges(X, n_bins)
            bin_indices = compute_bin_indices(X, bin_edges)
            bin_ratios = compute_bin_linear_ratios(X, bin_edges, bin_indices)
    """
    is_torch = isinstance(X, Tensor)
    X = as_tensor(X)
    bin_indices = as_tensor(bin_indices)
    bin_edges = [as_tensor(x) for x in bin_edges]

    if X.ndim != 2:
        raise ValueError('X must have two dimensions')
    if X.shape != bin_indices.shape:
        raise ValueError('X and bin_indices must be of the same shape')
    if X.shape[1] != len(bin_edges):
        raise ValueError(
            'The number of columns in X must be equal to the number of items in bin_edges'
        )

    inf = torch.tensor([math.inf], dtype=X.dtype, device=X.device)
    values_list = []
    # "c_" ~ "column_"
    for c_i, (c_values, c_indices, c_bin_edges) in enumerate(
        zip(X.T, bin_indices.T, bin_edges)
    ):
        if (c_indices + 1 >= len(c_bin_edges)).any():
            raise ValueError(
                f'The indices in indices[:, {c_i}] are not compatible with bin_edges[{c_i}]'
            )
        effective_c_bin_edges = torch.cat((-inf, c_bin_edges[1:-1], inf))
        if (c_values < effective_c_bin_edges[c_indices]).any() or (
            c_values > effective_c_bin_edges[c_indices + 1]
        ).any():
            raise ValueError(
                'Values in X are not consistent with the provided bin indices and edges.'
            )
        c_left_edges = c_bin_edges[c_indices]
        c_right_edges = c_bin_edges[c_indices + 1]
        values_list.append((c_values - c_left_edges) / (c_right_edges - c_left_edges))
    values = torch.stack(values_list, 1)
    return values if is_torch else values.numpy()


@overload
def _LVR_encoding(
    indices: Tensor,
    values: Tensor,
    d_encoding: Union[int, List[int]],
    left: Number,
    right: Number,
    *,
    stack: bool,
) -> Tensor:
    ...


@overload
def _LVR_encoding(
    indices: np.ndarray,
    values: np.ndarray,
    d_encoding: Union[int, List[int]],
    left: Number,
    right: Number,
    *,
    stack: bool,
) -> np.ndarray:
    ...


def _LVR_encoding(
    indices,
    values,
    d_encoding: Union[int, List[int]],
    left: Number,
    right: Number,
    *,
    stack: bool,
):
    """Left-Value-Right encoding

    For one feature:
    f(x) = [left, left, ..., left, <value at the given index>, right, right, ... right]
    """
    is_torch = isinstance(values, Tensor)
    values = as_tensor(values)
    indices = as_tensor(indices)

    if type(left) is not type(right):
        raise ValueError('left and right must be of the same type')
    if type(left).__name__ not in str(values.dtype):
        raise ValueError(
            'The `values` array has dtype incompatible with left and right'
        )
    if values.ndim != 2:
        raise ValueError('values must have two dimensions')
    if values.shape != indices.shape:
        raise ValueError('values and indices must be of the same shape')

    if stack and not isinstance(d_encoding, int):
        raise ValueError('stack can be True only if d_encoding is an integer')
    if isinstance(d_encoding, int):
        if (indices >= d_encoding).any():
            raise ValueError('All indices must be less than d_encoding')
    else:
        if values.shape[1] != len(d_encoding):
            raise ValueError(
                'If d_encoding is a list, then its size must be equal to `values.shape[1]`'
            )
        if (indices >= torch.tensor(d_encoding).to(indices)[None]).any():
            raise ValueError(
                'All indices must be less than the corresponding d_encoding'
            )

    dtype = values.dtype
    device = values.device
    n_objects, n_features = values.shape
    left_tensor = torch.tensor(left, dtype=dtype, device=device)
    right_tensor = torch.tensor(right, dtype=dtype, device=device)

    shared_d_encoding = (
        d_encoding
        if isinstance(d_encoding, int)
        else d_encoding[0]
        if all(d == d_encoding[0] for d in d_encoding)
        else None
    )

    if shared_d_encoding is None:
        encoding_list = []
        for c_values, c_indices, c_d_encoding in zip(
            values.T, indices.T, cast(List[int], d_encoding)
        ):
            c_left_mask = (
                torch.arange(c_d_encoding, device=device)[None] < c_indices[:, None]
            )
            c_encoding = torch.where(c_left_mask, left_tensor, right_tensor)
            c_encoding[torch.arange(n_objects, device=device), c_indices] = c_values
            encoding_list.append(c_encoding)
        encoding = torch.cat(encoding_list, 1)
    else:
        left_mask = (
            torch.arange(shared_d_encoding, device=device)[None, None]
            < indices[:, :, None]
        )
        encoding = torch.where(left_mask, left_tensor, right_tensor)
        # object_indices:  [0, 0, 0, ..., 1, 1, 1, ..., 2, 2, 2, ...]
        # feature_indices: [0, 1, 2, ..., 0, 1, 2, ..., 0, 1, 2, ...]
        object_indices = (
            torch.arange(n_objects, device=device)[:, None]
            .repeat(1, n_features)
            .reshape(-1)
        )
        feature_indices = torch.arange(n_features, device=device).repeat(n_objects)
        encoding[object_indices, feature_indices, indices.flatten()] = values.flatten()
        if not stack:
            encoding = encoding.reshape(n_objects, -1)
    return encoding if is_torch else encoding.numpy()


@overload
def piecewise_linear_encoding(
    bin_edges: Tensor,
    bin_indices: Tensor,
    bin_ratios: Tensor,
    d_encoding: Union[int, List[int]],
    *,
    stack: bool,
) -> Tensor:
    ...


@overload
def piecewise_linear_encoding(
    bin_edges: np.ndarray,
    bin_indices: np.ndarray,
    bin_ratios: np.ndarray,
    d_encoding: Union[int, List[int]],
    *,
    stack: bool,
) -> np.ndarray:
    ...


def piecewise_linear_encoding(
    bin_edges,
    bin_indices,
    bin_ratios,
    d_encoding: Union[int, List[int]],
    *,
    stack: bool,
):
    """Construct piecewise linear encoding as described in [1].

    See `compute_piecewise_linear_encoding` for details.

    Note:
        To compute the encoding from the original feature valies, see
        `compute_piecewise_linear_encoding`.

    Args:
        bin_ratios: linear ratios (can be computed via `compute_bin_linear_ratios`).
            Shape: ``(n_objects, n_features)``.
        bin_indices: bin indices (can be computed via `compute_bin_indices`).
            Shape: ``(n_objects, n_features)``.
        d_encoding: the encoding sizes for all features (if an integer, it is used for
            all the features)
        stack: if `True`, then d_encoding must be an integer, and the output shape is
            ``(n_objects, n_features, d_encoding)``. Otherwise, the output shape is
            ``(n_objects, sum(d_encoding))``.
    Returns:
        encoded input
    Raises:
        ValueError: for invalid input

    References:
        * [1] Yury Gorishniy, Ivan Rubachev, Artem Babenko, "On Embeddings for Numerical Features in Tabular Deep Learning", 2022

    Examples:
        .. testcode::

            n_objects = 100
            n_features = 4
            X = torch.randn(n_objects, n_features)
            n_bins = 3
            bin_edges = compute_quantile_bin_edges(X, n_bins)
            bin_indices = compute_bin_indices(X, bin_edges)
            bin_ratios = compute_bin_linear_ratios(X, bin_edges, bin_indices)
            bin_counts = [len(x) - 1 for x in bin_edges]
            X_ple = piecewise_linear_encoding(bin_edges, bin_indices, bin_ratios, bin_counts, stack=True)
    """
    is_torch = isinstance(bin_ratios, Tensor)
    bin_edges = torch.as_tensor(bin_ratios)
    bin_ratios = torch.as_tensor(bin_ratios)
    bin_indices = torch.as_tensor(bin_indices)

    if bin_ratios.ndim != 2:
        raise ValueError('bin_ratios must have two dimensions')
    if bin_ratios.shape != bin_indices.shape:
        raise ValueError('rations and bin_indices must be of the same shape')

    if isinstance(d_encoding, list) and bin_ratios.shape[1] != len(d_encoding):
        raise ValueError(
            'the number of columns in bin_ratios must be equal to the size of d_encoding'
        )

    message = (
        'bin_ratios do not satisfy requirements for the piecewise linear encoding.'
        ' Use rtdl.data.compute_bin_linear_ratios to obtain valid values.'
    )

    lower_bounds = torch.zeros_like(bin_ratios)
    is_first_bin = bin_indices == 0
    lower_bounds[is_first_bin] = -math.inf
    if (bin_ratios < lower_bounds).any():
        raise ValueError(message)
    del lower_bounds

    upper_bounds = torch.ones_like(bin_ratios)
    # it is important to use bin_edges here, not d_encoding
    is_last_bin = bin_indices + 1 == as_tensor(list(map(len, bin_edges)))
    upper_bounds[is_last_bin] = math.inf
    if (bin_ratios > upper_bounds).any():
        raise ValueError(message)
    del upper_bounds

    encoding = _LVR_encoding(bin_indices, bin_ratios, d_encoding, 1.0, 0.0, stack=stack)
    return encoding if is_torch else encoding.numpy()


@overload
def compute_piecewise_linear_encoding(
    X: Tensor, bin_edges: List[Tensor], *, stack: bool
) -> Tensor:
    ...


@overload
def compute_piecewise_linear_encoding(
    X: np.ndarray, bin_edges: List[np.ndarray], *, stack: bool
) -> np.ndarray:
    ...


def compute_piecewise_linear_encoding(X, bin_edges, *, stack: bool):
    """Compute piecewise linear encoding as described in [1].

    .. image:: ../images/piecewise_linear_encoding_figure.png
        :scale: 25%
        :alt: obtaining bins from decision trees (figure)

    .. image:: ../images/piecewise_linear_encoding_equation.png
        :scale: 25%
        :alt: obtaining bins from decision trees (equation)

    Args:
        X: the feature matrix. Shape: ``(n_objects, n_features)``.
        bin_edges: the bin edges. Size: ``n_features``. Can be computed via
            `compute_quantile_bin_edges` and `compute_decision_tree_bin_edges`.
        stack: (let ``bin_counts = [len(x) - 1 for x in bin_edges]``) if `True`, then
            the output shape is ``(n_objects, n_features, max(bin_counts))``, otherwise
            the output shape is ``(n_objects, sum(bin_counts))``.
    Returns:
        encoded input
    Raises:
        ValueError: for invalid input

    References:
        * [1] Yury Gorishniy, Ivan Rubachev, Artem Babenko, "On Embeddings for Numerical Features in Tabular Deep Learning", 2022

    Examples:
        .. testcode::

            n_objects = 100
            n_features = 4
            X = torch.randn(n_objects, n_features)
            n_bins = 3
            bin_edges = compute_quantile_bin_edges(X, n_bins)
            X_ple = compute_piecewise_linear_encoding(X, bin_edges, stack=False)
    """
    bin_indices = compute_bin_indices(X, bin_edges)
    bin_ratios = compute_bin_linear_ratios(X, bin_edges, bin_indices)
    bin_counts = [len(x) - 1 for x in bin_edges]
    return piecewise_linear_encoding(
        bin_edges,
        bin_indices,
        bin_ratios,
        d_encoding=max(bin_counts) if stack else bin_counts,
        stack=stack,
    )


class PiecewiseLinearEncoder(BaseEstimator, TransformerMixin):
    """Piecewise linear encoding as described in [1].

    The Scikit-learn Transformer-style wrapper for `compute_piecewise_linear_encoding`.
    Works only with dense NumPy arrays.

    Attributes:
        bin_edges_: the computed bin edges

    References:
        * [1] Yury Gorishniy, Ivan Rubachev, Artem Babenko, "On Embeddings for Numerical Features in Tabular Deep Learning", 2022

    Examples:
        .. testcode::

            from sklearn.linear_model import LinearRegression

            n_features = 4
            X_train = np.random.randn(70, n_features)
            X_test = np.random.randn(30, n_features)
            y_train = np.random.randn(len(X_train))
            encoder = PiecewiseLinearEncoder(
                'decision_tree',
                dict(
                    n_bins=3,
                    regression=True,
                    tree_kwargs={'min_samples_leaf': 16},
                ),
                stack=False,  # to make the output suitable for a linear model
            )
            encoder.fit(X_train)
            X_ple_train = encoder.transform(X_train)
            X_ple_test = encoder.transform(X_test)
            model = LinearRegression()
            model.fit(X_ple_train, y_train)
            y_pred_test = model.predict(X_ple_test)
    """

    def __init__(
        self,
        bin_edges: Union[
            Literal['quantile', 'decision_tree'],
            Callable[..., List[np.ndarray]],
        ],
        bin_edges_params: Optional[Dict[str, Any]],
        *,
        stack: bool,
    ) -> None:
        """
        Args:
            bin_edges: if ``'quantile'``, then `compute_quantile_bin_edges` is used.
                 If ``'decision_tree'``, then `compute_decision_tree_bin_edges` is used (
                 ``y`` is passed automatically).
                 If a custom function ``f``, then it will be called as follows:
                 ``f(X_train, **bin_edges_params)`` and it is expected to return the list
                 of NumPy arrays (bin edges).
            bin_edges_params: the keyword arguments for the corresponding function
        """
        self.bin_edges = bin_edges
        self.bin_edges_params = bin_edges_params
        self.stack = stack

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> 'PiecewiseLinearEncoder':
        """Fit the transformer."""
        if y is not None and len(X) != len(y):
            raise ValueError('X and y must have the same first dimension')
        compute_fn = cast(
            Callable[..., List[np.ndarray]],
            (
                {
                    'quantile': compute_quantile_bin_edges,
                    'decision_tree': compute_decision_tree_bin_edges,
                }[self.bin_edges]
                if isinstance(self.bin_edges, str)
                else self.bin_edges
            ),
        )
        y_kwarg = (
            {} if y is None or compute_fn is compute_quantile_bin_edges else {'y': y}
        )
        kwargs = {} if self.bin_edges_params is None else self.bin_edges_params
        self.bin_edges_ = compute_fn(X, **y_kwarg, **kwargs)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data."""
        return compute_piecewise_linear_encoding(X, self.bin_edges_, stack=self.stack)


@experimental
class NoisyQuantileTransformer(QuantileTransformer):
    """**[EXPERIMENTAL]** A variation of `sklearn.preprocessing.QuantileTransformer`.

    This transformer can be considered as one of the default preprocessing strategies
    for tabular data problems (in addition to more popular ones such as
    `sklearn.preprocessing.StandardScaler`).

    Compared to the bare `sklearn.preprocessing.QuantileTransformer`
    (which is the base class for this transformer), `NoisyQuantileTransformer` is more
    robust to columns with few unique values. It is achieved by applying noise
    (typically, of a very low magnitude) to the data during the fitting stage
    (but not during the transformation!) to deduplicate equal values.

    Note:

        As of now, no default parameter values are provided. However, a good starting
        point is the configuration used in some papers on tabular deep learning [1,2]:

        * ``n_quantiles=min(train_size // 30, 1000)`` where ``train_size`` is the number of
            objects passed to the ``.fit()`` method. This heuristic rule was tested on
            datasets with ``train_size >= 5000``.
        * ``output_distribution='normal'``
        * ``subsample=10**9``
        * ``noise_std=1e-3``

    References:
        * [1] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", NeurIPS 2021
        * [2] Yury Gorishniy, Ivan Rubachev, Artem Babenko, "On Embeddings for Numerical Features in Tabular Deep Learning", arXiv 2022
    """

    def __init__(
        self,
        *,
        n_quantiles: int,
        output_distribution: str,
        subsample: int,
        noise_std: float,
        **kwargs,
    ) -> None:
        """
        Args:
            n_quantiles: the argument for `sklearn.preprocessing.QuantileTransformer`
            output_distribution: the argument for `sklearn.preprocessing.QuantileTransformer`
            subsample: the argument for `sklearn.preprocessing.QuantileTransformer`
            noise_std: the scale of noise that is applied to "deduplicate" equal values
                during the fitting stage.
            kwargs: other arguments for `sklearn.preprocessing.QuantileTransformer`
        """
        if noise_std <= 0.0:
            raise ValueError(
                'noise_std must be positive. Note that with noise_std=0 the transformer'
                ' is equivalent to `sklearn.preprocessing.QuantileTransformer`'
            )
        super().__init__(
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            subsample=subsample,
            **kwargs,
        )
        self.noise_std = noise_std

    def fit(self, X, y=None):
        exception = ValueError('X must be either `numpy.ndarray` or `pandas.DataFrame`')
        if isinstance(X, np.ndarray):
            X_ = X
        elif hasattr(X, 'values'):
            try:
                import pandas

                if not isinstance(X, pandas.DataFrame):
                    raise exception
            except ImportError:
                raise exception
            X_ = X.values
        else:
            raise exception
        if scipy.sparse.issparse(X_):
            raise ValueError(
                'rtdl.data.NoisyQuantileTransformer does not support sparse input'
            )

        self.scaler_ = StandardScaler(with_mean=False)
        X_ = self.scaler_.fit_transform(X_)
        random_state = check_random_state(self.random_state)
        return super().fit(X_ + random_state.normal(0.0, self.noise_std, X_.shape), y)

    def transform(self, X):
        return super().transform(self.scaler_.transform(X))


def get_category_sizes(X: np.ndarray) -> List[int]:
    """Validate encoded categorical features and count distinct values.

    The function calculates the "category sizes" that can be used to construct
    `rtdl.CategoricalFeatureTokenizer` and `rtdl.FTTransformer`. Additionally, the
    following conditions are checked:

    * the data is a two-dimensional array of signed integers
    * distinct values of each column form zero-based ranges

    Note:
        For valid inputs, the result equals :code:`X.max(0) + 1`.

    Args:
        X: encoded categorical features (e.g. the output of :code:`sklearn.preprocessing.OrdinalEncoder`)

    Returns:
        The counts of distinct values for all columns.

    Examples:
        .. testcode::

            assert get_category_sizes(np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [2, 1, 0],
                ]
            )) == [3, 2, 1]
    """
    if X.ndim != 2:
        raise ValueError('X must be two-dimensional')
    if not issubclass(X.dtype.type, np.signedinteger):
        raise ValueError('X data type must be integer')
    sizes = []
    for i, column in enumerate(X.T):
        unique_values = np.unique(column)
        min_value = unique_values.min()
        if min_value != 0:
            raise ValueError(
                f'The minimum value of column {i} is {min_value}, but it must be zero.'
            )
        max_value = unique_values.max()
        if max_value + 1 != len(unique_values):
            raise ValueError(
                f'The values of column {i} do not fully cover the range from zero to maximum_value={max_value}'
            )

        sizes.append(len(unique_values))
    return sizes
