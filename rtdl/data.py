"""Tools for data (pre)processing."""

__all__ = [
    'compute_quantile_bin_edges',
    'compute_decision_tree_bin_edges',
    'compute_bin_indices',
    'compute_bin_linear_ratios',
    'piecewise_linear_encoding',
    'PiecewiseLinearEncoder',
    'get_category_sizes',
]

import math
import warnings
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast, overload

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from torch import Tensor, as_tensor
from typing_extensions import Literal

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
    is_torch = isinstance(X, Tensor)
    X = as_tensor(X)

    if X.ndim != 2:
        raise ValueError('X must have two dimensions')
    adjusted_bin_counts = _adjust_bin_counts(X, n_bins)
    edges = []
    for column, adjusted_n_bins in zip(X.T, adjusted_bin_counts):
        # quantiles include 0.0 and 1.0
        quantiles = torch.linspace(0.0, 1.0, adjusted_n_bins + 1)
        edges.append(torch.sort(torch.unique(torch.quantile(column, quantiles))).values)
    return edges if is_torch else [x.numpy() for x in edges]


def compute_decision_tree_bin_edges(
    X: np.ndarray,
    n_bins: int,
    *,
    y: np.ndarray,
    regression: bool,
    tree_kwargs: Dict[str, Any],
) -> List[np.ndarray]:
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
        edges.append(np.array(sorted(set(tree_thresholds))))
    return edges


@overload
def compute_bin_indices(X: Tensor, bin_edges: List[Tensor]) -> Tensor:
    ...


@overload
def compute_bin_indices(X: np.ndarray, bin_edges: List[np.ndarray]) -> np.ndarray:
    ...


def compute_bin_indices(X, bin_edges):
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
    X: np.ndarray, indices: np.ndarray, bin_edges: List[np.ndarray]
) -> np.ndarray:
    ...


@overload
def compute_bin_linear_ratios(
    X: Tensor, indices: Tensor, bin_edges: List[Tensor]
) -> Tensor:
    ...


def compute_bin_linear_ratios(X, indices, bin_edges):
    is_torch = isinstance(X, Tensor)
    X = as_tensor(X)
    indices = as_tensor(indices)
    bin_edges = [as_tensor(x) for x in bin_edges]

    if X.ndim != 2:
        raise ValueError('X must have two dimensions')
    if X.shape != indices.shape:
        raise ValueError('X and indices must be of the same shape')
    if X.shape[1] != len(bin_edges):
        raise ValueError(
            'The number of columns in X must be equal to the number of items in bin_edges'
        )

    inf = torch.tensor([math.inf], dtype=X.dtype, device=X.device)
    values_list = []
    # "c_" ~ "column_"
    for c_i, (c_values, c_indices, c_bin_edges) in enumerate(
        zip(X.T, indices.T, bin_edges)
    ):
        if (c_indices + 1 >= len(c_bin_edges)).any():
            raise ValueError(
                f'The indices in indices[:, {c_i}] are not compatible with bin_edges[{c_i}]'
            )
        effective_c_bin_edges = torch.cat((-inf, c_bin_edges[1:-1], inf))
        if (
            (c_values < effective_c_bin_edges[c_indices]).any()
            or (c_values > effective_c_bin_edges[c_indices + 1])
        ).any():
            raise ValueError('Values in X do not satisfy the provided bin edges.')
        c_left_edges = c_bin_edges[c_indices]
        c_right_edges = c_bin_edges[c_indices + 1]
        values_list.append((c_values - c_left_edges) / (c_right_edges - c_left_edges))
    values = torch.stack(values_list, 1)
    return values if is_torch else values.numpy()


@overload
def _LVR_encoding(
    values: Tensor,
    indices: Tensor,
    d_encoding: Union[int, List[int]],
    left: Number,
    right: Number,
) -> Tensor:
    ...


@overload
def _LVR_encoding(
    values: np.ndarray,
    indices: np.ndarray,
    d_encoding: Union[int, List[int]],
    left: Number,
    right: Number,
) -> np.ndarray:
    ...


def _LVR_encoding(
    values, indices, d_encoding: Union[int, List[int]], left: Number, right: Number
):
    """Left-Value-Right encoding

    For one feature:
    f(x) = [left, left, ..., left, <value at the given index>, right, right, ... right]
    """
    is_torch = isinstance(values, Tensor)
    values = as_tensor(values)
    indices = as_tensor(values)

    if type(left) is not type(right):
        raise ValueError('left and right must be of the same type')
    if str(type(left)) not in str(values.dtype):
        raise ValueError(
            'The `values` array has dtype incompatible with left and right'
        )
    if values.ndim != 2:
        raise ValueError('values must have two dimensions')
    if values.shape != indices.shape:
        raise ValueError('values and indices must be of the same shape')

    if isinstance(d_encoding, int):
        output_ndim = 2
        if (indices >= d_encoding).any():
            raise ValueError('All indices must be less than d_encoding')
    else:
        output_ndim = 3
        if values.shape[1] != len(d_encoding):
            raise ValueError(
                'If d_encoding is a list, then its size must be equal to `values.shape[1]`'
            )
        if (indices >= np.array(d_encoding)[None]).any():
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
        if output_ndim == 2:
            encoding = encoding.reshape(n_objects, -1)
    return encoding if is_torch else encoding.numpy()


@overload
def piecewise_linear_encoding(
    ratios: Tensor, indices: Tensor, d_encoding: Union[int, List[int]]
) -> Tensor:
    ...


@overload
def piecewise_linear_encoding(
    ratios: np.ndarray, indices: np.ndarray, d_encoding: Union[int, List[int]]
) -> np.ndarray:
    ...


def piecewise_linear_encoding(ratios, indices, d_encoding: Union[int, List[int]]):
    is_torch = isinstance(ratios, Tensor)
    ratios = torch.as_tensor(ratios)
    indices = torch.as_tensor(indices)

    if ratios.ndim != 2:
        raise ValueError('ratios must have two dimensions')
    if ratios.shape != indices.shape:
        raise ValueError('rations and indices must be of the same shape')

    if isinstance(d_encoding, list) and ratios.shape[1] != len(d_encoding):
        raise ValueError(
            'the number of columns in ratios must be equal to the size of d_encoding'
        )

    message = (
        'ratios do not satisfy requirements for the piecewise linear encoding.'
        ' Use rtdl.data.compute_bin_linear_ratios to obtain valid values.'
    )

    lower_bounds = torch.zeros_like(ratios)
    is_first_bin = indices == 0
    lower_bounds[is_first_bin] = -math.inf
    if (ratios < lower_bounds).any():
        raise ValueError(message)
    del lower_bounds

    upper_bounds = torch.ones_like(ratios)
    is_last_bin = indices + 1 == (
        d_encoding
        if isinstance(d_encoding, int)
        else torch.as_tensor(d_encoding, dtype=indices.dtype, device=indices.device)
    )
    upper_bounds[is_last_bin] = math.inf
    if (ratios > upper_bounds).any():
        raise ValueError(message)
    del upper_bounds

    encoding = _LVR_encoding(ratios, indices, d_encoding, 1.0, 0.0)
    return encoding if is_torch else encoding.numpy()


@overload
def compute_piecewise_linear_encoding(
    X: Tensor, bin_edges: List[Tensor], *, optimize_shape: bool
) -> Tensor:
    ...


@overload
def compute_piecewise_linear_encoding(
    X: np.ndarray, bin_edges: List[np.ndarray], *, optimize_shape: bool
) -> np.ndarray:
    ...


def compute_piecewise_linear_encoding(X, bin_edges, *, optimize_shape: bool):
    bin_indices = compute_bin_indices(X, bin_edges)
    bin_ratios = compute_bin_linear_ratios(X, bin_indices, bin_edges)
    bin_counts = [len(x) - 1 for x in bin_edges]
    max_n_bins = max(bin_counts)
    return piecewise_linear_encoding(
        bin_ratios, bin_indices, d_encoding=bin_counts if optimize_shape else max_n_bins
    )


class PiecewiseLinearEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        bin_edges: Union[
            Literal['quantile', 'decision_tree'],
            Callable[..., List[np.ndarray]],
        ],
        bin_edges_params: Optional[Dict[str, Any]],
        optimize_shape: bool,
    ) -> None:
        self.bin_edges = bin_edges
        self.bin_edges_params = bin_edges_params
        self.optimize_shape = optimize_shape

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> 'PiecewiseLinearEncoder':
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
        return compute_piecewise_linear_encoding(
            X, self.bin_edges_, optimize_shape=self.optimize_shape
        )


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
