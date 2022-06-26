"""Tools for data (pre)processing."""

__all__ = [
    'compute_quantile_bin_edges',
    'compute_decision_tree_bin_edges',
    'compute_bin_indices',
    'compute_piecewise_linear_bin_values',
    'one_hot_encoding',
    'ordinal_binary_encoding',
    'piecewise_linear_encoding',
    'get_category_sizes',
]

import warnings
from typing import Any, Dict, List, TypeVar, Union

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

Number = TypeVar('Number', int, float)


def _adjust_bin_counts(X: np.ndarray, n_bins: int) -> List[int]:
    if n_bins < 2:
        raise ValueError('n_bins must be greater than 1')
    adjusted_bin_counts = []
    for i, column in enumerate(X.T):
        n_unique = len(np.unique(column))
        if n_unique < 2:
            raise ValueError(f'All elements in the column {i} are the same')
        if n_unique < n_bins:
            warnings.warn(
                f'For the feature {i}, the number of bins will be set to the number of'
                ' distinct values, becuase the provided n_bins is greater than this number.'
            )
        adjusted_bin_counts.append(min(n_bins, n_unique))
    return adjusted_bin_counts


def compute_quantile_bin_edges(X: np.ndarray, n_bins: int) -> List[np.ndarray]:
    if X.ndim != 2:
        raise ValueError('X must have two dimensions')
    adjusted_bin_counts = _adjust_bin_counts(X, n_bins)
    edges = []
    for column, adjusted_n_bins in zip(X.T, adjusted_bin_counts):
        # quantiles include 0.0 and 1.0
        quantiles = np.linspace(0.0, 1.0, adjusted_n_bins + 1)
        edges.append(np.unique(np.quantile(column, quantiles)))
    return edges


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


def compute_bin_indices(
    X: np.ndarray, bin_edges: List[np.ndarray], dtype: Union[str, type] = np.int64
) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError('X must have two dimensions')
    if X.shape[1] != len(bin_edges):
        raise ValueError(
            'The number of columns must be equal to the size of the `bin_edges` list'
        )

    bin_indices = [
        np.digitize(column, np.r_[-np.inf, column_bins[1:-1], np.inf]).astype(dtype) - 1
        for column, column_bins in zip(X.T, bin_edges)
    ]
    return np.column_stack(bin_indices)


def compute_piecewise_linear_bin_values(
    X: np.ndarray, indices: np.ndarray, bin_edges: List[np.ndarray]
) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError('X must have two dimensions')
    if X.shape != indices.shape:
        raise ValueError('X and indices must be of the same shape')
    if X.shape[1] != len(bin_edges):
        raise ValueError(
            'The number of columns in X must be equal to the number of items in bin_edges'
        )
    values = []
    # "c_" ~ "column_"
    for c_i, (c_values, c_indices, c_bin_edges) in enumerate(
        zip(X.T, indices.T, bin_edges)
    ):
        if (c_indices + 1 >= len(c_bin_edges)).any():
            raise ValueError(
                f'The indices in indices[:, {c_i}] are not compatible with bin_edges[{c_i}]'
            )
        c_left_edges = c_bin_edges[c_indices]
        c_right_edges = c_bin_edges[c_indices + 1]
        values.append((c_values - c_left_edges) / (c_right_edges - c_left_edges))
    return np.column_stack(values)


# LVR stands for "left-value-right"
def _LVR_encoding(
    values: np.ndarray,
    indices: np.ndarray,
    d_encoding: int,
    left: Number,
    right: Number,
) -> np.ndarray:
    if type(left) is not type(right):
        raise ValueError('left and right must be of the same type')
    if not str(values.dtype).startswith(str(type(left))):
        raise ValueError(
            'The `values` array has dtype incompatible with left and right'
        )
    if values.ndim != 2:
        raise ValueError('values must have two dimensions')
    if values.shape != indices.shape:
        raise ValueError('values and indices must be of the same shape')
    if (indices >= d_encoding).any():
        raise ValueError('All indices must be less than d_encoding')

    n_objects, n_features = values.shape
    left_mask = np.arange(d_encoding)[None, None] < indices[:, :, None]
    encoding = np.where(
        left_mask, np.array(left, values.dtype), np.array(right, values.dtype)
    )
    # object_indices:  [0, 0, 0, ..., 1, 1, 1, ..., 2, 2, 2, ...]
    # feature_indices: [0, 1, 2, ..., 0, 1, 2, ..., 0, 1, 2, ...]
    object_indices = np.arange(n_objects).repeat(n_features)
    feature_indices = np.tile(np.arange(n_features), n_objects)
    encoding[object_indices, feature_indices, indices.flatten()] = values.flatten()
    return encoding


def one_hot_encoding(indices: np.ndarray, d_encoding: int, dtype: type) -> np.ndarray:
    return _LVR_encoding(
        np.full(indices.shape, 1, dtype=dtype), indices, d_encoding, dtype(0), dtype(0)
    )


def ordinal_binary_encoding(
    indices: np.ndarray, d_encoding: int, dtype: type
) -> np.ndarray:
    return _LVR_encoding(
        np.full(indices.shape, 1, dtype=dtype), indices, d_encoding, dtype(1), dtype(0)
    )


def piecewise_linear_encoding(
    values: np.ndarray, indices: np.ndarray, d_encoding: int
) -> np.ndarray:
    message = (
        'values do not satisfy requirements for the piecewise linear encoding.'
        ' Use rtdl.data.compute_piecewise_linear_bin_values to obtain valid values.'
    )

    lower_bounds = np.zeros_like(values)
    is_first_bin = indices == 0
    lower_bounds[is_first_bin] = -np.inf
    if (values < lower_bounds).any():
        raise ValueError(message)
    del lower_bounds

    upper_bounds = np.ones_like(values)
    is_last_bin = indices + 1 == d_encoding
    upper_bounds[is_last_bin] = np.inf
    if (values > upper_bounds).any():
        raise ValueError(message)
    del upper_bounds

    dtype = values.dtype
    return _LVR_encoding(values, indices, d_encoding, dtype(1), dtype(0))


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
