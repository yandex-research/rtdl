"""Tools for data (pre)processing."""

from typing import List

import numpy as np


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
    assert X.ndim == 2
    assert issubclass(X.dtype.type, np.signedinteger)
    sizes = []
    for i, column in enumerate(X.T):
        unique_values = np.unique(column)
        min_value = unique_values.min()
        assert (
            min_value == 0
        ), f'The minimum value of column {i} is {min_value}, but it must be zero.'
        max_value = unique_values.max()
        assert max_value + 1 == len(
            unique_values
        ), f'The values of column {i} do not fully cover the range from zero to maximum_value={max_value}'

        sizes.append(len(unique_values))
    return sizes
