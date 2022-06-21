"""Tools for data (pre)processing."""

from typing import Any, Callable, Dict, List, Union

import numpy as np

_RawSplittedData = Dict[str, Dict[str, Any]]


class SplittedData:
    def __init__(self, data: _RawSplittedData) -> None:
        self._data = data

    @property
    def data(self) -> _RawSplittedData:
        return self._data

    def __getitem__(self, index) -> Dict[str, Any]:
        if isinstance(index, str):
            key_data = self.data.get(index)
            if key_data is None:
                raise ValueError(
                    'If the index is one value, then it must be a valid key for the'
                    f' original data dictionary. Valid keys: [{", ".join(self.data)}]'
                )
            return key_data
        elif isinstance(index, tuple):
            error = ValueError(
                'If you access the data like splitted_data[part, idx])'
                ' then `part` must be a valid key for all second-level dictionaries'
                ' and `idx` must be a valid index for the corresponding values of all'
                ' second-level dictionaries'
            )
            if len(index) != 2:
                raise error
            part, idx = index
            try:
                return {key: key_data[part][idx] for key, key_data in self.data.items()}
            except Exception as err:
                raise error from err
        else:
            raise ValueError(
                'index must be either one value (string) or two values (string, index)'
            )

    def map(
        self,
        fn: Callable,
        keys: Union[None, str, List[str]] = None,
        parts: Union[None, str, List[str]] = None,
    ) -> 'SplittedData':
        # TODO: mention that the copy is not deep
        if keys is None:
            keys = list(self.data)
        elif isinstance(keys, str):
            keys = [keys]

        if isinstance(parts, str):
            parts = [parts]

        new_data = {k: v.copy() for k, v in self.data.items()}
        for key in keys:
            key_parts = self.data[key] if parts is None else parts
            for part in key_parts:
                new_data[key][part] = fn(self.data[key][part])
        return SplittedData(new_data)


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
