import numpy as np
import pytest

import rtdl.data


def test_check_category_sizes():
    check_category_sizes = rtdl.data.check_category_sizes

    # not two dimensions
    with pytest.raises(AssertionError):
        check_category_sizes(np.array([0, 0, 0]))
    with pytest.raises(AssertionError):
        check_category_sizes(np.array([[[0, 0, 0]]]))

    # not signed integers
    for dtype in [np.uint32, np.float32, str]:
        with pytest.raises(AssertionError):
            check_category_sizes(np.array([[0, 0, 0]], dtype=dtype))

    # non-zero min value
    for x in [-1, 1]:
        with pytest.raises(AssertionError):
            check_category_sizes(np.array([[0, 0, x]]))

    # not full range
    with pytest.raises(AssertionError):
        check_category_sizes(np.array([[0, 0, 0], [2, 1, 1]]))

    # correctness
    assert check_category_sizes(np.array([[0]])) == [1]
    assert check_category_sizes(np.array([[0], [1]])) == [2]
    assert check_category_sizes(np.array([[0, 0]])) == [1, 1]
    assert check_category_sizes(np.array([[0, 0], [0, 1]])) == [1, 2]
    assert check_category_sizes(np.array([[1, 0], [0, 1]])) == [2, 2]
