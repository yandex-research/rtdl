import functools
import warnings
from typing import Callable, TypeVar

from typing_extensions import ParamSpec

from .exceptions import ExperimentalWarning

INTERNAL_ERROR_MESSAGE = (
    'Internal error. Please, open an issue here:'
    ' https://github.com/Yura52/rtdl/issues/new'
)


def all_or_none(values):
    return all(x is None for x in values) or all(x is not None for x in values)


P = ParamSpec('P')
T = TypeVar('T')


def experimental(x: Callable[P, T]) -> Callable[P, T]:
    if not callable(x):
        raise ValueError('Only callable objects can be experimental')

    @functools.wraps(x)
    def experimental_x(*args: P.args, **kwargs: P.kwargs):
        warnings.warn(
            f'{x.__name__} (full name: {x.__qualname__}) is an experimental feature of rtdl',
            ExperimentalWarning,
        )
        return x(*args, **kwargs)

    return experimental_x
