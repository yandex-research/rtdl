from typing import TypeVar

from typing_extensions import ParamSpec

INTERNAL_ERROR_MESSAGE = (
    'Internal error. Please, open an issue here:'
    ' https://github.com/Yura52/rtdl/issues/new'
)


def all_or_none(values):
    return all(x is None for x in values) or all(x is not None for x in values)


P = ParamSpec('P')
T = TypeVar('T')
