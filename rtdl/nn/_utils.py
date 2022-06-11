from typing import Callable, Union

import torch.nn as nn

ModuleType = Union[str, Callable[..., nn.Module]]
ModuleType0 = Union[str, Callable[[], nn.Module]]


def make_nn_module(module_type: ModuleType, *args) -> nn.Module:
    if isinstance(module_type, str):
        try:
            cls = getattr(nn, module_type)
        except AttributeError as err:
            raise ValueError(
                f'Failed to construct the module {module_type} with the arguments {args}'
            ) from err
        return cls(*args)
    else:
        return module_type(*args)
