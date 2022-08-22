from typing import Callable, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

ModuleType = Union[str, Callable[..., nn.Module]]
ModuleType0 = Union[str, Callable[[], nn.Module]]


class ReGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] % 2 != 0:
            raise ValueError('The size of the last dimension must be even.')
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)


def make_nn_module(module_type: ModuleType, *args) -> nn.Module:
    if isinstance(module_type, str):
        if module_type == 'ReGLU':
            cls = ReGLU
        else:
            try:
                cls = getattr(nn, module_type)
            except AttributeError:
                raise ValueError(
                    f'There is no such module as {module_type} in torch.nn'
                )
        return cls(*args)
    else:
        return module_type(*args)
