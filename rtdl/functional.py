import torch.nn.functional as F
from torch import Tensor


def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].

    References:

        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].

    References:

        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)
