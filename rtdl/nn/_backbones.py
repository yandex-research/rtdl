from typing import List, Optional, Union

import torch.nn as nn
from torch import Tensor

from ._utils import ModuleType0, make_nn_module


class MLP(nn.Module):
    """The MLP model used in the paper "Revisiting Deep Learning Models for Tabular Data" [1].

    The following scheme describes the architecture:

    .. code-block:: text

          MLP: (in) -> Block -> ... -> Block -> Head -> (out)
        Block: (in) -> Linear -> Activation -> Dropout -> (out)
        Head == Linear

    Examples:
        .. testcode::

            x = torch.randn(4, 2)
            model = MLP.make_baseline(
                d_in=x.shape[1],
                d_out=1,
                n_blocks=2,
                d_layer=3,
                dropout=0.1,
            )
            assert model(x).shape == (len(x), 1)

    Attributes:
        blocks: the main blocks of the model (`torch.nn.Sequential` of `MLP.Block`s)
        head: (optional) the last layer (`MLP.Head`)

    References:
        * [1] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    class Block(nn.Module):
        """The main building block of `MLP`."""

        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            activation: ModuleType0,
            dropout: float,
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            return self.dropout(self.activation(self.linear(x)))

    Head = nn.Linear
    """The last layer of `MLP`."""

    def __init__(
        self,
        *,
        d_in: int,
        d_out: Optional[int],
        d_layers: List[int],
        dropouts: Union[float, List[float]],
        activation: ModuleType0,
    ) -> None:
        """
        Note:
            Use the `make_baseline` method instead of the constructor unless you need
            more control over the architecture.
        """
        if not d_layers:
            raise ValueError('d_layers must be non-empty')
        super().__init__()
        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(d_layers)
        assert len(d_layers) == len(dropouts)

        self.blocks = nn.Sequential(
            *[
                MLP.Block(
                    d_in=d_layers[i - 1] if i else d_in,
                    d_out=d,
                    bias=True,
                    activation=activation,
                    dropout=dropout,
                )
                for i, (d, dropout) in enumerate(zip(d_layers, dropouts))
            ]
        )
        self.head = (
            None
            if d_out is None
            else MLP.Head(d_layers[-1] if d_layers else d_in, d_out)
        )

    @classmethod
    def make_baseline(
        cls,
        *,
        d_in: int,
        d_out: Optional[int],
        n_blocks: int,
        d_layer: int,
        dropout: float,
    ) -> 'MLP':
        """Create a "baseline" `MLP`.

        Features:

        * :code:`Activation` = :code:`ReLU`
        * all linear layers have the same dimension
        * the dropout rate is the same for all blocks

        Args:
            d_in: the input size.
            d_out: the output size of the `MLP.Head`. If `None`, then the output of MLP
                will be the output of the last block, i.e. the model will be
                backbone-only.
            n_blocks: the number of blocks.
            d_layer: the dimension of each linear layer.
            dropout: the dropout rate for all hidden layers.
        Returns:
            MLP
        """
        if n_blocks <= 0:
            raise ValueError('n_blocks must be positive')
        if not isinstance(dropout, float):
            raise ValueError('In this constructor, dropout must be float')
        return MLP(
            d_in=d_in,
            d_out=d_out,
            d_layers=[d_layer] * n_blocks if n_blocks else [],  # type: ignore
            dropouts=dropout,
            activation='ReLU',
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        if self.head is not None:
            x = self.head(x)
        return x
