from typing import List, Optional

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

    Attributes:
        blocks: the main blocks of the model (`torch.nn.Sequential` of `MLP.Block`s)
        head: (optional) the last layer (`MLP.Head`)

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
    """The output module of `MLP`."""

    def __init__(
        self,
        *,
        d_in: int,
        d_out: Optional[int],
        d_layers: List[int],
        dropout: float,
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

        self.blocks = nn.Sequential(
            *[
                MLP.Block(
                    d_in=d_layers[i - 1] if i else d_in,
                    d_out=d,
                    bias=True,
                    activation=activation,
                    dropout=dropout,
                )
                for i, d in enumerate(d_layers)
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

        * all linear layers have the same dimension
        * :code:`Activation` = :code:`ReLU`

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
            dropout=dropout,
            activation='ReLU',
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        if self.head is not None:
            x = self.head(x)
        return x


class ResNet(nn.Module):
    """The ResNet model used in the paper "Revisiting Deep Learning Models for Tabular Data" [1].

    The following scheme describes the architecture:

    .. code-block:: text

        ResNet: (in) -> Linear -> Block -> ... -> Block -> Head -> (out)

                 |-> Norm -> Linear -> Activation -> Dropout -> Linear -> Dropout ->|
                 |                                                                  |
         Block: (in) ------------------------------------------------------------> Add -> (out)

          Head: (in) -> Norm -> Activation -> Linear -> (out)

    Attributes:
        blocks: the main blocks of the model (`torch.nn.Sequential` of `ResNet.Block`s)
        head: (optional) the last layer (`ResNet.Head`)

    Examples:
        .. testcode::

            x = torch.randn(4, 2)
            module = ResNet.make_baseline(
                d_in=x.shape[1],
                d_out=1,
                n_blocks=2,
                d_main=3,
                d_hidden=4,
                dropout_first=0.25,
                dropout_second=0.0,
            )
            assert module(x).shape == (len(x), 1)

    References:
        * [1] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    class Block(nn.Module):
        """The main building block of `ResNet`."""

        def __init__(
            self,
            *,
            d_main: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout_first: float,
            dropout_second: float,
            normalization: ModuleType0,
            activation: ModuleType0,
            skip_connection: bool,
        ) -> None:
            super().__init__()
            self.normalization = make_nn_module(normalization, d_main)
            self.linear_first = nn.Linear(d_main, d_hidden, bias_first)
            self.activation = make_nn_module(activation)
            self.dropout_first = nn.Dropout(dropout_first)
            self.linear_second = nn.Linear(d_hidden, d_main, bias_second)
            self.dropout_second = nn.Dropout(dropout_second)
            self.skip_connection = skip_connection

        def forward(self, x: Tensor) -> Tensor:
            x_input = x
            x = self.normalization(x)
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout_first(x)
            x = self.linear_second(x)
            x = self.dropout_second(x)
            if self.skip_connection:
                x = x_input + x
            return x

    class Head(nn.Module):
        """The output module of `ResNet`."""

        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            normalization: ModuleType0,
            activation: ModuleType0,
        ) -> None:
            super().__init__()
            self.normalization = make_nn_module(normalization, d_in)
            self.activation = make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias)

        def forward(self, x: Tensor) -> Tensor:
            if self.normalization is not None:
                x = self.normalization(x)
            x = self.activation(x)
            x = self.linear(x)
            return x

    def __init__(
        self,
        *,
        d_in: int,
        d_out: Optional[int],
        n_blocks: int,
        d_main: int,
        d_hidden: int,
        dropout_first: float,
        dropout_second: float,
        normalization: ModuleType0,
        activation: ModuleType0,
    ) -> None:
        """
        Note:
            Use the `make_baseline` method instead of the constructor unless you need
            more control over the architecture.


        """
        super().__init__()

        self.first_layer = nn.Linear(d_in, d_main)
        self.blocks = nn.Sequential(
            *[
                ResNet.Block(
                    d_main=d_main,
                    d_hidden=d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout_first=dropout_first,
                    dropout_second=dropout_second,
                    normalization=normalization,
                    activation=activation,
                    skip_connection=True,
                )
                for _ in range(n_blocks)
            ]
        )
        self.head = (
            None
            if d_out is None
            else ResNet.Head(
                d_in=d_main,
                d_out=d_out,
                bias=True,
                normalization=normalization,
                activation=activation,
            )
        )

    @classmethod
    def make_baseline(
        cls,
        *,
        d_in: int,
        d_out: Optional[int],
        n_blocks: int,
        d_main: int,
        d_hidden: int,
        dropout_first: float,
        dropout_second: float,
    ) -> 'ResNet':
        """Create a "baseline" `ResNet`.

        Features:

        * :code:`Activation` = :code:`ReLU`
        * :code:`Norm` = :code:`BatchNorm1d`

        Args:
            d_in: the input size
            d_out: the output size of the `ResNet.Head`. If `None`, then the output of
                ResNet will be the output of the last block, i.e. the model will be
                backbone-only.
            n_blocks: the number of blocks
            d_main: the input size (or, equivalently, the output size) of each block
            d_hidden: the output size of the first linear layer in each block
            dropout_first: the dropout rate of the first dropout layer in each block.
            dropout_second: the dropout rate of the second dropout layer in each block.
                The value `0.0` is a good starting point.
        """
        return cls(
            d_in=d_in,
            d_out=d_out,
            n_blocks=n_blocks,
            d_main=d_main,
            d_hidden=d_hidden,
            dropout_first=dropout_first,
            dropout_second=dropout_second,
            normalization='BatchNorm1d',
            activation='ReLU',
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.first_layer(x)
        x = self.blocks(x)
        if self.head is not None:
            x = self.head(x)
        return x
