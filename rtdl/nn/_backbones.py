import time
import warnings
from collections import OrderedDict
from typing import List, Optional, Union

import torch.nn as nn
from torch import Tensor
from typing_extensions import Literal, get_args

from .._utils import INTERNAL_ERROR_MESSAGE, all_or_none
from ._attention import MultiheadAttention
from ._embeddings import CLSEmbedding
from ._utils import ModuleType, ModuleType0, ReGLU, make_nn_module


class MLP(nn.Module):
    """The MLP model used in the paper "Revisiting Deep Learning Models for Tabular Data" [1].

    **Input shape**: ``(n_objects, n_features)``.

    The following scheme describes the architecture:

    .. code-block:: text

          MLP: (in) -> Block -> ... -> Block -> Head -> (out)
        Block: (in) -> Linear -> Activation -> Dropout -> (out)
        Head == Linear

    Attributes:
        blocks: the main blocks of the model (`torch.nn.Sequential` of `MLP.Block`)
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
        dropouts: Union[float, List[float]],
        activation: ModuleType0,
    ) -> None:
        """
        Note:
            Use the `make_baseline` method instead of the constructor unless you need more
            control over the architecture.
        """
        if not d_layers:
            raise ValueError('d_layers must be non-empty')
        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(d_layers)
        if len(dropouts) != len(d_layers):
            raise ValueError(
                'if dropouts is a list, then its size must be equal to the size of d_layers'
            )

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
        """A simplified constructor for building baseline MLPs.

        Features:

        * all linear layers have the same dimension
        * all dropout layers have the same dropout rate
        * all activations are ``ReLU``

        Args:
            d_in: the input size.
            d_out: the output size of `MLP.Head`. If `None`, then the output of MLP
                will be the output of the last block, i.e. the model will be
                backbone-only.
            n_blocks: the number of blocks.
            d_layer: the dimension of each linear layer.
            dropout: the dropout rate for all hidden layers.
        Returns:
            mlp
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


class ResNet(nn.Module):
    """The ResNet model used in the paper "Revisiting Deep Learning Models for Tabular Data" [1].

    **Input shape**: ``(n_objects, n_features)``.

    The following scheme describes the architecture:

    .. code-block:: text

        ResNet: (in) -> Linear -> Block -> ... -> Block -> Head -> (out)

                 |-> Norm -> Linear -> Activation -> Dropout -> Linear -> Dropout ->|
                 |                                                                  |
         Block: (in) ------------------------------------------------------------> Add -> (out)

          Head: (in) -> Norm -> Activation -> Linear -> (out)

    Attributes:
        blocks: the main blocks of the model (`torch.nn.Sequential` of `ResNet.Block`)
        head: (optional) the last module (`ResNet.Head`)

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
        """A simplified constructor for building baseline ResNets.

        Features:

        * all activations are ``ReLU``
        * all normalizations are ``BatchNorm1d``

        Args:
            d_in: the input size
            d_out: the output size of `ResNet.Head`. If `None`, then the output of
                ResNet will be the output of the last block, i.e. the model will be
                backbone-only.
            n_blocks: the number of blocks
            d_main: the input size (or, equivalently, the output size) of each block
            d_hidden: the output size of the first linear layer in each block
            dropout_first: the dropout rate of the first dropout layer in each block.
            dropout_second: the dropout rate of the second dropout layer in each block.
                The value `0.0` is a good starting point.
        Return:
            resnet
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


def _is_reglu(module: ModuleType) -> bool:
    return isinstance(module, str) and module == 'ReGLU' or module is ReGLU


Pooling = Literal['cls', 'avg', 'first-token']


class Transformer(nn.Module):
    """Transformer with extra features.

    **Input shape**: ``(n_objects, n_tokens, d_embedding)``.

    The following scheme describes the architecture:

    .. code-block:: text

        Transformer: (in: 3d) -> Block -> ... -> Block -> Head -> (out: 2d)
        Block: (in: 3d) -> <see the paper [1] or the impementation> -> (out: 3d)
        Head: (in: 2d) -> HeadNormalization -> HeadActivation -> Linear -> (out: 2d)

    Compared to the vanilla Transformer [1], this implementation provides several optional features:

    * prenormalization (the common belief is that it enables easier optimization,
        but sometimes at the cost of worse results)
    * technique from Linformer [2] for faster attention when the number of tokens is large
    * inference by average pooling instead of the CLS token
    * managed CLS token

    Attributes:
        blocks: the main blocks of the model (`torch.nn.Sequential` of `Transformer.Block`)
        head: (optional) the last layer (`Transformer.Head`)

    Examples:
        .. testcode::

            x = torch.randn(4, 2, 3)
            model = Transformer.make_baseline(
                d_embedding=x.shape[-1],
                d_out=1,
                n_blocks=1,
                attention_n_heads=1,
                attention_dropout=0.0,
                ffn_d_hidden=1,
                ffn_dropout=0.0,
                activation='Identity',
                residual_dropout=0.0,
                pooling: Optional[Pooling]='cls',
                last_block_pooling_token_only=True,
            )
            assert model(x).shape == (len(x), 1)

    References:
        * [1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, "Attention Is All You Need", NeurIPS 2017
        * [2] Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma "Linformer: Self-Attention with Linear Complexity", 2020
    """

    WARNINGS = {'first_prenormalization': True, 'prenormalization': True}

    class Block(nn.Module):
        def __init__(
            self,
            *,
            d_embedding: int,
            attention_n_heads: int,
            attention_dropout: float,
            attention_normalization: ModuleType,
            attention_residual_dropout: float,
            attention_skip_connection: bool,
            linformer_compression_ratio: Optional[float],
            linformer_sharing_policy: Optional[str],
            n_tokens: Optional[int],
            ffn_d_hidden: int,
            ffn_dropout: float,
            ffn_activation: ModuleType,
            ffn_normalization: ModuleType,
            ffn_residual_dropout: float,
            ffn_skip_connection: bool,
            prenormalization: bool,
            pooling_index: Optional[int],
        ):
            """The main building block of `Transformer`."""
            super().__init__()
            self.prenormalization = prenormalization
            self.pooling_index = pooling_index

            self.attention_normalization = make_nn_module(
                attention_normalization, d_embedding
            )
            self.attention = MultiheadAttention(
                d_embedding=d_embedding,
                n_heads=attention_n_heads,
                dropout=attention_dropout,
                linformer_compression_ratio=linformer_compression_ratio,
                linformer_sharing_policy=linformer_sharing_policy,
                n_tokens=n_tokens,
            )
            self.attention_residual_dropout = nn.Dropout(attention_residual_dropout)
            self.attention_skip_connection = attention_skip_connection

            self.ffn_normalization = make_nn_module(ffn_normalization, d_embedding)
            ffn_d_hidden_first = ffn_d_hidden * (2 if _is_reglu(ffn_activation) else 1)
            self.ffn = nn.Sequential(
                OrderedDict(
                    [
                        ('first_linear', nn.Linear(d_embedding, ffn_d_hidden_first)),
                        ('activation', make_nn_module(ffn_activation)),
                        ('dropout', nn.Dropout(ffn_dropout)),
                        ('second_linear', nn.Linear(ffn_d_hidden, d_embedding)),
                    ]
                )
            )
            self.ffn_residual_dropout = nn.Dropout(ffn_residual_dropout)
            self.ffn_skip_connection = ffn_skip_connection

        def forward(self, x: Tensor) -> Tensor:
            for stage in ['attention', 'ffn']:
                normalization = getattr(self, stage + '_normalization')
                residual_dropout = getattr(self, stage + '_residual_dropout')
                skip_connection = getattr(self, stage + '_skip_connection')

                # start residual
                x_residual = x
                if self.prenormalization:
                    x_residual = normalization(x_residual)

                # apply the module
                if stage == 'attention':
                    if self.pooling_index is None:
                        x_residual = self.attention(x_residual, x_residual)
                    else:
                        pooling_slice = slice(
                            self.pooling_index, self.pooling_index + 1
                        )
                        x_residual = self.attention(
                            x_residual[:, pooling_slice], x_residual
                        )
                        x = x[:, pooling_slice]
                else:
                    x_residual = self.ffn(x_residual)

                # end residual
                x_residual = residual_dropout(x_residual)
                x = x + x_residual if skip_connection else x_residual
                if not self.prenormalization:
                    x = normalization(x)

            return x

    class Head(nn.Module):
        """The output module of `Transformer`."""

        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            activation: ModuleType0,
            normalization: ModuleType,
        ):
            super().__init__()
            self.normalization = make_nn_module(normalization, d_in)
            self.activation = make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias)

        def forward(self, x: Tensor) -> Tensor:
            x = self.normalization(x)
            x = self.activation(x)
            x = self.linear(x)
            return x

    def __init__(
        self,
        *,
        d_embedding: int,
        d_out: Optional[int],
        n_blocks: int,
        # attention
        attention_n_heads: int,
        attention_dropout: float,
        attention_normalization: str,
        attention_residual_dropout: float,
        # ffn
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: str,
        ffn_normalization: str,
        ffn_residual_dropout: float,
        # block
        prenormalization: bool,
        first_prenormalization: bool,
        # inference
        pooling: Optional[Pooling],
        # head
        head_activation: Optional[ModuleType0],
        head_normalization: Optional[ModuleType],
        # linformer
        linformer_compression_ratio: Optional[float] = None,
        linformer_sharing_policy: Optional[str] = None,
        n_tokens: Optional[int] = None,
    ) -> None:
        """
        Note:
            Use the `make_baseline` method instead of the constructor unless you need
            more control over the architecture.
        """
        super().__init__()
        if n_blocks < 1:
            raise ValueError('n_blocks must be positive')
        pooling_valid_values = get_args(Pooling)
        if pooling not in pooling_valid_values:
            raise ValueError(f'pooling must be one of: {pooling_valid_values}')
        if not all_or_none([d_out, pooling, head_activation, head_normalization]):
            raise ValueError(
                'The arguments d_out, pooling, head_activation and head_normalization'
                ' must be either all None or all not-None'
            )
        if not prenormalization:
            if self.WARNINGS['prenormalization']:
                warnings.warn(
                    'prenormalization is set to False. Are you sure about this? '
                    'The training can become less stable. '
                    'You can turn off this warning by tweaking the '
                    'rtdl.nn.Transformer.WARNINGS dictionary.',
                    UserWarning,
                )
            if first_prenormalization:
                raise ValueError(
                    'If prenormalization is False, then first_prenormalization must be False'
                )
        if (
            prenormalization
            and first_prenormalization
            and self.WARNINGS['first_prenormalization']
        ):
            warnings.warn(
                'first_prenormalization is set to True. Are you sure about this? '
                'For example, the vanilla FTTransformer with '
                'first_prenormalization=True performs SIGNIFICANTLY worse. '
                'You can turn off this warning by tweaking the '
                'rtdl.nn.Transformer.WARNINGS dictionary.',
                UserWarning,
            )
            time.sleep(3)

        self.pooling = pooling
        self.pooling_index = None if pooling == 'avg' else 0
        self.cls_embedding = CLSEmbedding(d_embedding) if pooling == 'cls' else None

        # for CLS-based inference, in the last block there is no need to perform
        # computations for any token except for the CLS token
        last_block_pooling_token_only = pooling != 'avg'
        self.blocks = nn.Sequential(
            *[
                Transformer.Block(
                    d_embedding=d_embedding,
                    attention_n_heads=attention_n_heads,
                    attention_dropout=attention_dropout,
                    attention_normalization=(
                        'Identity'
                        if prenormalization
                        and block_idx == 0
                        and not first_prenormalization
                        else attention_normalization
                    ),
                    attention_residual_dropout=attention_residual_dropout,
                    attention_skip_connection=True,
                    linformer_compression_ratio=linformer_compression_ratio,
                    linformer_sharing_policy=linformer_sharing_policy,
                    n_tokens=n_tokens,
                    ffn_d_hidden=ffn_d_hidden,
                    ffn_dropout=ffn_dropout,
                    ffn_activation=ffn_activation,
                    ffn_normalization=ffn_normalization,
                    ffn_residual_dropout=ffn_residual_dropout,
                    ffn_skip_connection=True,
                    prenormalization=prenormalization,
                    pooling_index=(
                        self.pooling_index
                        if last_block_pooling_token_only and block_idx == n_blocks - 1
                        else None
                    ),
                )
                for block_idx in range(n_blocks)
            ]
        )
        self.head = (
            None
            if d_out is None
            else Transformer.Head(
                d_in=d_embedding,
                d_out=d_out,
                bias=True,
                activation=head_activation,  # type: ignore
                normalization=head_normalization if prenormalization else 'Identity',  # type: ignore
            )
        )

    @classmethod
    def make_baseline(
        cls,
        *,
        d_embedding: int,
        d_out: Optional[int],
        n_blocks: int,
        attention_n_heads: int,
        attention_dropout: float,
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: str,
        residual_dropout: float,
        pooling: Optional[Pooling],
        linformer_compression_ratio: Optional[float] = None,
        linformer_sharing_policy: Optional[str] = None,
        n_tokens: Optional[int] = None,
    ) -> 'Transformer':
        """A simplified constructor for building baseline Transformers.

        Features:

        * Head activation is ``ReLU``
        * normalizations are ``LayerNorm``
        * prenormalization is on
        * first prenormalization is off

        Args:
            d_embedding: the size of the embedding dimension
            d_out: the output size of `Transformer.Head`. If `None`, then the output of
                the Transformer will be the output of the last block, i.e. the model will be
                backbone-only.
            n_blocks: the number of blocks
            attention_n_heads: the number of attention heads
            attention_dropout: the dropout rate for attention maps
            ffn_d_hidden: the hidden FFN representation size (formally, the input
                size of the second linear layer of the FFN)
            ffn_dropout: the dropout for the hidden FFN representation
            ffn_activation: the activation used in FFN
            residual_dropout: the dropout rate for outputs of all residual branches
            pooling: the pooling strategy defining how the output of the last block is
                transformed to the input for `Transformer.Head`. If ``'cls'``, then the
                CLS token is used (it is created and appended under the hood, no need to pass it).
                If ``'first_token'``, then the first token is treated as the CLS token.
                If 'avg', the average pooling is used. ``pooling='cls'`` is equivalent
                to placing `rtdl.nn.CLSEmbedding` before the transformer with ``pooling='first_token'``.
            linformer_compression_ratio: the option for the fast linear attention.
                See `rtdl.nn.MultiheadAttention` for details.
            linformer_sharing_policy: the option for the fast linear attention.
                See `rtdl.nn.MultiheadAttention` for details.
            n_tokens: the option for fast linear attention.
                See `rtdl.nn.MultiheadAttention` for details.
        Return:
            transformer

        References:
            * [1] Noam Shazeer, "GLU Variants Improve Transformer"
        """
        normalization = 'LayerNorm'
        return Transformer(
            d_embedding=d_embedding,
            d_out=d_out,
            n_blocks=n_blocks,
            attention_n_heads=attention_n_heads,
            attention_dropout=attention_dropout,
            attention_normalization=normalization,
            attention_residual_dropout=residual_dropout,
            ffn_d_hidden=ffn_d_hidden,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation,
            ffn_normalization=normalization,
            ffn_residual_dropout=residual_dropout,
            prenormalization=True,
            first_prenormalization=False,
            pooling=pooling,
            head_activation='ReLU',
            head_normalization=normalization,
            linformer_compression_ratio=linformer_compression_ratio,
            linformer_sharing_policy=linformer_sharing_policy,
            n_tokens=n_tokens,
        )

    def forward(self, x: Tensor) -> Tensor:
        assert (
            x.ndim == 3
        ), 'The input must have 3 dimensions: (n_objects, n_tokens, d_embedding)'

        if self.cls_embedding is not None:
            x = self.cls_embedding(x)
        x = self.blocks(x)
        if self.pooling == 'avg':
            x = x.mean(1)
        elif self.pooling in ('cls', 'first-token'):
            x = x[:, 0 if x.shape[1] == 1 else self.pooling_index]
        else:
            assert False, INTERNAL_ERROR_MESSAGE
        if self.head is not None:
            x = self.head(x)
        return x
