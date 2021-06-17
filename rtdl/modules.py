import enum
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import functional as rtdlF

ModuleType = Union[str, Callable[..., nn.Module]]


def _is_glu_activation(activation: ModuleType):
    return (
        isinstance(activation, str)
        and activation.endswith('GLU')
        or activation in [ReGLU, GEGLU]
    )


def _all_or_none(values):
    assert all(x is None for x in values) or all(x is not None for x in values)


class _TokenInitialization(enum.Enum):
    UNIFORM = 'uniform'
    NORMAL = 'normal'

    @classmethod
    def from_str(cls, initialization: str) -> '_TokenInitialization':
        try:
            return cls(initialization)
        except ValueError:
            valid_values = [x.value for x in _TokenInitialization]
            raise ValueError(f'initialization must be one of {valid_values}')

    def apply(self, x: Tensor, d: int) -> None:
        d_sqrt_inv = 1 / math.sqrt(d)
        if self == _TokenInitialization.UNIFORM:
            # used in the paper "Revisiting Deep Learning Models for Tabular Data";
            # is equivalent to `nn.init.kaiming_uniform_(x, a=math.sqrt(5))` (which is
            # used by torch to initialize nn.Linear.weight, for example)
            nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
        elif self == _TokenInitialization.NORMAL:
            nn.init.normal_(x, std=d_sqrt_inv)


class NumericalFeatureTokenizer(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_token: int,
        bias: bool,
        initialization: str,
    ) -> None:
        super().__init__()
        initialization_ = _TokenInitialization.from_str(initialization)
        self.weight = nn.Parameter(Tensor(n_features, d_token))
        self.bias = nn.Parameter(Tensor(n_features, d_token)) if bias else None
        for parameter in [self.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_token)

    @property
    def n_tokens(self) -> int:
        return len(self.weight)

    @property
    def d_token(self) -> int:
        return self.weight.shape[1]

    def forward(self, x: Tensor) -> Tensor:
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class CategoricalFeatureTokenizer(nn.Module):
    category_offsets: Tensor

    def __init__(
        self,
        cardinalities: List[int],
        d_token: int,
        bias: bool,
        initialization: str,
    ) -> None:
        super().__init__()
        assert cardinalities
        assert d_token > 0
        initialization_ = _TokenInitialization.from_str(initialization)

        self.category_offsets = torch.tensor([0] + cardinalities[:-1]).cumsum(0)
        self.embeddings = nn.Embedding(sum(cardinalities), d_token)
        self.bias = nn.Parameter(Tensor(len(cardinalities), d_token)) if bias else None

        for parameter in [self.embeddings.weight, self.bias]:
            if parameter is not None:
                initialization_.apply(parameter, d_token)

    @property
    def n_tokens(self) -> int:
        return len(self.category_offsets)

    @property
    def d_token(self) -> int:
        return self.embeddings.embedding_dim

    def forward(self, x: Tensor) -> Tensor:
        x = self.embeddings(x + self.category_offsets[None])
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class FlatEmbedding(nn.Module):
    def __init__(self, *modules: Optional[nn.Module]) -> None:
        super().__init__()
        assert modules
        self.modules_ = nn.ModuleList(
            [nn.Identity() if x is None else x for x in modules]
        )

    def forward(self, *inputs) -> Tensor:
        assert len(self.modules_) == len(inputs)
        return torch.cat(
            [torch.flatten(m(x), 1, -1) for m, x in zip(self.modules_, inputs)],
            dim=1,
        )


class FeatureTokenizer(nn.Module):
    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: Optional[List[int]],
        d_token: int,
    ) -> None:
        super().__init__()
        assert n_num_features >= 0
        assert n_num_features or cat_cardinalities
        self.initialization = 'uniform'
        self.num_tokenizer = (
            NumericalFeatureTokenizer(
                n_features=n_num_features,
                d_token=d_token,
                bias=True,
                initialization=self.initialization,
            )
            if n_num_features
            else None
        )
        self.cat_tokenizer = (
            CategoricalFeatureTokenizer(
                cat_cardinalities, d_token, True, self.initialization
            )
            if cat_cardinalities
            else None
        )

    @property
    def n_tokens(self) -> int:
        return sum(
            x.n_tokens
            for x in [self.num_tokenizer, self.cat_tokenizer]
            if x is not None
        )

    @property
    def d_token(self) -> int:
        return (
            self.cat_tokenizer.d_token  # type: ignore
            if self.num_tokenizer is None
            else self.num_tokenizer.d_token
        )

    def forward(self, x_num: Optional[Tensor], x_cat: Optional[Tensor]) -> Tensor:
        assert x_num is not None or x_cat is not None
        _all_or_none([self.num_tokenizer, x_num])
        _all_or_none([self.cat_tokenizer, x_cat])
        x = []
        if self.num_tokenizer is not None:
            x.append(self.num_tokenizer(x_num))
        if self.cat_tokenizer is not None:
            x.append(self.cat_tokenizer(x_cat))
        return x[0] if len(x) == 1 else torch.cat(x, dim=1)


class AppendCLSToken(nn.Module):
    def __init__(self, d_token: int, initialization: str) -> None:
        super().__init__()
        initialization_ = _TokenInitialization.from_str(initialization)
        self.token = nn.Parameter(Tensor(d_token))
        initialization_.apply(self.token, d_token)

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 3
        return torch.cat([x, self.token.view(1, 1, -1).repeat(len(x), 1, 1)], dim=1)


def _make_nn_module(module_type: ModuleType, *args) -> nn.Module:
    return (
        (
            ReGLU()
            if module_type == 'ReGLU'
            else GEGLU()
            if module_type == 'GEGLU'
            else getattr(nn, module_type)(*args)
        )
        if isinstance(module_type, str)
        else module_type(*args)
    )


class MLP(nn.Module):
    class Block(nn.Module):
        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            activation: ModuleType,
            dropout: float,
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            return self.dropout(self.activation(self.linear(x)))

    def __init__(
        self,
        *,
        d_in: int,
        d_layers: List[int],
        dropouts: Union[float, List[float]],
        activation: Union[str, Callable[[], nn.Module]],
        d_out: Optional[int],
    ) -> None:
        super().__init__()
        assert d_layers or d_out
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
            else nn.Linear(d_layers[-1] if d_layers else d_in, d_out)
        )

    @classmethod
    def make_baseline(
        cls: Type['MLP'],
        d_in: int,
        d_first: Optional[int],
        d_intermidiate: Optional[int],
        d_last: Optional[int],
        n_blocks: int,
        dropout: float,
        d_out: int,
    ) -> 'MLP':
        assert isinstance(dropout, float)
        for (d, n) in [(d_first, 1), (d_last, 2), (d_intermidiate, 3)]:
            assert (n_blocks >= n) ^ (d is None)
        d_layers = []
        if n_blocks >= 1:
            d_layers.append(d_first)
        if n_blocks >= 3:
            d_layers.extend([d_intermidiate] * (n_blocks - 2))
        if n_blocks >= 2:
            d_layers.append(d_last)
        return MLP(
            d_in=d_in,
            d_layers=d_layers,  # type: ignore
            dropouts=dropout,
            activation='ReLU',
            d_out=d_out,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x if self.head is None else self.head(x)


class ResNet(nn.Module):
    class Block(nn.Module):
        def __init__(
            self,
            *,
            d: int,
            d_intermidiate: int,
            bias_first: bool,
            bias_second: bool,
            dropout_first: float,
            dropout_second: float,
            normalization: ModuleType,
            activation: ModuleType,
            skip_connection: bool,
        ) -> None:
            super().__init__()
            self.normalization = _make_nn_module(normalization, d)
            self.linear_first = nn.Linear(d, d_intermidiate, bias_first)
            self.activation = _make_nn_module(activation)
            self.dropout_first = nn.Dropout(dropout_first)
            self.linear_second = nn.Linear(d_intermidiate, d, bias_second)
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
        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            normalization: ModuleType,
            activation: ModuleType,
        ) -> None:
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_in)
            self.activation = _make_nn_module(activation)
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
        n_blocks: int,
        d: Optional[int],
        d_intermidiate: int,
        dropout_first: float,
        dropout_second: float,
        normalization: ModuleType,
        activation: ModuleType,
        d_out: Optional[int],
    ) -> None:
        super().__init__()

        self.first_layer = None if d is None else nn.Linear(d_in, d)
        if d is None:
            d = d_in
        self.blocks = nn.Sequential(
            *[
                ResNet.Block(
                    d=d,
                    d_intermidiate=d_intermidiate,
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
                d_in=d,
                d_out=d_out,
                bias=True,
                normalization=normalization,
                activation=activation,
            )
        )

    @classmethod
    def make_baseline(
        cls: Type['ResNet'],
        *,
        d_in: int,
        d: int,
        d_intermidiate: int,
        dropout_first: float,
        dropout_second: float,
        n_blocks: int,
        d_out: Optional[int],
    ) -> 'ResNet':
        return cls(
            d_in=d_in,
            n_blocks=n_blocks,
            d=d,
            d_intermidiate=d_intermidiate,
            dropout_first=dropout_first,
            dropout_second=dropout_second,
            normalization='BatchNorm1d',
            activation='ReLU',
            d_out=d_out,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.first_layer is not None:
            x = self.first_layer(x)
        x = self.blocks(x)
        if self.head is not None:
            x = self.head(x)
        return x


class ReGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return rtdlF.reglu(x)


class GEGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return rtdlF.geglu(x)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        d_token: int,
        n_heads: int,
        dropout: float,
        bias: bool,
        initialization: str,
    ) -> None:
        super().__init__()
        if n_heads > 1:
            assert d_token % n_heads == 0
        assert initialization in ['kaiming', 'xavier']

        self.W_q = nn.Linear(d_token, d_token, bias)
        self.W_k = nn.Linear(d_token, d_token, bias)
        self.W_v = nn.Linear(d_token, d_token, bias)
        self.W_out = nn.Linear(d_token, d_token, bias) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            # the "xavier" branch tries to follow torch.nn.MultiheadAttention;
            # the second condition checks if W_v plays the role of W_out; the latter one
            # is initialized with Kaiming in torch
            if initialization == 'xavier' and (
                self.W_out is not None or m is not self.W_v
            ):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn.init.zeros_(m.bias)
        if self.W_out is not None:
            nn.init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        key_compression: Optional[nn.Linear],
        value_compression: Optional[nn.Linear],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        _all_or_none([key_compression, value_compression])
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)  # type: ignore

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention_logits = q @ k.transpose(1, 2) / math.sqrt(d_head_key)
        attention_probs = F.softmax(attention_logits, dim=-1)
        if self.dropout is not None:
            attention_probs = self.dropout(attention_probs)
        x = attention_probs @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x, {
            'attention_logits': attention_logits,
            'attention_probs': attention_probs,
        }


class Transformer(nn.Module):
    class FFN(nn.Module):
        def __init__(
            self,
            *,
            d_token: int,
            d_intermidiate: int,
            bias_first: bool,
            bias_second: bool,
            dropout: float,
            activation: ModuleType,
        ):
            super().__init__()
            self.linear_first = nn.Linear(
                d_token,
                d_intermidiate * (2 if _is_glu_activation(activation) else 1),
                bias_first,
            )
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)
            self.linear_second = nn.Linear(d_intermidiate, d_token, bias_second)

        def forward(self, x: Tensor) -> Tensor:
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_second(x)
            return x

    class Head(nn.Module):
        def __init__(
            self,
            *,
            d_in: int,
            bias: bool,
            activation: ModuleType,
            normalization: ModuleType,
            d_out: int,
        ):
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_in)
            self.activation = _make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias)

        def forward(self, x: Tensor) -> Tensor:
            x = x[:, -1].squeeze()
            x = self.normalization(x)
            x = self.activation(x)
            x = self.linear(x)
            return x

    def __init__(
        self,
        *,
        d_token: int,
        n_blocks: int,
        attention_n_heads: int,
        attention_dropout: float,
        attention_initialization: str,
        ffn_d_intermidiate: int,
        ffn_dropout: int,
        ffn_activation: str,
        residual_dropout: float,
        normalization: ModuleType,
        prenormalization: bool,
        first_prenormalization: bool,
        last_layer_query_idx: Union[None, List[int], slice],
        n_tokens: Optional[int],
        kv_compression_ratio: Optional[float],
        kv_compression_sharing: Optional[str],
        head_activation: Optional[ModuleType],
        d_out: Optional[int],
    ) -> None:
        super().__init__()
        _all_or_none([n_tokens, kv_compression_ratio, kv_compression_sharing])
        _all_or_none([head_activation, d_out])
        assert kv_compression_sharing in [None, 'headwise', 'key-value', 'layerwise']

        def make_kv_compression():
            assert kv_compression_ratio and n_tokens
            # https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/examples/linformer/linformer_src/modules/multihead_linear_attention.py#L83
            return nn.Linear(n_tokens, int(n_tokens * kv_compression_ratio), bias=False)

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression_ratio and kv_compression_sharing == 'layerwise'
            else None
        )

        self.prenormalization = prenormalization
        self.last_layer_query_idx = last_layer_query_idx

        self.blocks = nn.ModuleList([])
        for layer_idx in range(n_blocks):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token=d_token,
                        n_heads=attention_n_heads,
                        dropout=attention_dropout,
                        bias=True,
                        initialization=attention_initialization,
                    ),
                    'ffn': self.FFN(
                        d_token=d_token,
                        d_intermidiate=ffn_d_intermidiate,
                        bias_first=True,
                        bias_second=True,
                        dropout=ffn_dropout,
                        activation=ffn_activation,
                    ),
                    'attention_residual_dropout': nn.Dropout(residual_dropout),
                    'ffn_residual_dropout': nn.Dropout(residual_dropout),
                    'output': nn.Identity(),  # for hooks-based introspection
                }
            )
            if layer_idx or not prenormalization or first_prenormalization:
                layer['attention_normalization'] = _make_nn_module(
                    normalization, d_token
                )
            layer['ffn_normalization'] = _make_nn_module(normalization, d_token)
            if kv_compression_ratio and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert kv_compression_sharing == 'key-value'
            self.blocks.append(layer)

        self.head = (
            None
            if d_out is None
            else self.Head(
                d_in=d_token,
                d_out=d_out,
                bias=True,
                activation=head_activation,  # type: ignore
                normalization=normalization if prenormalization else 'Identity',
            )
        )

    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer['key_compression'], layer['value_compression'])
            if 'key_compression' in layer and 'value_compression' in layer
            else (layer['key_compression'], layer['key_compression'])
            if 'key_compression' in layer
            else (None, None)
        )

    def _start_residual(self, layer, stage, x):
        assert stage in ['attention', 'ffn']
        x_residual = x
        if self.prenormalization:
            norm_key = f'{stage}_normalization'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, layer, stage, x, x_residual):
        assert stage in ['attention', 'ffn']
        x_residual = layer[f'{stage}_residual_dropout'](x_residual)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'{stage}_normalization'](x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 3
        for layer_idx, layer in enumerate(self.blocks):
            layer = cast(nn.ModuleDict, layer)

            query_idx = (
                self.last_layer_query_idx if layer_idx + 1 == len(self.blocks) else None
            )
            x_residual = self._start_residual(layer, 'attention', x)
            x_residual, _ = layer['attention'](
                x_residual if query_idx is None else x_residual[:, query_idx],
                x_residual,
                *self._get_kv_compressions(layer),
            )
            if query_idx is not None:
                x = x[:, query_idx]
            x = self._end_residual(layer, 'attention', x, x_residual)

            x_residual = self._start_residual(layer, 'ffn', x)
            x_residual = layer['ffn'](x_residual)
            x = self._end_residual(layer, 'ffn', x, x_residual)
            x = layer['output'](x)

        if self.head is not None:
            x = self.head(x)
        return x


class FTTransformer(nn.Module):
    def __init__(
        self, feature_tokenizer: FeatureTokenizer, transformer: Transformer
    ) -> None:
        super().__init__()
        if transformer.prenormalization:
            assert 'attention_normalization' not in transformer.blocks[0]  # type: ignore
        self.feature_tokenizer = feature_tokenizer
        self.append_cls_token = AppendCLSToken(
            feature_tokenizer.d_token, feature_tokenizer.initialization
        )
        self.transformer = transformer

    @classmethod
    def get_baseline_transformer_subconfig(
        cls: Type['FTTransformer'],
    ) -> Dict[str, Any]:
        return {
            'attention_n_heads': 8,
            'attention_initialization': 'kaiming',
            'ffn_activation': 'ReGLU',
            'normalization': 'LayerNorm',
            'prenormalization': True,
            'first_prenormalization': False,
            'last_layer_query_idx': None,
            'n_tokens': None,
            'kv_compression_ratio': None,
            'kv_compression_sharing': None,
            'head_activation': 'ReLU',
        }

    @classmethod
    def get_default_transformer_config(
        cls: Type['FTTransformer'], *, n_blocks: int = 3
    ) -> Dict[str, Any]:
        assert 1 <= n_blocks <= 6
        grid = {
            'd_token': [96, 128, 192, 256, 320, 384],
            'attention_dropout': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
            'ffn_dropout': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25],
        }
        arch_subconfig = {k: v[n_blocks - 1] for k, v in grid.items()}  # type: ignore
        baseline_subconfig = cls.get_baseline_transformer_subconfig()
        # (4 / 3) for xGLU activations results in almost the same parameter count
        # as (2.0) for element-wise activations (e.g. ReLU; see the "else" branch)
        ffn_d_intermidiate_factor = (
            (4 / 3) if _is_glu_activation(baseline_subconfig['ffn_activation']) else 2.0
        )
        return {
            'n_blocks': n_blocks,
            'residual_dropout': 0.0,
            'ffn_d_intermidiate': int(
                arch_subconfig['d_token'] * ffn_d_intermidiate_factor
            ),
            **arch_subconfig,
            **baseline_subconfig,
        }

    @classmethod
    def _make(
        cls,
        n_num_features,
        cat_cardinalities,
        transformer_config,
    ):
        feature_tokenizer = FeatureTokenizer(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_token=transformer_config['d_token'],
        )
        if transformer_config['d_out'] is None:
            transformer_config['head_activation'] = None
        if transformer_config['kv_compression_ratio'] is not None:
            transformer_config['n_tokens'] = feature_tokenizer.n_tokens + 1
        return FTTransformer(
            feature_tokenizer,
            Transformer(**transformer_config),
        )

    @classmethod
    def make_baseline(
        cls: Type['FTTransformer'],
        *,
        n_num_features: int,
        cat_cardinalities: Optional[List[int]],
        d_token: int,
        n_blocks: int,
        attention_dropout: float,
        ffn_d_intermidiate: int,
        ffn_dropout: int,
        residual_dropout: float,
        last_layer_query_idx: Union[None, List[int], slice] = None,
        kv_compression_ratio: Optional[float] = None,
        kv_compression_sharing: Optional[str] = None,
        d_out: Optional[int],
    ) -> 'FTTransformer':
        transformer_config = cls.get_baseline_transformer_subconfig()
        for arg_name in [
            'n_blocks',
            'attention_dropout',
            'ffn_d_intermidiate',
            'ffn_dropout',
            'residual_dropout',
            'last_layer_query_idx',
            'kv_compression_ratio',
            'kv_compression_sharing',
            'd_out',
        ]:
            transformer_config[arg_name] = locals()[arg_name]
        return cls._make(n_num_features, cat_cardinalities, transformer_config)

    @classmethod
    def make_default(
        cls: Type['FTTransformer'],
        *,
        n_num_features: int,
        cat_cardinalities: Optional[List[int]],
        n_blocks: int = 3,
        last_layer_query_idx: Union[None, List[int], slice] = None,
        kv_compression_ratio: Optional[float] = None,
        kv_compression_sharing: Optional[str] = None,
        d_out: Optional[int],
    ) -> 'FTTransformer':
        transformer_config = cls.get_default_transformer_config(n_blocks=n_blocks)
        for arg_name in [
            'last_layer_query_idx',
            'kv_compression_ratio',
            'kv_compression_sharing',
            'd_out',
        ]:
            transformer_config[arg_name] = locals()[arg_name]
        return cls._make(n_num_features, cat_cardinalities, transformer_config)

    def forward(self, x_num: Optional[Tensor], x_cat: Optional[Tensor]) -> Tensor:
        x = self.feature_tokenizer(x_num, x_cat)
        x = self.append_cls_token(x)
        x = self.transformer(x)
        return x
