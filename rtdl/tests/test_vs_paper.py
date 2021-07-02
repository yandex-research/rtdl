# Tests in this file validate that the models from `rtdl` are LITERALLY THE SAME as
# the ones used in the paper (https://github.com/yandex-research/rtdl/tree/main/bin)
# The testing approach:
# (1) copy weights from the correct model to the RTDL model
# (2) check that the two models produce the same output for the same input
import math
import random
import typing as ty

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytest import mark
from torch import Tensor

import rtdl


class Model(nn.Module):
    def __init__(self, embedding: rtdl.FlatEmbedding, model: nn.Module):
        super().__init__()
        self.embedding = embedding
        self.model = model

    def forward(self, x_num, x_cat):
        return self.model(self.embedding(x_num, x_cat))


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def copy_layer(dst: nn.Module, src: nn.Module):
    for key, _ in dst.named_parameters():
        getattr(dst, key).copy_(getattr(src, key))


@torch.no_grad()
@mark.parametrize('seed', range(10))
def test_mlp(seed):
    # Source: https://github.com/yandex-research/rtdl/blob/0e5169659c7ce552bc05bbaa85f7e204adc3d88e/bin/mlp.py

    class CorrectMLP(nn.Module):
        def __init__(
            self,
            *,
            d_in: int,
            d_layers: ty.List[int],
            dropout: float,
            d_out: int,
            categories: ty.Optional[ty.List[int]],
            d_embedding: int,
        ) -> None:
            super().__init__()

            if categories is not None:
                d_in += len(categories) * d_embedding
                category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
                self.register_buffer('category_offsets', category_offsets)
                self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
                nn.init.kaiming_uniform_(
                    self.category_embeddings.weight, a=math.sqrt(5)
                )
                print(f'{self.category_embeddings.weight.shape=}')

            self.layers = nn.ModuleList(
                [
                    nn.Linear(d_layers[i - 1] if i else d_in, x)
                    for i, x in enumerate(d_layers)
                ]
            )
            self.dropout = dropout
            self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

        def forward(self, x_num, x_cat):
            if x_cat is not None:
                x_cat = self.category_embeddings(x_cat + self.category_offsets[None])  # type: ignore
                x = torch.cat([x_num, x_cat.view(x_cat.size(0), -1)], dim=-1)
            else:
                x = x_num

            for layer in self.layers:
                x = layer(x)
                x = F.relu(x)
                if self.dropout:
                    x = F.dropout(x, self.dropout, self.training)
            x = self.head(x)
            return x

    n = 32
    d_num = 2
    categories = [2, 3]
    d_embedding = 3
    d_in = d_num + len(categories) * d_embedding
    d_layers = [3, 4, 5]
    dropout = 0.1
    d_out = 2

    set_seeds(seed)
    correct_model = CorrectMLP(
        d_in=d_num,
        d_layers=d_layers,
        dropout=dropout,
        d_out=d_out,
        categories=categories,
        d_embedding=d_embedding,
    )
    rtdl_tokenizer = rtdl.CategoricalFeatureTokenizer(
        categories, d_embedding, False, 'uniform'
    )
    rtdl_backbone = rtdl.MLP(
        d_in=d_in,
        d_layers=d_layers,
        dropouts=dropout,
        activation='ReLU',
        d_out=d_out,
    )

    rtdl_tokenizer.embeddings.weight.copy_(correct_model.category_embeddings.weight)
    for correct_layer, block in zip(correct_model.layers, rtdl_backbone.blocks):
        copy_layer(block.linear, correct_layer)
    copy_layer(rtdl_backbone.head, correct_model.head)

    rtdl_model = Model(rtdl.FlatEmbedding(None, rtdl_tokenizer), rtdl_backbone)
    x_num = torch.randn(n, d_num)
    x_cat = torch.cat([torch.randint(x, (n, 1)) for x in categories], dim=1)
    set_seeds(seed)
    correct_result = correct_model(x_num, x_cat)
    set_seeds(seed)
    rtdl_result = rtdl_model(x_num, x_cat)
    assert (correct_result == rtdl_result).all()


@torch.no_grad()
@mark.parametrize('seed', range(10))
def test_resnet(seed):
    # Source: https://github.com/yandex-research/rtdl/blob/0e5169659c7ce552bc05bbaa85f7e204adc3d88e/bin/resnet.py

    class CorrectResNet(nn.Module):
        def __init__(
            self,
            *,
            d_numerical: int,
            categories: ty.Optional[ty.List[int]],
            d_embedding: int,
            d: int,
            d_hidden_factor: float,
            n_layers: int,
            activation: str,
            normalization: str,
            hidden_dropout: float,
            residual_dropout: float,
            d_out: int,
        ) -> None:
            super().__init__()

            def make_normalization():
                return {'BatchNorm1d': nn.BatchNorm1d}[normalization](d)

            assert activation == 'ReLU'
            self.main_activation = F.relu
            self.last_activation = F.relu
            self.residual_dropout = residual_dropout
            self.hidden_dropout = hidden_dropout

            d_in = d_numerical
            d_hidden = int(d * d_hidden_factor)

            if categories is not None:
                d_in += len(categories) * d_embedding
                category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
                self.register_buffer('category_offsets', category_offsets)
                self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
                nn.init.kaiming_uniform_(
                    self.category_embeddings.weight, a=math.sqrt(5)
                )
                print(f'{self.category_embeddings.weight.shape=}')

            self.first_layer = nn.Linear(d_in, d)
            self.layers = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            'norm': make_normalization(),
                            'linear0': nn.Linear(
                                d, d_hidden * (2 if activation.endswith('glu') else 1)
                            ),
                            'linear1': nn.Linear(d_hidden, d),
                        }
                    )
                    for _ in range(n_layers)
                ]
            )
            self.last_normalization = make_normalization()
            self.head = nn.Linear(d, d_out)

        def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
            if x_cat is not None:
                x_cat = self.category_embeddings(x_cat + self.category_offsets[None])  # type: ignore
                x = torch.cat([x_num, x_cat.view(x_cat.size(0), -1)], dim=-1)  # type: ignore
            else:
                x = x_num

            x = self.first_layer(x)
            for layer in self.layers:
                layer = ty.cast(ty.Dict[str, nn.Module], layer)
                z = x
                z = layer['norm'](z)
                z = layer['linear0'](z)
                z = self.main_activation(z)
                if self.hidden_dropout:
                    z = F.dropout(z, self.hidden_dropout, self.training)
                z = layer['linear1'](z)
                if self.residual_dropout:
                    z = F.dropout(z, self.residual_dropout, self.training)
                x = x + z
            x = self.last_normalization(x)
            x = self.last_activation(x)
            x = self.head(x)
            return x

    n = 32
    d_num = 2
    categories = [2, 3]
    d_embedding = 3
    d_in = d_num + len(categories) * d_embedding
    d = 4
    d_hidden_factor = 1.5
    n_layers = 2
    activation = 'ReLU'
    normalization = 'BatchNorm1d'
    hidden_dropout = 0.1
    residual_dropout = 0.2
    d_out = 2

    set_seeds(seed)
    correct_model = CorrectResNet(
        d_numerical=d_num,
        categories=categories,
        d_embedding=d_embedding,
        d=d,
        d_hidden_factor=d_hidden_factor,
        n_layers=n_layers,
        activation=activation,
        normalization=normalization,
        hidden_dropout=0.1,
        residual_dropout=0.2,
        d_out=d_out,
    )
    rtdl_tokenizer = rtdl.CategoricalFeatureTokenizer(
        categories, d_embedding, False, 'uniform'
    )
    rtdl_backbone = rtdl.ResNet(
        d_in=d_in,
        n_blocks=n_layers,
        d_main=d,
        d_intermidiate=int(d * d_hidden_factor),
        dropout_first=hidden_dropout,
        dropout_second=residual_dropout,
        normalization=normalization,
        activation=activation,
        d_out=d_out,
    )

    rtdl_tokenizer.embeddings.weight.copy_(correct_model.category_embeddings.weight)
    copy_layer(rtdl_backbone.first_layer, correct_model.first_layer)
    for correct_layer, block in zip(correct_model.layers, rtdl_backbone.blocks):
        copy_layer(block.normalization, correct_layer['norm'])
        copy_layer(block.linear_first, correct_layer['linear0'])
        copy_layer(block.linear_second, correct_layer['linear1'])

    copy_layer(rtdl_backbone.head.normalization, correct_model.last_normalization)
    copy_layer(rtdl_backbone.head.linear, correct_model.head)

    rtdl_model = Model(rtdl.FlatEmbedding(None, rtdl_tokenizer), rtdl_backbone)
    x_num = torch.randn(n, d_num)
    x_cat = torch.cat([torch.randint(x, (n, 1)) for x in categories], dim=1)
    set_seeds(seed)
    correct_result = correct_model(x_num, x_cat)
    set_seeds(seed)
    rtdl_result = rtdl_model(x_num, x_cat)
    assert (correct_result == rtdl_result).all()


@torch.no_grad()
@mark.parametrize('seed', range(10))
@mark.parametrize('kv_compression_ratio', [None, 0.5])
def test_ft_transformer(seed, kv_compression_ratio):
    # """Source: https://github.com/yandex-research/rtdl/blob/0e5169659c7ce552bc05bbaa85f7e204adc3d88e/bin/ft_transformer.py"""
    # The only difference is that [CLS] is now the last token.

    def correct_reglu(x: Tensor) -> Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)

    class CorrectTokenizer(nn.Module):
        category_offsets: ty.Optional[Tensor]

        def __init__(
            self,
            d_numerical: int,
            categories: ty.Optional[ty.List[int]],
            d_token: int,
            bias: bool,
        ) -> None:
            super().__init__()
            if categories is None:
                d_bias = d_numerical
                self.category_offsets = None
                self.category_embeddings = None
            else:
                d_bias = d_numerical + len(categories)
                category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
                self.register_buffer('category_offsets', category_offsets)
                self.category_embeddings = nn.Embedding(sum(categories), d_token)
                nn.init.kaiming_uniform_(
                    self.category_embeddings.weight, a=math.sqrt(5)
                )

            # take [CLS] token into account
            self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
            self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
            # The initialization is inspired by nn.Linear
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

        @property
        def n_tokens(self) -> int:
            return len(self.weight) + (
                0 if self.category_offsets is None else len(self.category_offsets)
            )

        def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
            # x_num = torch.cat(
            #     [
            #         torch.ones(len(x_num), 1, device=x_num.device),
            #         x_num,
            #     ],
            #     dim=1,
            # )
            x = self.weight[:-1][None] * x_num[:, :, None]
            if x_cat is not None:
                x = torch.cat(
                    [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                    dim=1,
                )
            x = torch.cat(
                [x, self.weight[-1][None, None].repeat(len(x), 1, 1)],
                dim=1,
            )
            if self.bias is not None:
                bias = torch.cat(
                    [
                        self.bias,
                        torch.zeros(1, self.bias.shape[1], device=x_num.device),
                    ]
                )
                x = x + bias[None]
            return x

    class CorrectMultiheadAttention(nn.Module):
        def __init__(
            self, d: int, n_heads: int, dropout: float, initialization: str
        ) -> None:
            if n_heads > 1:
                assert d % n_heads == 0
            assert initialization in ['xavier', 'kaiming']

            super().__init__()
            self.W_q = nn.Linear(d, d)
            self.W_k = nn.Linear(d, d)
            self.W_v = nn.Linear(d, d)
            self.W_out = nn.Linear(d, d) if n_heads > 1 else None
            self.n_heads = n_heads
            self.dropout = nn.Dropout(dropout) if dropout else None

            for m in [self.W_q, self.W_k, self.W_v]:
                if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
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
            key_compression: ty.Optional[nn.Linear],
            value_compression: ty.Optional[nn.Linear],
        ) -> Tensor:
            q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
            for tensor in [q, k, v]:
                assert tensor.shape[-1] % self.n_heads == 0
            if key_compression is not None:
                assert value_compression is not None
                k = key_compression(k.transpose(1, 2)).transpose(1, 2)
                v = value_compression(v.transpose(1, 2)).transpose(1, 2)
            else:
                assert value_compression is None

            batch_size = len(q)
            d_head_key = k.shape[-1] // self.n_heads
            d_head_value = v.shape[-1] // self.n_heads
            n_q_tokens = q.shape[1]

            q = self._reshape(q)
            k = self._reshape(k)
            attention = F.softmax(q @ k.transpose(1, 2) / math.sqrt(d_head_key), dim=-1)
            if self.dropout is not None:
                attention = self.dropout(attention)
            x = attention @ self._reshape(v)
            x = (
                x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
                .transpose(1, 2)
                .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
            )
            if self.W_out is not None:
                x = self.W_out(x)
            return x

    class CorrectFTTransformer(nn.Module):
        def __init__(
            self,
            *,
            # tokenizer
            d_numerical: int,
            categories: ty.Optional[ty.List[int]],
            token_bias: bool,
            # transformer
            n_layers: int,
            d_token: int,
            n_heads: int,
            d_ffn_factor: float,
            attention_dropout: float,
            ffn_dropout: float,
            residual_dropout: float,
            activation: str,
            prenormalization: bool,
            initialization: str,
            # linformer
            kv_compression: ty.Optional[float],
            kv_compression_sharing: ty.Optional[str],
            #
            d_out: int,
        ) -> None:
            assert (kv_compression is None) ^ (kv_compression_sharing is not None)

            super().__init__()
            self.tokenizer = CorrectTokenizer(
                d_numerical, categories, d_token, token_bias
            )
            n_tokens = self.tokenizer.n_tokens

            def make_kv_compression():
                assert kv_compression
                compression = nn.Linear(
                    n_tokens, int(n_tokens * kv_compression), bias=False
                )
                if initialization == 'xavier':
                    nn.init.xavier_uniform_(compression.weight)
                return compression

            self.shared_kv_compression = (
                make_kv_compression()
                if kv_compression and kv_compression_sharing == 'layerwise'
                else None
            )

            def make_normalization():
                return nn.LayerNorm(d_token)

            d_hidden = int(d_token * d_ffn_factor)
            self.layers = nn.ModuleList([])
            for layer_idx in range(n_layers):
                layer = nn.ModuleDict(
                    {
                        'attention': CorrectMultiheadAttention(
                            d_token, n_heads, attention_dropout, initialization
                        ),
                        'linear0': nn.Linear(
                            d_token, d_hidden * (2 if activation.endswith('glu') else 1)
                        ),
                        'linear1': nn.Linear(d_hidden, d_token),
                        'norm1': make_normalization(),
                    }
                )
                if not prenormalization or layer_idx:
                    layer['norm0'] = make_normalization()
                if kv_compression and self.shared_kv_compression is None:
                    layer['key_compression'] = make_kv_compression()
                    if kv_compression_sharing == 'headwise':
                        layer['value_compression'] = make_kv_compression()
                    else:
                        assert kv_compression_sharing == 'key-value'
                self.layers.append(layer)

            assert activation == 'reglu'
            self.activation = correct_reglu
            self.last_activation = F.relu
            self.prenormalization = prenormalization
            self.last_normalization = make_normalization() if prenormalization else None
            self.ffn_dropout = ffn_dropout
            self.residual_dropout = residual_dropout
            self.head = nn.Linear(d_token, d_out)

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

        def _start_residual(self, x, layer, norm_idx):
            x_residual = x
            if self.prenormalization:
                norm_key = f'norm{norm_idx}'
                if norm_key in layer:
                    x_residual = layer[norm_key](x_residual)
            return x_residual

        def _end_residual(self, x, x_residual, layer, norm_idx):
            if self.residual_dropout:
                x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
            x = x + x_residual
            if not self.prenormalization:
                x = layer[f'norm{norm_idx}'](x)
            return x

        def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
            x = self.tokenizer(x_num, x_cat)

            for layer_idx, layer in enumerate(self.layers):
                is_last_layer = layer_idx + 1 == len(self.layers)
                layer = ty.cast(ty.Dict[str, nn.Module], layer)

                x_residual = self._start_residual(x, layer, 0)
                x_residual = layer['attention'](
                    # for the last attention, it is enough to process only [CLS]
                    (x_residual[:, -1:] if is_last_layer else x_residual),
                    x_residual,
                    *self._get_kv_compressions(layer),
                )
                if is_last_layer:
                    x = x[:, -1:]
                x = self._end_residual(x, x_residual, layer, 0)

                x_residual = self._start_residual(x, layer, 1)
                x_residual = layer['linear0'](x_residual)
                x_residual = self.activation(x_residual)
                if self.ffn_dropout:
                    x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
                x_residual = layer['linear1'](x_residual)
                x = self._end_residual(x, x_residual, layer, 1)

            assert x.shape[1] == 1
            x = x[:, 0]
            if self.last_normalization is not None:
                x = self.last_normalization(x)
            x = self.last_activation(x)
            x = self.head(x)
            x = x.squeeze(-1)
            return x

    # Source: https://github.com/yandex-research/rtdl/blob/0e5169659c7ce552bc05bbaa85f7e204adc3d88e/output/california_housing/ft_transformer/default/0.toml
    default_config = {
        'seed': 0,
        'data': {
            'normalization': 'quantile_normal',
            'path': 'data/california_housing',
            'y_policy': 'mean_std',
        },
        'model': {
            'activation': 'reglu',
            'attention_dropout': 0.2,
            'd_ffn_factor': 4 / 3,
            'd_token': 192,
            'ffn_dropout': 0.1,
            'initialization': 'kaiming',
            'n_heads': 8,
            'n_layers': 3,
            'prenormalization': True,
            'residual_dropout': 0.0,
        },
        'training': {
            'batch_size': 256,
            'eval_batch_size': 8192,
            'lr': 0.0001,
            'lr_n_decays': 0,
            'n_epochs': 1000000000,
            'optimizer': 'adamw',
            'patience': 16,
            'weight_decay': 1e-05,
        },
    }
    dcm = default_config['model']

    n = 4
    d_num = 2
    categories = [2, 3]
    n_tokens = d_num + len(categories) + 1
    kv_compression_sharing = 'key-value' if kv_compression_ratio else None
    d_out = 2

    set_seeds(seed)
    correct_model = CorrectFTTransformer(
        d_numerical=d_num,
        categories=categories,
        token_bias=True,
        kv_compression=kv_compression_ratio,
        kv_compression_sharing=kv_compression_sharing,
        d_out=d_out,
        **dcm,
    )
    rtdl_model = rtdl.FTTransformer(
        rtdl.FeatureTokenizer(d_num, categories, dcm['d_token']),
        rtdl.Transformer(
            d_token=dcm['d_token'],
            n_blocks=dcm['n_layers'],
            attention_n_heads=dcm['n_heads'],
            attention_dropout=dcm['attention_dropout'],
            attention_initialization=dcm['initialization'],
            ffn_d_intermidiate=int(dcm['d_token'] * dcm['d_ffn_factor']),
            ffn_dropout=dcm['ffn_dropout'],
            ffn_activation='ReGLU',
            residual_dropout=dcm['residual_dropout'],
            normalization='LayerNorm',
            prenormalization=dcm['prenormalization'],
            first_prenormalization=False,
            last_layer_query_idx=[-1],
            n_tokens=n_tokens if kv_compression_ratio else None,
            kv_compression_ratio=kv_compression_ratio,
            kv_compression_sharing=kv_compression_sharing,
            head_activation='ReLU',
            d_out=d_out,
        ),
    )
    rtdl_default_model = rtdl.FTTransformer.make_default(
        n_num_features=d_num,
        cat_cardinalities=categories,
        last_layer_query_idx=[-1],
        kv_compression_ratio=kv_compression_ratio,
        kv_compression_sharing=kv_compression_sharing,
        d_out=d_out,
    )

    rtdl_model.feature_tokenizer.num_tokenizer.weight.copy_(
        correct_model.tokenizer.weight[:-1]
    )
    rtdl_model.feature_tokenizer.num_tokenizer.bias.copy_(
        correct_model.tokenizer.bias[:d_num]
    )
    rtdl_model.feature_tokenizer.cat_tokenizer.embeddings.weight.copy_(
        correct_model.tokenizer.category_embeddings.weight
    )
    rtdl_model.feature_tokenizer.cat_tokenizer.bias.copy_(
        correct_model.tokenizer.bias[-len(categories) :]
    )
    rtdl_model.append_cls_token.weight.copy_(correct_model.tokenizer.weight[-1])
    for correct_layer, block in zip(
        correct_model.layers, rtdl_model.transformer.blocks
    ):
        for key in ['W_q', 'W_k', 'W_v', 'W_out']:
            copy_layer(
                getattr(block['attention'], key),
                getattr(correct_layer['attention'], key),
            )
        copy_layer(block['ffn'].linear_first, correct_layer['linear0'])
        copy_layer(block['ffn'].linear_second, correct_layer['linear1'])
        copy_layer(block['ffn_normalization'], correct_layer['norm1'])
        if 'norm0' in correct_layer:
            copy_layer(block['attention_normalization'], correct_layer['norm0'])
        for key in ['key_compression', 'value_compression']:
            if key in correct_layer:
                copy_layer(block[key], correct_layer[key])
    copy_layer(
        rtdl_model.transformer.head.normalization, correct_model.last_normalization
    )
    copy_layer(rtdl_model.transformer.head.linear, correct_model.head)
    rtdl_default_model.load_state_dict(rtdl_model.state_dict())

    x_num = torch.randn(n, d_num)
    x_cat = torch.cat([torch.randint(x, (n, 1)) for x in categories], dim=1)
    results = []
    for m in [correct_model, rtdl_model, rtdl_default_model]:
        set_seeds(seed)
        results.append(m(x_num, x_cat))
    correct_result = results[0]
    assert (results[1] == results[2]).all()
    assert (results[1] == correct_result).all()
