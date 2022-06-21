from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim
from torch import Tensor

from .._utils import INTERNAL_ERROR_MESSAGE
from ..optim import get_parameter_groups
from ._backbones import Transformer
from ._embeddings import CatEmbeddings, CLSEmbedding, LinearEmbeddings


class _FTTransformerEmbeddings(nn.Module):
    def __init__(
        self, n_num_features: int, cat_cardinalities: List[int], d_embedding: int
    ) -> None:
        if not n_num_features and not cat_cardinalities:
            raise ValueError(
                f'At least one kind of features must be presented. The provided arguments are:'
                f' n_num_features={n_num_features}, cat_cardinalities={cat_cardinalities}.'
            )
        self.num_embeddings = (
            LinearEmbeddings(n_num_features, d_embedding) if n_num_features else None
        )
        self.cat_embeddings = (
            CatEmbeddings(cat_cardinalities, d_embedding, True)
            if cat_cardinalities
            else None
        )

    def forward(self, x_num: Optional[Tensor], x_cat: Optional[Tensor]) -> Tensor:
        x = []
        if x_num is not None:
            if self.num_embeddings is None:
                raise ValueError(
                    'The numerical features were not expected, but they were provided'
                )
            x.append(self.num_embeddings(x_num))
        elif self.num_embeddings is not None:
            raise ValueError(
                'The numerical features were expected, but they were not provided'
            )
        if x_cat is not None:
            if self.cat_embeddings is None:
                raise ValueError(
                    'The categorical features were not expected, but they were provided'
                )
            x.append(self.cat_embeddings(x_cat))
        elif self.cat_embeddings is not None:
            raise ValueError(
                'The categorical features were expected, but they were not provided'
            )

        if len(x) == 1:
            return x[0]
        elif len(x) == 2:
            return torch.cat(x, dim=1)
        else:
            assert False, INTERNAL_ERROR_MESSAGE


_FT_TRANSFORMER_ACTIVATION = 'ReGLU'


def make_ft_transformer(
    *,
    n_num_features: int,
    cat_cardinalities: List[int],
    d_out: Optional[int],
    d_embedding: int,
    n_blocks: int,
    attention_dropout: float,
    ffn_d_hidden: int,
    ffn_dropout: float,
    residual_dropout: float,
    last_block_cls_only: bool = True,
    linformer_compression_ratio: Optional[float] = None,
    linformer_sharing_policy: Optional[str] = None,
) -> nn.Module:
    embeddings = _FTTransformerEmbeddings(
        n_num_features, cat_cardinalities, d_embedding
    )
    backbone = Transformer.make_baseline(
        d_embedding=d_embedding,
        d_out=d_out,
        n_blocks=n_blocks,
        attention_n_heads=8,
        attention_dropout=attention_dropout,
        ffn_d_hidden=ffn_d_hidden,
        ffn_dropout=ffn_dropout,
        activation=_FT_TRANSFORMER_ACTIVATION,
        residual_dropout=residual_dropout,
        pooling='cls',
        cls_token_index=0,
        last_block_cls_only=last_block_cls_only,
        linformer_compression_ratio=linformer_compression_ratio,
        linformer_sharing_policy=linformer_sharing_policy,
        n_tokens=(
            None
            if linformer_compression_ratio is None
            else 1 + n_num_features + len(cat_cardinalities)  # +1 because of CLS
        ),
    )
    return nn.Sequential(embeddings, CLSEmbedding(d_embedding), backbone)


def make_ft_transformer_default(
    *,
    n_num_features: int,
    cat_cardinalities: List[int],
    d_out: Optional[int],
    n_blocks: int = 3,
    last_block_cls_only: bool = True,
    linformer_compression_ratio: Optional[float] = None,
    linformer_sharing_policy: Optional[str] = None,
) -> Tuple[nn.Module, torch.optim.Optimizer]:
    if not (1 <= n_blocks <= 6):
        raise ValueError('n_blocks must be in the range from 1 to 6 (inclusive)')

    from ._backbones import _is_reglu

    default_value_index = n_blocks - 1
    d_embedding = [96, 128, 192, 256, 320, 384][default_value_index]
    ffn_d_hidden_factor = (4 / 3) if _is_reglu(_FT_TRANSFORMER_ACTIVATION) else 2.0
    model = make_ft_transformer(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        d_out=d_out,
        d_embedding=d_embedding,
        n_blocks=n_blocks,
        attention_dropout=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35][default_value_index],
        ffn_dropout=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25][default_value_index],
        ffn_d_hidden=int(d_embedding * ffn_d_hidden_factor),
        residual_dropout=0.0,
        last_block_cls_only=last_block_cls_only,
        linformer_compression_ratio=linformer_compression_ratio,
        linformer_sharing_policy=linformer_sharing_policy,
    )
    optimizer = torch.optim.AdamW(get_parameter_groups(model), 1e-4, weight_decay=1e-5)
    return model, optimizer
