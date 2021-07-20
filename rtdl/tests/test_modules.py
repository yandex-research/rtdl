import torch
from pytest import mark, raises

import rtdl


def test_flat_embeddings():
    d_cat_token = 3
    cardinalities = [2, 3]
    n = 2
    d_num = 4
    assert (
        rtdl.FlatEmbedding(
            None,
            rtdl.CategoricalFeatureTokenizer(
                cardinalities, d_cat_token, True, 'uniform'
            ),
            rtdl.CategoricalFeatureTokenizer(
                cardinalities, d_cat_token, True, 'uniform'
            ),
            None,
        )(
            torch.randn(n, d_num),
            torch.tensor([[0, 2], [1, 1]]),
            torch.tensor([[0, 2], [1, 1]]),
            torch.randn(n, d_num),
        ).shape
        == (n, 2 * (d_num + d_cat_token * len(cardinalities)))
    )


def test_bad_mlp():
    with raises(AssertionError):
        rtdl.MLP.make_baseline(1, [1, 2, 3, 4], 0.0, 1)


@mark.parametrize('n_blocks', range(5))
@mark.parametrize('d_out', [1, 2])
def test_mlp(n_blocks, d_out):
    if not n_blocks and not d_out:
        return
    d = 4
    d_last = d + 1
    d_layers = []
    if n_blocks:
        d_layers.append(d)
    if n_blocks > 2:
        d_layers.extend([d + d_out] * (n_blocks - 2))
    if n_blocks > 1:
        d_layers.append(d_last)
    model = rtdl.MLP.make_baseline(d_in=d, d_layers=d_layers, dropout=0.1, d_out=d_out)
    n = 2
    assert model(torch.randn(n, d)).shape == (
        (n, d_out) if d_out else (n, d_last) if n_blocks > 1 else (n, d)
    )


@mark.parametrize('n_blocks', [1, 2])
@mark.parametrize('d_out', [1, 2])
def test_resnet(n_blocks, d_out):
    d = 4
    model = rtdl.ResNet.make_baseline(
        d_in=d,
        d_main=d,
        d_intermidiate=d * 3,
        dropout_first=0.1,
        dropout_second=0.2,
        n_blocks=n_blocks,
        d_out=d_out,
    )
    n = 2
    assert model(torch.randn(n, d)).shape == ((n, d_out) if d_out else (n, d))


@mark.parametrize('n_blocks', range(1, 7))
@mark.parametrize('d_out', [1, 2])
@mark.parametrize('last_layer_query_idx', [None, [-1]])
def test_ft_transformer(n_blocks, d_out, last_layer_query_idx):
    n_num_features = 4
    model = rtdl.FTTransformer.make_default(
        n_num_features=4,
        cat_cardinalities=[2, 3],
        n_blocks=n_blocks,
        last_layer_query_idx=last_layer_query_idx,
        kv_compression_ratio=0.5,
        kv_compression_sharing='headwise',
        d_out=d_out,
    )
    n = 2
    x = model(torch.randn(n, n_num_features), torch.tensor([[0, 2], [1, 2]]))
    if d_out:
        assert x.shape == (n, d_out)
    else:
        assert x.shape == (
            n,
            model.feature_tokenizer.n_tokens + 1 if last_layer_query_idx is None else 1,
            model.feature_tokenizer.d_token,
        )
