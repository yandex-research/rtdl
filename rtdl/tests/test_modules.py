import pytest
import torch

import rtdl


def get_devices():
    return ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']


def test_bad_mlp():
    with pytest.raises(AssertionError):
        rtdl.MLP.make_baseline(1, [1, 2, 3, 4], 0.0, 1)


@pytest.mark.parametrize('n_blocks', range(5))
@pytest.mark.parametrize('d_out', [1, 2])
@pytest.mark.parametrize('constructor', range(2))
@pytest.mark.parametrize('device', get_devices())
def test_mlp(n_blocks, d_out, constructor, device):
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

    def f0():
        dropouts = [0.1 * x for x in range(len(d_layers))]
        return rtdl.MLP(
            d_in=d, d_layers=d_layers, dropouts=dropouts, activation='GELU', d_out=d_out
        )

    def f1():
        return rtdl.MLP.make_baseline(
            d_in=d, d_layers=d_layers, dropout=0.1, d_out=d_out
        )

    model = locals()[f'f{constructor}']().to(device)
    n = 2
    assert model(torch.randn(n, d, device=device)).shape == (
        (n, d_out) if d_out else (n, d_last) if n_blocks > 1 else (n, d)
    )


@pytest.mark.parametrize('n_blocks', [1, 2])
@pytest.mark.parametrize('d_out', [1, 2])
@pytest.mark.parametrize('constructor', range(2))
@pytest.mark.parametrize('device', get_devices())
def test_resnet(n_blocks, d_out, constructor, device):
    d = 4

    def f0():
        return rtdl.ResNet.make_baseline(
            d_in=d,
            d_main=d,
            d_hidden=d * 3,
            dropout_first=0.1,
            dropout_second=0.2,
            n_blocks=n_blocks,
            d_out=d_out,
        )

    def f1():
        return rtdl.ResNet(
            d_in=d,
            d_main=d,
            d_hidden=d * 3,
            dropout_first=0.1,
            dropout_second=0.2,
            n_blocks=n_blocks,
            normalization='Identity',
            activation='ReLU6',
            d_out=d_out,
        )

    model = locals()[f'f{constructor}']().to(device)
    n = 2
    assert model(torch.randn(n, d, device=device)).shape == (
        (n, d_out) if d_out else (n, d)
    )


@pytest.mark.parametrize('n_blocks', range(1, 7))
@pytest.mark.parametrize('d_out', [1, 2])
@pytest.mark.parametrize('last_layer_query_idx', [None, [-1]])
@pytest.mark.parametrize('constructor', range(3))
@pytest.mark.parametrize('device', get_devices())
def test_ft_transformer(n_blocks, d_out, last_layer_query_idx, constructor, device):
    n_num_features = 4
    model = rtdl.FTTransformer.make_default(
        n_num_features=4,
        cat_cardinalities=[2, 3],
        n_blocks=n_blocks,
        last_layer_query_idx=last_layer_query_idx,
        kv_compression_ratio=0.5,
        kv_compression_sharing='headwise',
        d_out=d_out,
    ).to(device)
    n = 2

    # check that the following methods do not fail
    model.optimization_param_groups()
    model.make_default_optimizer()

    def f0():
        return rtdl.FTTransformer.make_default(
            n_num_features=4,
            cat_cardinalities=[2, 3],
            n_blocks=n_blocks,
            last_layer_query_idx=last_layer_query_idx,
            kv_compression_ratio=0.5,
            kv_compression_sharing='headwise',
            d_out=d_out,
        )

    def f1():
        return rtdl.FTTransformer.make_baseline(
            n_num_features=4,
            cat_cardinalities=[2, 3],
            n_blocks=n_blocks,
            d_token=8,
            attention_dropout=0.2,
            ffn_d_hidden=8,
            ffn_dropout=0.3,
            residual_dropout=0.4,
            last_layer_query_idx=last_layer_query_idx,
            kv_compression_ratio=0.5,
            kv_compression_sharing='headwise',
            d_out=d_out,
        )

    def f2():
        d_token = 8
        rtdl.Transformer.WARNINGS['prenormalization'] = False
        model = rtdl.FTTransformer(
            rtdl.FeatureTokenizer(4, [2, 3], d_token),
            rtdl.Transformer(
                d_token=d_token,
                attention_n_heads=1,
                attention_dropout=0.3,
                attention_initialization='xavier',
                attention_normalization='Identity',
                ffn_d_hidden=4,
                ffn_dropout=0.3,
                ffn_activation='SELU',
                residual_dropout=0.2,
                ffn_normalization='Identity',
                prenormalization=False,
                first_prenormalization=False,
                n_tokens=7,
                head_activation='PReLU',
                head_normalization='Identity',
                n_blocks=n_blocks,
                last_layer_query_idx=last_layer_query_idx,
                kv_compression_ratio=0.5,
                kv_compression_sharing='headwise',
                d_out=d_out,
            ),
        )
        rtdl.Transformer.WARNINGS['prenormalization'] = True
        return model

    model = locals()[f'f{constructor}']().to(device)
    x = model(
        torch.randn(n, n_num_features, device=device),
        torch.tensor([[0, 2], [1, 2]], device=device),
    )
    if d_out:
        assert x.shape == (n, d_out)
    else:
        assert x.shape == (
            n,
            model.feature_tokenizer.n_tokens + 1 if last_layer_query_idx is None else 1,
            model.feature_tokenizer.d_token,
        )
