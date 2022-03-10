# %%
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, Union

import numpy as np
import rtdl
import torch
import torch.nn as nn
import zero
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tqdm import trange

import lib


# %%
@dataclass
class Config:
    @dataclass
    class Data:
        path: str
        T: lib.Transformations = field(default_factory=lib.Transformations)
        T_cache: bool = False

    @dataclass
    class Bins:
        count: int
        tree: Optional[dict[str, Any]] = None
        subsample: Union[None, int, float] = None

    @dataclass
    class Model:
        d_num_embedding: Optional[int] = None
        num_embedding_arch: list[str] = field(default_factory=list)
        d_cat_embedding: Union[None, int, Literal['d_num_embedding']] = None
        mlp: Optional[dict[str, Any]] = None
        resnet: Optional[dict[str, Any]] = None
        transformer: Optional[dict[str, Any]] = None
        transformer_default: bool = False
        transformer_baseline: bool = True
        # The following parameter is purely technical and does not affect the "algorithm".
        # Setting it to False leads to much better speed at the cost of high peak memory
        # consumption in the case when 'linear' layers are presented in `num_embedding_arch`.
        memory_efficient: bool = False

    @dataclass
    class Training:
        batch_size: int
        lr: float
        weight_decay: float
        optimizer: str = 'AdamW'
        patience: Optional[int] = 16
        n_epochs: Union[int, float] = math.inf
        eval_batch_size: int = 8192

    seed: int
    data: Data
    model: Model
    training: Training
    bins: Optional[Bins] = None

    @property
    def is_mlp(self):
        return self.model.mlp is not None

    @property
    def is_resnet(self):
        return self.model.resnet is not None

    @property
    def is_transformer(self):
        return self.model.transformer is not None

    def __post_init__(self):
        assert sum([self.is_mlp, self.is_resnet, self.is_transformer]) == 1
        if self.model.d_num_embedding is not None:
            assert (
                self.model.num_embedding_arch
                and self.model.num_embedding_arch[0] == 'linear'
            )
        if self.is_resnet:
            lib.replace_factor_with_value(
                self.model.resnet,
                'd_hidden',
                self.model.resnet['d_main'],
                (1.0, 8.0),
            )
        if self.is_transformer and not self.model.transformer_default:
            assert self.model.d_num_embedding is not None
            lib.replace_factor_with_value(
                self.model.transformer,
                'ffn_d_hidden',
                self.model.d_num_embedding,
                (0.5, 4.0),
            )


class NLinear(nn.Module):
    def __init__(self, n: int, d_in: int, d_out: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(n, d_in, d_out))
        self.bias = nn.Parameter(torch.Tensor(n, d_out))
        with torch.no_grad():
            for i in range(n):
                layer = nn.Linear(d_in, d_out)
                self.weight[i] = layer.weight.T
                self.bias[i] = layer.bias

    def forward(self, x):
        x = x[..., None] * self.weight[None]
        x = x.sum(-2)
        x = x + self.bias[None]
        return x


class NLinearMemoryEfficient(nn.Module):
    def __init__(self, n: int, d_in: int, d_out: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for _ in range(n)])

    def forward(self, x):
        return torch.stack([l(x[:, i]) for i, l in enumerate(self.layers)], 1)


class NumEmbeddings(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_embedding: int,
        embedding_arch: list[str],
        d_feature: Optional[int],
        memory_efficient: bool,
    ) -> None:
        super().__init__()
        assert set(embedding_arch) <= {'relu', 'linear', 'shared_linear'}
        assert embedding_arch
        assert embedding_arch[0] == 'linear'
        NLinear_ = NLinearMemoryEfficient if memory_efficient else NLinear
        layers: list[nn.Module] = [
            rtdl.NumericalFeatureTokenizer(n_features, d_embedding, True, 'uniform')
            if d_feature is None
            else NLinear_(n_features, d_feature, d_embedding)
        ]
        for x in embedding_arch[1:]:
            layers.append(
                nn.ReLU()
                if x == 'relu'
                else NLinear_(n_features, d_embedding, d_embedding)
                if x == 'linear'
                else nn.Linear(d_embedding, d_embedding)
                if x == 'shared_linear'
                else None
            )
            assert layers[-1] is not None
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class BaseModel(nn.Module):
    def __init__(self, config: Config, dataset: lib.Dataset, n_bins: Optional[int]):
        super().__init__()
        self.num_embeddings = (
            None
            if config.model.d_num_embedding is None
            else NumEmbeddings(
                D.n_num_features,
                config.model.d_num_embedding,
                config.model.num_embedding_arch,
                n_bins,
                config.model.memory_efficient,
            )
        )
        self.category_sizes = dataset.get_category_sizes('train')
        d_cat_embedding = (
            config.model.d_num_embedding
            if self.category_sizes
            and (
                config.is_transformer
                or config.model.d_cat_embedding == 'd_num_embedding'
            )
            else config.model.d_cat_embedding
        )
        self.cat_embeddings = (
            None
            if d_cat_embedding is None
            else rtdl.CategoricalFeatureTokenizer(
                self.category_sizes, d_cat_embedding, True, 'uniform'
            )
        )

    def _encode_input(self, x_num, x_cat):
        if self.num_embeddings is not None:
            assert x_num is not None
            x_num = self.num_embeddings(x_num)
        if self.cat_embeddings is not None:
            assert x_cat is not None
            x_cat = self.cat_embeddings(x_cat)
        elif x_cat is not None:
            x_cat = torch.concat(
                [
                    nn.functional.one_hot(x_cat[:, i], category_size)
                    for i, category_size in enumerate(self.category_sizes)
                ],
                1,
            )
        return x_num, x_cat


class FlatModel(BaseModel):
    def __init__(self, config, dataset, n_bins):
        super().__init__(config, dataset, n_bins)
        in_out_options = {
            'd_in': (
                D.n_num_features * (config.model.d_num_embedding or n_bins or 1)
                + (
                    sum(self.category_sizes)
                    if config.model.d_cat_embedding is None
                    else len(self.category_sizes) * config.model.d_cat_embedding
                )
            ),
            'd_out': dataset.nn_output_dim,
        }
        if config.is_mlp:
            self.main_module = rtdl.MLP.make_baseline(
                **config.model.mlp, **in_out_options
            )
        elif config.is_resnet:
            self.main_module = rtdl.ResNet.make_baseline(
                **config.model.resnet, **in_out_options
            )
        else:
            assert False

    def forward(self, x_num, x_cat):
        assert x_num is not None or x_cat is not None
        x_num, x_cat = self._encode_input(x_num, x_cat)
        x = torch.concat(
            [
                (x_.flatten(1, 2) if x_.ndim == 3 else x_)
                for x_ in [x_num, x_cat]
                if x_ is not None
            ],
            1,
        )
        return self.main_module(x)


class NonFlatModel(BaseModel):
    def __init__(self, config, dataset, n_bins):
        super().__init__(config, dataset, n_bins)
        assert config.model.transformer is not None
        transformer_options = deepcopy(config.model.transformer)
        if config.model.transformer_default:
            transformer_options = (
                rtdl.FTTransformer.get_default_transformer_config(
                    n_blocks=transformer_options.get('n_blocks', 3)
                )
                | transformer_options
            )
            del transformer_options["d_token"]
        elif config.model.transformer_baseline:
            transformer_options = (
                rtdl.FTTransformer.get_baseline_transformer_subconfig()
                | transformer_options
            )
        d_cat_embedding = (
            config.model.d_num_embedding
            if self.category_sizes
            else config.model.d_cat_embedding
        )
        d_embedding = config.model.d_num_embedding or d_cat_embedding
        assert d_embedding is not None
        self.cls_embedding = rtdl.CLSToken(d_embedding, 'uniform')
        self.main_module = rtdl.Transformer(
            d_token=d_embedding,
            **transformer_options,
            d_out=dataset.nn_output_dim,
        )

    def forward(self, x_num, x_cat):
        assert x_num is not None or x_cat is not None
        x_num, x_cat = self._encode_input(x_num, x_cat)
        for x in x_num, x_cat:
            if x is not None:
                assert x.ndim == 3
        x = torch.concat([x_ for x_ in [x_num, x_cat] if x_ is not None], 1)
        x = self.cls_embedding(x)
        return self.main_module(x)


# %%
def patch_raw_config(raw_config):
    # Before this option was introduced, the code was always "memory efficient"
    raw_config['model'].setdefault('memory_efficient', True)


C, output, report = lib.start(Config, patch_raw_config=patch_raw_config)

# %%
zero.improve_reproducibility(C.seed)
device = lib.get_device()
D = lib.build_dataset(C.data.path, C.data.T, C.data.T_cache)
report['prediction_type'] = None if D.is_regression else 'logits'
report['epoch_size'] = math.ceil(D.size('train') / C.training.batch_size)
lib.dump_pickle(D.y_info, output / 'y_info.pickle')

if C.bins is None:
    bin_edges = None
    bins = None
    ratios = None
    n_bins = None
else:
    print('\nRunning bin-based encoding...')
    assert D.X_num is not None
    bin_edges = []
    _bins = {x: [] for x in D.X_num}
    _ratios = {x: [] for x in D.X_num}
    rng = np.random.default_rng(C.seed)
    for feature_idx in trange(D.n_num_features):
        train_column = D.X_num['train'][:, feature_idx].copy()
        if C.bins.subsample is not None:
            subsample_n = (
                C.bins.subsample
                if isinstance(C.bins.subsample, int)
                else int(C.bins.subsample * D.size('train'))
            )
            subsample_idx = rng.choice(len(train_column), subsample_n, replace=False)
            train_column = train_column[subsample_idx]
        else:
            subsample_idx = None

        if C.bins.tree is not None:
            _y = D.y['train']
            if subsample_idx is not None:
                _y = _y[subsample_idx]
            tree = (
                (DecisionTreeRegressor if D.is_regression else DecisionTreeClassifier)(
                    max_leaf_nodes=C.bins.count, **C.bins.tree
                )
                .fit(train_column.reshape(-1, 1), D.y['train'])
                .tree_
            )
            del _y
            tree_thresholds = []
            for node_id in range(tree.node_count):
                # the following condition is True only for split nodes
                # See https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
                if tree.children_left[node_id] != tree.children_right[node_id]:
                    tree_thresholds.append(tree.threshold[node_id])
            tree_thresholds.append(train_column.min())
            tree_thresholds.append(train_column.max())
            bin_edges.append(np.array(sorted(set(tree_thresholds))))
        else:
            feature_n_bins = min(C.bins.count, len(np.unique(train_column)))
            quantiles = np.linspace(
                0.0, 1.0, feature_n_bins + 1
            )  # includes 0.0 and 1.0
            bin_edges.append(np.unique(np.quantile(train_column, quantiles)))

        feature_bin_sizes = bin_edges[feature_idx][1:] - bin_edges[feature_idx][:-1]
        for part in D.X_num:
            _bins[part].append(
                np.digitize(
                    D.X_num[part][:, feature_idx],
                    np.r_[-np.inf, bin_edges[feature_idx][1:-1], np.inf],
                ).astype(np.int32)
                - 1
            )
            part_feature_bins = _bins[part][feature_idx]
            _ratios[part].append(
                (
                    D.X_num[part][:, feature_idx]
                    - bin_edges[feature_idx][part_feature_bins]
                )
                / feature_bin_sizes[part_feature_bins]
            )

    n_bins = max(map(len, bin_edges)) - 1

    bins = {
        k: torch.as_tensor(np.stack(v, axis=1), dtype=torch.int64, device=device)
        for k, v in _bins.items()
    }
    del _bins
    ratios = {
        k: torch.as_tensor(np.stack(v, axis=1), dtype=torch.float32, device=device)
        for k, v in _ratios.items()
    }
    del _ratios
    lib.dump_pickle(bin_edges, output / 'bin_edges.pickle')
    print()

X_num, X_cat, Y = lib.prepare_tensors(D, device=device)
if C.bins is not None:
    X_num = None
zero.hardware.free_memory()
loss_fn = lib.get_loss_fn(D.task_type)
model = (FlatModel if C.is_mlp or C.is_resnet else NonFlatModel)(C, D, n_bins).to(
    device
)
if torch.cuda.device_count() > 1:
    print('Using nn.DataParallel')
    model = nn.DataParallel(model)  # type: ignore[code]
report['n_parameters'] = lib.get_n_parameters(model)
optimizer = lib.make_optimizer(
    asdict(C.training), lib.split_parameters_by_weight_decay(model)
)

stream = zero.Stream(
    zero.data.IndexLoader(D.size('train'), C.training.batch_size, True, device=device)
)
progress = zero.ProgressTracker(C.training.patience)
training_log = {}
checkpoint_path = output / 'checkpoint.pt'
eval_batch_size = C.training.eval_batch_size
chunk_size = None


def print_epoch_info():
    print(f'\n>>> Epoch {stream.epoch} | {timer} | {output}')
    print(
        ' | '.join(
            f'{k} = {v}'
            for k, v in {
                'lr': lib.get_lr(optimizer),
                'batch_size': C.training.batch_size,  # type: ignore[code]
                'chunk_size': chunk_size,
                'epoch_size': report['epoch_size'],
                'n_parameters': report['n_parameters'],
            }.items()
        )
    )


def encode(part, idx):
    bins_ = bins[part][idx]
    bin_mask_ = torch.eye(n_bins, device=device)[bins_]
    x = bin_mask_ * ratios[part][idx, ..., None]
    previous_bins_mask = torch.arange(n_bins, device=device)[None, None].repeat(
        len(idx), D.n_num_features, 1
    ) < bins_.reshape(len(idx), D.n_num_features, 1)
    x[previous_bins_mask] = 1.0
    return x


def apply_model(part, idx):
    return model(
        (
            X_num[part][idx]
            if X_num is not None
            else encode(part, idx)
            if C.bins is not None
            else None
        ),
        None if X_cat is None else X_cat[part][idx],
    ).squeeze(-1)


@torch.inference_mode()
def evaluate(parts):
    global eval_batch_size
    model.eval()
    predictions = {}
    for part in parts:
        while eval_batch_size:
            try:
                predictions[part] = (
                    torch.cat(
                        [
                            apply_model(part, idx)
                            for idx in zero.data.IndexLoader(
                                D.size(part), eval_batch_size, False, device=device
                            )
                        ]
                    )
                    .cpu()
                    .numpy()
                )
            except RuntimeError as err:
                if not lib.is_oom_exception(err):
                    raise
                eval_batch_size //= 2
                print('New eval batch size:', eval_batch_size)
                report['eval_batch_size'] = eval_batch_size
            else:
                break
        if not eval_batch_size:
            RuntimeError('Not enough memory even for eval_batch_size=1')
    return D.calculate_metrics(predictions, report['prediction_type']), predictions


def save_checkpoint():
    torch.save(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'stream': stream.state_dict(),
            'random_state': zero.random.get_state(),
            **{
                x: globals()[x]
                for x in [
                    'progress',
                    'report',
                    'timer',
                    'training_log',
                ]
            },
        },
        checkpoint_path,
    )
    lib.dump_report(report, output)
    lib.backup_output(output)


# %%
timer = lib.Timer.launch()
for epoch in stream.epochs(C.training.n_epochs):
    print_epoch_info()

    model.train()
    epoch_losses = []
    for batch_idx in epoch:
        loss, new_chunk_size = lib.train_with_auto_virtual_batch(
            optimizer,
            loss_fn,
            lambda x: (apply_model('train', x), Y['train'][x]),
            batch_idx,
            chunk_size or C.training.batch_size,
        )
        epoch_losses.append(loss.detach())
        if new_chunk_size and new_chunk_size < (chunk_size or C.training.batch_size):
            report['chunk_size'] = chunk_size = new_chunk_size
            print('New chunk size:', chunk_size)

    epoch_losses, mean_loss = lib.process_epoch_losses(epoch_losses)
    metrics, predictions = evaluate(['val', 'test'])
    lib.update_training_log(
        training_log,
        {
            'train_loss': epoch_losses,
            'mean_train_loss': mean_loss,
            'time': timer(),
        },
        metrics,
    )
    print(f'\n{lib.format_scores(metrics)} [loss] {mean_loss:.3f}')

    progress.update(metrics['val']['score'])
    if progress.success:
        print('New best epoch!')
        report['best_epoch'] = stream.epoch
        report['metrics'] = metrics
        save_checkpoint()
        lib.dump_predictions(predictions, output)

    elif progress.fail:
        break

# %%
model.load_state_dict(torch.load(checkpoint_path)['model'])
report['metrics'], predictions = evaluate(['train', 'val', 'test'])
lib.dump_predictions(predictions, output)
report['time'] = str(timer)
save_checkpoint()
lib.finish(output, report)
