# Implementation of "AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks"
# Some differences from a more "conventional" transformer:
# - no FFN module, but one linear layer before adding the result of attention
# - no bias for numerical embeddings
# - no CLS token, the final embedding is formed by concatenation of all the tokens
# - n_heads = 2 is recommended in the paper
# - d_token is supposed to be small
# - the placement of normalizations and activations is different

# %%
import math
import typing as ty
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
import zero
from torch import Tensor

import lib


# %%
class Tokenizer(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        n_latent_tokens: int,
        d_token: int,
    ) -> None:
        super().__init__()
        assert n_latent_tokens == 0
        self.n_latent_tokens = n_latent_tokens
        self.weight = nn.Parameter(Tensor(d_numerical + n_latent_tokens, d_token))
        # The initialization is inspired by nn.Linear
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if categories is None:
            self.category_offsets = None
            self.category_embeddings = None
        else:
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x_num = torch.cat(
            [
                torch.ones(len(x_num), self.n_latent_tokens, device=x_num.device),
                x_num,
            ],
            dim=1,
        )
        x = self.weight[None] * x_num[:, :, None]
        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
        return x


class MultiheadAttention(nn.Module):
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
        self.W_out = None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

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


class AutoInt(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        n_layers: int,
        d_token: int,
        n_heads: int,
        attention_dropout: float,
        residual_dropout: float,
        activation: str,
        prenormalization: bool,
        initialization: str,
        kv_compression: ty.Optional[float],
        kv_compression_sharing: ty.Optional[str],
        d_out: int,
    ) -> None:
        assert not prenormalization
        assert activation == 'relu'
        assert (kv_compression is None) ^ (kv_compression_sharing is not None)

        super().__init__()
        self.tokenizer = Tokenizer(d_numerical, categories, 0, d_token)
        n_tokens = self.tokenizer.n_tokens

        def make_kv_compression():
            assert kv_compression
            compression = nn.Linear(
                n_tokens, int(n_tokens * kv_compression), bias=False
            )
            if initialization == 'xavier':
                nn_init.xavier_uniform_(compression.weight)
            return compression

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression and kv_compression_sharing == 'layerwise'
            else None
        )

        def make_normalization():
            return nn.LayerNorm(d_token)

        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    'linear': nn.Linear(d_token, d_token, bias=False),
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

        self.activation = lib.get_activation_fn(activation)
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token * n_tokens, d_out)

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

        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                x_residual,
                x_residual,
                *self._get_kv_compressions(layer),
            )
            x = layer['linear'](x)
            x = self._end_residual(x, x_residual, layer, 0)
            x = self.activation(x)

        x = x.flatten(1, 2)
        x = self.head(x)
        x = x.squeeze(-1)
        return x


# %%
args, output = lib.load_config()
args['model'].setdefault('kv_compression', None)
args['model'].setdefault('kv_compression_sharing', None)

# %%
zero.set_randomness(args['seed'])
dataset_dir = lib.get_path(args['data']['path'])
stats: ty.Dict[str, ty.Any] = {
    'dataset': dataset_dir.name,
    'algorithm': Path(__file__).stem,
    **lib.load_json(output / 'stats.json'),
}
timer = zero.Timer()
timer.run()

D = lib.Dataset.from_dir(dataset_dir)
X = D.build_X(
    normalization=args['data'].get('normalization'),
    num_nan_policy='mean',
    cat_nan_policy='new',
    cat_policy=args['data'].get('cat_policy', 'counter'),
    cat_min_frequency=args['data'].get('cat_min_frequency', 0.0),
    seed=args['seed'],
)
if not isinstance(X, tuple):
    X = (X, None)
zero.set_randomness(args['seed'])
Y, y_info = D.build_y(args['data'].get('y_policy'))
lib.dump_pickle(y_info, output / 'y_info.pickle')
X = tuple(None if x is None else lib.to_tensors(x) for x in X)
Y = lib.to_tensors(Y)
device = lib.get_device()
if device.type != 'cpu':
    X = tuple(None if x is None else {k: v.to(device) for k, v in x.items()} for x in X)
    Y_device = {k: v.to(device) for k, v in Y.items()}
else:
    Y_device = Y
X_num, X_cat = X
del X
if not D.is_multiclass:
    Y_device = {k: v.float() for k, v in Y_device.items()}

train_size = len(X_num[lib.TRAIN])
batch_size = args['training']['batch_size']
epoch_size = stats['epoch_size'] = math.ceil(train_size / batch_size)
eval_batch_size = args['training']['eval_batch_size']
chunk_size = None

loss_fn = (
    F.binary_cross_entropy_with_logits
    if D.is_binclass
    else F.cross_entropy
    if D.is_multiclass
    else F.mse_loss
)
model = AutoInt(
    d_numerical=X_num[lib.TRAIN].shape[1],
    categories=lib.get_categories(X_cat),
    d_out=D.info['n_classes'] if D.is_multiclass else 1,
    **args['model'],
).to(device)
if torch.cuda.device_count() > 1:  # type: ignore[code]
    print('Using nn.DataParallel')
    model = nn.DataParallel(model)
stats['n_parameters'] = lib.get_n_parameters(model)


def needs_wd(name):
    return all(x not in name for x in ['tokenizer', '.norm', '.bias'])


for x in ['tokenizer', '.norm', '.bias']:
    assert any(x in a for a in (b[0] for b in model.named_parameters()))
parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k)]
parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]
optimizer = lib.make_optimizer(
    args['training']['optimizer'],
    (
        [
            {'params': parameters_with_wd},
            {'params': parameters_without_wd, 'weight_decay': 0.0},
        ]
    ),
    args['training']['lr'],
    args['training']['weight_decay'],
)

stream = zero.Stream(lib.IndexLoader(train_size, batch_size, True, device))
progress = zero.ProgressTracker(args['training']['patience'])
training_log = {lib.TRAIN: [], lib.VAL: [], lib.TEST: []}
timer = zero.Timer()
checkpoint_path = output / 'checkpoint.pt'


def print_epoch_info():
    print(f'\n>>> Epoch {stream.epoch} | {lib.format_seconds(timer())} | {output}')
    print(
        ' | '.join(
            f'{k} = {v}'
            for k, v in {
                'lr': lib.get_lr(optimizer),
                'batch_size': batch_size,
                'chunk_size': chunk_size,
                'epoch_size': stats['epoch_size'],
                'n_parameters': stats['n_parameters'],
            }.items()
        )
    )


def apply_model(part, idx):
    return model(X_num[part][idx], None if X_cat is None else X_cat[part][idx])


@torch.no_grad()
def evaluate(parts):
    global eval_batch_size
    model.eval()
    metrics = {}
    predictions = {}
    for part in parts:
        while eval_batch_size:
            try:
                predictions[part] = (
                    torch.cat(
                        [
                            apply_model(part, idx)
                            for idx in lib.IndexLoader(
                                len(X_num[part]), eval_batch_size, False, device
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
                stats['eval_batch_size'] = eval_batch_size
            else:
                break
        if not eval_batch_size:
            RuntimeError('Not enough memory even for eval_batch_size=1')
        metrics[part] = lib.calculate_metrics(
            D.info['task_type'],
            Y[part].numpy(),  # type: ignore[code]
            predictions[part],  # type: ignore[code]
            'logits',
            y_info,
        )
    for part, part_metrics in metrics.items():
        print(f'[{part:<5}]', lib.make_summary(part_metrics))
    return metrics, predictions


def save_checkpoint(final):
    torch.save(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'stream': stream.state_dict(),
            'random_state': zero.get_random_state(),
            **{
                x: globals()[x]
                for x in [
                    'progress',
                    'stats',
                    'timer',
                    'training_log',
                ]
            },
        },
        checkpoint_path,
    )
    lib.dump_stats(stats, output, final)
    lib.backup_output(output)


# %%
timer.run()
for epoch in stream.epochs(args['training']['n_epochs']):
    print_epoch_info()

    model.train()
    epoch_losses = []
    for batch_idx in epoch:
        loss, new_chunk_size = lib.train_with_auto_virtual_batch(
            optimizer,
            loss_fn,
            lambda x: (apply_model(lib.TRAIN, x), Y_device[lib.TRAIN][x]),
            batch_idx,
            chunk_size or batch_size,
        )
        epoch_losses.append(loss.detach())
        if new_chunk_size and new_chunk_size < (chunk_size or batch_size):
            stats['chunk_size'] = chunk_size = new_chunk_size
            print('New chunk size:', chunk_size)
    epoch_losses = torch.stack(epoch_losses).tolist()
    training_log[lib.TRAIN].extend(epoch_losses)
    print(f'[{lib.TRAIN}] loss = {round(sum(epoch_losses) / len(epoch_losses), 3)}')

    metrics, predictions = evaluate([lib.VAL, lib.TEST])
    for k, v in metrics.items():
        training_log[k].append(v)
    progress.update(metrics[lib.VAL]['score'])

    if progress.success:
        print('New best epoch!')
        stats['best_epoch'] = stream.epoch
        stats['metrics'] = metrics
        save_checkpoint(False)
        for k, v in predictions.items():
            np.save(output / f'p_{k}.npy', v)

    elif progress.fail:
        break


# %%
print('\nRunning the final evaluation...')
model.load_state_dict(torch.load(checkpoint_path)['model'])
stats['metrics'], predictions = evaluate(lib.PARTS)
for k, v in predictions.items():
    np.save(output / f'p_{k}.npy', v)
stats['time'] = lib.format_seconds(timer())
save_checkpoint(True)
print('Done!')
