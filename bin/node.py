# %%
import itertools
import math
import typing as ty
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.swa_utils as swa_utils
import zero
from torch import Tensor

import lib
import lib.node as node


# %%
class NODE(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        num_layers: int,
        layer_dim: int,
        depth: int,
        tree_dim: int,
        choice_function: str,
        bin_function: str,
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
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')

        self.d_out = d_out
        self.block = node.DenseBlock(
            input_dim=d_in,
            num_layers=num_layers,
            layer_dim=layer_dim,
            depth=depth,
            tree_dim=tree_dim,
            bin_function=getattr(node, bin_function),
            choice_function=getattr(node, choice_function),
            flatten_output=False,
        )

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        if x_cat is not None:
            x_cat = self.category_embeddings(x_cat + self.category_offsets[None])
            x = torch.cat([x_num, x_cat.view(x_cat.size(0), -1)], dim=-1)
        else:
            x = x_num

        x = self.block(x)
        x = x[..., : self.d_out].mean(dim=-2)
        x = x.squeeze(-1)
        return x


# %%
args, output = lib.load_config()
assert 'weight_decay' not in args, 'NODE architecture performs badly with weight decay'
if 'swa' in args:
    assert args['swa']['n_checkpoints'] > 1

# %%
zero.set_randomness(args['seed'])
dataset_dir = lib.get_path(args['data']['path'])
stats: ty.Dict[str, ty.Any] = {
    'dataset': dataset_dir.name,
    'algorithm': Path(__file__).stem,
    **lib.load_json(output / 'stats.json'),
}

D = lib.Dataset.from_dir(dataset_dir)
X = D.build_X(
    normalization=args['data'].get('normalization'),
    num_nan_policy='mean',
    cat_nan_policy='new',
    cat_policy='counter',
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
if not D.is_multiclass:
    Y_device = {k: v.float() for k, v in Y_device.items()}

train_size = len(X_num[lib.TRAIN])
batch_size, epoch_size = (
    stats['batch_size'],
    stats['epoch_size'],
) = lib.get_epoch_parameters(train_size, args['training'].get('batch_size', 'v3'))
eval_batch_size = args['training']['eval_batch_size']
chunk_size = None
stats['chunk_sizes'] = {}
stats['eval_batch_sizes'] = {}

loss_fn = (
    F.binary_cross_entropy_with_logits
    if D.is_binclass
    else F.cross_entropy
    if D.is_multiclass
    else F.mse_loss
)
args['model'].setdefault('d_embedding', None)
model = NODE(
    d_in=X_num[lib.TRAIN].shape[1],
    d_out=D.info['n_classes'] if D.is_multiclass else 1,
    categories=lib.get_categories(X_cat),
    **args['model'],
).to(device)
if torch.cuda.device_count() > 1:  # type: ignore[code]
    print('Using nn.DataParallel')
    model = nn.DataParallel(model)
stats['n_parameters'] = lib.get_n_parameters(model)
optimizer = lib.make_optimizer(
    args['training']['optimizer'],
    model.parameters(),
    args['training']['lr'],
    args['training']['weight_decay'],
)

stream = zero.Stream(lib.IndexLoader(train_size, batch_size, True, device))
progress = zero.ProgressTracker(args['training']['patience'])
training_log = {lib.TRAIN: [], lib.VAL: [], lib.TEST: []}
stage = 0
lr_n_decays = 0
timer = zero.Timer()
swa_stage_first_epoch = None


def print_epoch_info():
    print(
        f'\n>>> Epoch {stream.epoch} | Stage {stage} | {lib.format_seconds(timer())} | {output}'
    )
    details = {'lr': lib.get_lr(optimizer), 'chunk_size': chunk_size}
    details.update((x, stats[x]) for x in ['batch_size', 'epoch_size', 'n_parameters'])
    print(' | '.join(f'{k} = {v}' for k, v in details.items()))


def get_checkpoint_path(suffix):
    return output / f'checkpoint_{suffix}.pt'


def step(batch_idx):
    logits = model(
        X_num[lib.TRAIN][batch_idx],
        None if X_cat is None else X_cat[lib.TRAIN][batch_idx],
    )
    targets = Y_device[lib.TRAIN][batch_idx]  # type: ignore[code]
    if not D.is_multiclass:
        targets = targets.to(logits.dtype)
    return logits, targets


@torch.no_grad()
def predict(m, part):
    global eval_batch_size
    m.eval()
    random_state = zero.get_random_state()
    while eval_batch_size:
        try:
            zero.set_random_state(random_state)
            return torch.cat(
                [
                    model(X_num[part][idx], None if X_cat is None else X_cat[part][idx])
                    for idx in lib.IndexLoader(
                        len(X_num[part]),
                        args['training']['eval_batch_size'],
                        False,
                        device,
                    )
                ]
            ).cpu()
        except RuntimeError as err:
            if not lib.is_oom_exception(err):
                raise
            eval_batch_size //= 2
            print('New eval batch size:', eval_batch_size)
            stats['eval_batch_sizes'][stream.epoch] = eval_batch_size
    raise RuntimeError('Not enough memory even for eval_batch_size=1')


@torch.no_grad()
def evaluate(m, parts):
    metrics = {}
    predictions = {}
    for part in parts:
        predictions[part] = predict(m, part).numpy()
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


STATE_VARIABLES = [
    'progress',
    'stats',
    'timer',
    'training_log',
    'stage',
    'swa_stage_first_epoch',
    'lr_n_decays',
    'chunk_size',
    'eval_batch_size',
]


def save_checkpoint(suffix):
    torch.save(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'stream': stream.state_dict(),
            'random_state': zero.get_random_state(),
            **{x: globals()[x] for x in STATE_VARIABLES},
        },
        get_checkpoint_path(suffix),
    )
    lib.dump_stats(stats, output, suffix == 'final')
    lib.backup_output(output)


for stage in list(range(args.get('swa', {}).get('n_checkpoints', 1)))[::-1]:
    if get_checkpoint_path(stage).exists():
        print(f'Loading checkpoint {get_checkpoint_path(stage).name}')
        c = torch.load(get_checkpoint_path(stage))
        model.load_state_dict(c['model'])
        optimizer.load_state_dict(c['optimizer'])
        stream.load_state_dict(c['stream'])
        globals().update({x: c[x] for x in STATE_VARIABLES})
        stats.setdefault('old_stats', []).append(deepcopy(stats))
        stats.setdefault('continuations', []).append(stream.epoch)
        zero.set_random_state(c['random_state'])
        break


# %%
timer.run()
with torch.no_grad():
    # NODE-specific initialization
    if stream.epoch == 0:
        model.eval()
        size = 2048
        while True:
            try:
                zero.set_randomness(args['seed'])
                step(torch.randperm(train_size)[:size])
            except RuntimeError as err:
                if not lib.is_oom_exception(err):
                    raise
                size //= 2
            else:
                break
for epoch in stream.epochs(args['training']['n_epochs']):
    print_epoch_info()

    epoch_losses = []
    for batch_idx in epoch:
        loss, new_chunk_size = lib.learn_with_auto_virtual_batch(
            model, optimizer, loss_fn, step, batch_idx, batch_size, chunk_size
        )
        epoch_losses.append(loss.detach())
        if new_chunk_size and new_chunk_size < (chunk_size or batch_size):
            chunk_size = new_chunk_size
            print('New chunk size:', chunk_size)
            stats['chunk_sizes'][stream.iteration] = chunk_size
    epoch_losses = torch.stack(epoch_losses).tolist()
    training_log[lib.TRAIN].extend(epoch_losses)
    print(f'[{lib.TRAIN}] loss = {round(sum(epoch_losses) / len(epoch_losses), 3)}')

    metrics, predictions = evaluate(model, [lib.VAL, lib.TEST])
    for k, v in metrics.items():
        training_log[k].append(v)

    progress.update(metrics[lib.VAL]['score'])
    if progress.success:
        print('New best epoch!')
        stats[f'best_epoch_{stage}'] = stream.epoch
        stats[f'metrics_{stage}'] = metrics
        save_checkpoint(stage)
        for k, v in predictions.items():
            np.save(output / f'p_{stage}_{k}.npy', v)

    elif progress.fail:

        if stage == 0 and lr_n_decays < args['training']['lr_n_decays']:
            print('Reducing lr...')
            stats[f'lr_decay_{lr_n_decays}'] = stream.epoch
            lib.set_lr(optimizer, lib.get_lr(optimizer) * args['training']['lr_decay'])
            lr_n_decays += 1
            progress.forget_bad_updates()

        else:
            print(f'Finishing stage {stage}...')
            stats[f'time_{stage}'] = lib.format_seconds(timer())
            if 'swa' not in args or stage + 1 == args['swa']['n_checkpoints']:
                break

            best_stage_checkpoint = torch.load(get_checkpoint_path(stage))
            model.load_state_dict(best_stage_checkpoint['model'])
            optimizer.load_state_dict(best_stage_checkpoint['optimizer'])

            progress = zero.ProgressTracker(args['swa']['patience'])
            lib.set_lr(optimizer, args['training']['lr'] * args['swa']['lr_factor'])
            swa_stage_first_epoch = stream.epoch + 1
            stage += 1

    if stream.epoch == swa_stage_first_epoch:
        lib.set_lr(optimizer, args['training']['lr'])


# %%
def load_best_model(stage):
    model.load_state_dict(torch.load(get_checkpoint_path(stage))['model'])


if 'swa' in args:
    print('\nRunning SWA...')
    swa_model = swa_utils.AveragedModel(model)
    swa_progress = zero.ProgressTracker(None)
    best_swa_model = None

    for stage in range(args['swa']['n_checkpoints']):
        load_best_model(stage)
        swa_model.update_parameters(model)

        if stage > 0 and args['swa']['update_bn_n_epochs']:
            zero.set_randomness(args['seed'])
            with torch.no_grad():
                swa_utils.update_bn(
                    itertools.chain.from_iterable(
                        zero.iter_batches(
                            X[lib.TRAIN], chunk_size or batch_size, shuffle=True
                        )
                        for _ in range(args['swa']['update_bn_n_epochs'])
                    ),
                    swa_model,
                    device,
                )
        swa_progress.update(
            evaluate(swa_model if stage > 0 else model, [lib.VAL])[0][lib.VAL]['score']
        )
        if swa_progress.success:
            print('New best SWA checkpoint!')
            stats['n_swa_checkpoints'] = stage + 1
            if stage > 0:
                best_swa_model = deepcopy(swa_model)
    if best_swa_model is None:
        load_best_model(0)
    else:
        lib.load_swa_state_dict(model, best_swa_model)
else:
    load_best_model(0)

print('\nRunning the final evaluation...')
stats['metrics'], predictions = evaluate(model, lib.PARTS)
for k, v in predictions.items():
    np.save(output / f'p_{k}.npy', v)
stats['time_final'] = lib.format_seconds(timer())
save_checkpoint('final')
print(f'Done! Time elapsed: {stats["time_final"]}')
print(
    '\n!!! WARNING !!! The metrics for a single model are stored under the "metrics_0" key.\n'
)
