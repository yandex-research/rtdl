import atexit
import shutil
import subprocess
import tempfile
import typing as ty
import uuid
import warnings
from copy import deepcopy
from pathlib import Path

import optuna
import optuna.samplers
import optuna.trial
import torch
import zero

import lib

# %%
args, output = lib.load_config()


# %%
def sample_parameters(
    trial: optuna.trial.Trial,
    space: ty.Dict[str, ty.Any],
    base_config: ty.Dict[str, ty.Any],
) -> ty.Dict[str, ty.Any]:
    def get_distribution(distribution_name):
        return getattr(trial, f'suggest_{distribution_name}')

    result = {}
    for label, subspace in space.items():
        if isinstance(subspace, dict):
            result[label] = sample_parameters(trial, subspace, base_config)
        else:
            assert isinstance(subspace, list)
            distribution, *args = subspace

            if distribution.startswith('?'):
                default_value = args[0]
                result[label] = (
                    get_distribution(distribution.lstrip('?'))(label, *args[1:])
                    if trial.suggest_categorical(f'optional_{label}', [False, True])
                    else default_value
                )

            elif distribution == '$mlp_d_layers':
                min_n_layers, max_n_layers, d_min, d_max = args
                n_layers = trial.suggest_int('n_layers', min_n_layers, max_n_layers)
                suggest_dim = lambda name: trial.suggest_int(name, d_min, d_max)  # noqa
                d_first = [suggest_dim('d_first')] if n_layers else []
                d_middle = (
                    [suggest_dim('d_middle')] * (n_layers - 2) if n_layers > 2 else []
                )
                d_last = [suggest_dim('d_last')] if n_layers > 1 else []
                result[label] = d_first + d_middle + d_last

            elif distribution == '$d_token':
                assert len(args) == 2
                try:
                    n_heads = base_config['model']['n_heads']
                except KeyError:
                    n_heads = base_config['model']['n_latent_heads']

                for x in args:
                    assert x % n_heads == 0
                result[label] = trial.suggest_int('d_token', *args, n_heads)  # type: ignore[code]

            elif distribution in ['$d_ffn_factor', '$d_hidden_factor']:
                if base_config['model']['activation'].endswith('glu'):
                    args = (args[0] * 2 / 3, args[1] * 2 / 3)
                result[label] = trial.suggest_uniform('d_ffn_factor', *args)

            else:
                result[label] = get_distribution(distribution)(label, *args)
    return result


def merge_sampled_parameters(config, sampled_parameters):
    for k, v in sampled_parameters.items():
        if isinstance(v, dict):
            merge_sampled_parameters(config.setdefault(k, {}), v)
        else:
            assert k not in config
            config[k] = v


def objective(trial: optuna.trial.Trial) -> float:
    config = deepcopy(args['base_config'])
    merge_sampled_parameters(
        config, sample_parameters(trial, args['optimization']['space'], config)
    )
    if args.get('config_type') in ['trv2', 'trv4']:
        config['model']['d_token'] -= (
            config['model']['d_token'] % config['model']['n_heads']
        )
    if args.get('config_type') == 'trv4':
        if config['model']['activation'].endswith('glu'):
            # This adjustment is needed to keep the number of parameters roughly in the
            # same range as for non-glu activations
            config['model']['d_ffn_factor'] *= 2 / 3
    trial_configs.append(config)

    with tempfile.TemporaryDirectory() as dir_:
        dir_ = Path(dir_)
        out = dir_ / f'trial_{trial.number}'
        config_path = out.with_suffix('.toml')
        lib.dump_toml(config, config_path)
        python = Path('/miniconda3/envs/main/bin/python')
        subprocess.run(
            [
                str(python) if python.exists() else "python",
                str(program_copy),
                str(config_path),
            ],
            check=True,
        )
        stats = lib.load_json(out / 'stats.json')
        stats['algorithm'] = stats['algorithm'].rsplit('___', 1)[0]
        trial_stats.append({**stats, 'trial_id': trial.number})
        lib.dump_json(trial_stats, output / 'trial_stats.json', indent=4)
        lib.backup_output(output)
        print(f'Time: {lib.format_seconds(timer())}')
        return stats['metrics'][lib.VAL]['score']


def save_checkpoint(*_, **__):
    torch.save(
        {
            'trial_configs': trial_configs,
            'trial_stats': trial_stats,
            'study': study,
            'stats': stats,
            'timer': timer,
            'random_state': zero.get_random_state(),
        },
        checkpoint_path,
    )


program = lib.get_path(args['program'])
program_copy = program.with_name(
    program.stem + '___' + str(uuid.uuid4()).replace('-', '') + program.suffix
)
shutil.copyfile(program, program_copy)
atexit.register(lambda: program_copy.unlink())

checkpoint_path = output / 'checkpoint.pt'
if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path)
    trial_configs, trial_stats, study, stats, timer = (
        checkpoint['trial_configs'],
        checkpoint['trial_stats'],
        checkpoint['study'],
        checkpoint['stats'],
        checkpoint['timer'],
    )
    zero.set_random_state(checkpoint['random_state'])
    args['optimization']['options']['n_trials'] -= len(study.trials)
    stats.setdefault('continuations', []).append(len(study.trials))
    print(f'Loading checkpoint ({len(study.trials)})')
else:
    stats = lib.load_json(output / 'stats.json')
    trial_configs = []
    trial_stats = []
    timer = zero.Timer()
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(**args['optimization']['sampler']),
    )

timer.run()
# ignore the progress bar warning
warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
study.optimize(
    objective,
    **args['optimization']['options'],
    callbacks=[save_checkpoint],
    show_progress_bar=True,
)

best_trial_id = study.best_trial.number
lib.dump_toml(trial_configs[best_trial_id], output / 'best.toml')
stats['best_stats'] = trial_stats[best_trial_id]
stats['time'] = lib.format_seconds(timer())
lib.dump_stats(stats, output, True)
lib.backup_output(output)
