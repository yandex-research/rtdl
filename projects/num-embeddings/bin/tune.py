import atexit
import shutil
import subprocess
import tempfile
import uuid
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import optuna
import optuna.samplers
import optuna.trial
import torch
import zero

import lib


# %%
@dataclass
class Config:
    seed: int
    program: str
    base_config: dict[str, Any]
    space: dict[str, Any]
    n_trials: Optional[int] = None
    timeout: Optional[int] = None
    sampler: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert self.n_trials is not None or self.timeout is not None
        assert 'seed' not in self.sampler


C, output, report = lib.start(Config)


# %%
def sample_parameters(
    trial: optuna.trial.Trial,
    space: dict[str, Any],
    base_config: dict[str, Any],
) -> dict[str, Any]:
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
                assert distribution[2] != '?'
                has_default = not distribution.startswith('??')
                if trial.suggest_categorical(f'optional_{label}', [False, True]):
                    result[label] = get_distribution(distribution.lstrip('?'))(
                        label, *(args[1:] if has_default else args)
                    )
                elif has_default:
                    result[label] = args[0]

            elif distribution.startswith('$mlp_d_layers'):
                min_n_layers, max_n_layers, d_min, d_max = args
                n_layers = trial.suggest_int('n_layers', min_n_layers, max_n_layers)

                def suggest_dim(name):
                    return trial.suggest_int(name, d_min, d_max)

                if distribution == '$mlp_d_layers':
                    d_first = [suggest_dim('d_first')] if n_layers else []
                    d_middle = (
                        [suggest_dim('d_middle')] * (n_layers - 2)
                        if n_layers > 2
                        else []
                    )
                    d_last = [suggest_dim('d_last')] if n_layers > 1 else []
                    result[label] = d_first + d_middle + d_last
                else:
                    assert False

                # elif distribution == '$mlp_d_layers_v1':
                #     d_first = [suggest_dim('d_first')] if n_layers else []
                #     d_others = (
                #         [suggest_dim('d_middle')] * (n_layers - 1)
                #         if n_layers > 1
                #         else []
                #     )
                #     result[label] = d_first + d_others

                # else:
                #     assert distribution == '$mlp_d_layers_v2'
                #     result[label] = (
                #         [suggest_dim('d_main')] * n_layers if n_layers else []
                #     )

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
    config = deepcopy(C.base_config)
    merge_sampled_parameters(config, sample_parameters(trial, C.space, config))

    with tempfile.TemporaryDirectory() as dir_:
        dir_ = Path(dir_)
        out = dir_ / f'trial_{trial.number}'
        config_path = out.with_suffix('.toml')
        lib.dump_config(config, config_path)
        python = Path('/miniconda3/envs/main/bin/python')
        subprocess.run(
            [
                str(python) if python.exists() else "python",
                str(program_copy),
                str(config_path),
            ],
            check=True,
        )
        report = lib.load_report(out)
        report['program'] = C.program
        report['trial_id'] = trial.number
        report['tuning_time'] = str(timer)
        trial_reports.append(report)
        return report['metrics']['val']['score']


def callback(*_, **__):
    report['best'] = trial_reports[study.best_trial.number]
    report['time'] = str(timer)
    torch.save(
        {
            'report': report,
            'study': study,
            'trial_reports': trial_reports,
            'timer': timer,
            'random_state': zero.random.get_state(),
        },
        checkpoint_path,
    )
    lib.dump_report(report, output)
    lib.backup_output(output)
    print(f'Time: {timer}')


zero.improve_reproducibility(C.seed)
program = lib.get_path(C.program)
program_copy = program.with_name(
    program.stem + '___' + str(uuid.uuid4()).replace('-', '') + program.suffix
)
shutil.copyfile(program, program_copy)
atexit.register(lambda: program_copy.unlink())

checkpoint_path = output / 'checkpoint.pt'
if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path)
    report, study, trial_reports, timer = (
        checkpoint['report'],
        checkpoint['study'],
        checkpoint['trial_reports'],
        checkpoint['timer'],
    )
    zero.random.set_state(checkpoint['random_state'])
    if C.n_trials is not None:
        C.n_trials -= len(study.trials)
    if C.timeout is not None:
        C.timeout -= timer()

    report.setdefault('continuations', []).append(len(study.trials))
    print(f'Loading checkpoint ({len(study.trials)})')
else:
    report = lib.load_report(output)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(**C.sampler, seed=C.seed),
    )
    trial_reports = []
    timer = zero.Timer()

timer.run()
# ignore the progress bar warning
warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
study.optimize(
    objective,
    n_trials=C.n_trials,
    timeout=C.timeout,
    callbacks=[callback],
    show_progress_bar=True,
)
lib.finish(output, report)
