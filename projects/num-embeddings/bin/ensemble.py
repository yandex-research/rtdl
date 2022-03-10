# %%
import argparse
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import zero
from scipy.special import expit, softmax

import lib

# %%
parser = argparse.ArgumentParser()
parser.add_argument('evaluation_dir', metavar='DIR', type=Path)
parser.add_argument('-c', '--count', type=int, default=3)
parser.add_argument('-s', '--size', type=int, default=5)
parser.add_argument('--force', action='store_true')
args = parser.parse_args()


# %%
evaluation_dir = lib.get_path(args.evaluation_dir)
for ensemble_id in range(args.count):
    seeds = range(ensemble_id * args.size, (ensemble_id + 1) * args.size)
    single_outputs = [evaluation_dir / str(x) for x in seeds]
    output = evaluation_dir.with_name(
        evaluation_dir.name.replace('evaluation', f'ensemble_{args.size}')
    ) / str(ensemble_id)
    if not all((x / 'DONE').exists() for x in single_outputs):
        print(f'Not enough single models: {output.name}')
        continue
    if output.exists():
        if args.force:
            print(f'Removing because of --force: {output}')
            shutil.rmtree(output)
        else:
            print(f"Already exists: {output}")
            continue

    output.mkdir(parents=True)
    report: dict[str, Any] = {
        'program': 'bin/ensemble.py',
        'config': {
            'seeds': list(seeds),
        },
    }
    zero.improve_reproducibility(0)

    first_report = lib.load_report(single_outputs[0])
    report['single_model_program'] = first_report['program']
    report['data'] = first_report["config"]["data"]["path"]
    dataset = lib.Dataset.from_dir(lib.get_path(first_report['config']['data']['path']))
    report['prediction_type'] = None if dataset.is_regression else 'probs'
    y, y_info = lib.build_target(
        dataset.y, first_report['config']['data']['T']['y_policy'], dataset.task_type
    )
    dataset = replace(dataset, y=y, y_info=y_info)
    single_predictions = [lib.load_predictions(x) for x in single_outputs]

    # %%
    predictions = {}
    for part in ['train', 'val', 'test']:
        stacked_predictions = np.stack([x[part] for x in single_predictions])  # type: ignore[code]
        if dataset.is_binclass:
            if first_report['prediction_type'] == 'logits':
                stacked_predictions = expit(stacked_predictions)
        elif dataset.is_multiclass:
            if first_report['prediction_type'] == 'logits':
                stacked_predictions = softmax(stacked_predictions, -1)
        else:
            assert dataset.is_regression
        predictions[part] = stacked_predictions.mean(0)
    report['metrics'] = dataset.calculate_metrics(
        predictions, report['prediction_type']
    )
    lib.dump_predictions(predictions, output)
    lib.finish(output, report)
