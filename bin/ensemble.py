# %%
import argparse
from copy import deepcopy

import numpy as np
from scipy.special import expit, softmax

import lib
import lib.env as env

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='''Examples:
python bin/ensemble.py mlp output/california_housing/mlp/tuned
python bin/ensemble.py ft_transformer output/california_housing/ft_transformer/default
''',
)
parser.add_argument('algorithm')
parser.add_argument('experiment')
args = parser.parse_args()

PARTS = ['val', 'test'] if args.algorithm == 'node' else ['train', 'val', 'test']
SRC = env.PROJECT_DIR / args.experiment
DST = SRC.with_name(SRC.name + '_ensemble')

for seeds in [range(0, 5), range(5, 10), range(10, 15)]:
    if any(not (SRC / str(seed) / 'DONE').exists() for seed in seeds):
        continue

    output = DST / f'{min(seeds)}_{max(seeds)}'
    if output.exists():
        continue

    print(seeds)
    DST.mkdir(exist_ok=True)

    src_0 = SRC / '0'
    src_stats = lib.load_json(src_0 / 'stats.json')
    stats_template = {
        'dataset': src_stats['dataset'],
        'algorithm': src_stats['algorithm'],
        'config': {'source': str(SRC.relative_to(env.PROJECT_DIR))},
        'metrics': {},
    }
    D = lib.Dataset.from_dir(lib.get_path(src_stats['config']['data']['path']))
    Y, _ = D.build_y(src_stats['config']['data'].get('y_policy'))
    y_info = lib.load_pickle(src_0 / 'y_info.pickle')
    predictions = {
        x: np.stack([np.load(SRC / str(seed) / f'p_{x}.npy') for seed in seeds])
        for x in PARTS
    }

    stats = deepcopy(stats_template)
    stats['config']['seeds'] = list(seeds)
    stats['config']['count'] = len(seeds)
    for part in PARTS:
        single_predictions = predictions[part]  # type: ignore[code]
        if D.is_binclass:
            single_predictions = expit(single_predictions)
        elif D.is_multiclass and args.algorithm not in ('catboost', 'xgboost'):
            single_predictions = softmax(single_predictions, -1)
        else:
            assert D.is_regression
        stats['metrics'][part] = lib.calculate_metrics(
            D.info['task_type'],
            Y[part],  # type: ignore[code]
            single_predictions.mean(0),  # type: ignore[code]
            'probs',
            y_info,
        )
    output.mkdir()
    lib.dump_stats(stats, output, True)

# %%
