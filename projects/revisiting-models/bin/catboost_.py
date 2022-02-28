import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import zero
from catboost import CatBoostClassifier, CatBoostRegressor

import lib

args, output = lib.load_config()
args['model']['random_seed'] = args['seed']
assert (
    'task_type' in args['model']
)  # Significantly affects performance, so must be set explicitely
if args['model']['task_type'] == 'GPU':
    assert os.environ.get('CUDA_VISIBLE_DEVICES')

zero.set_randomness(args['seed'])
dataset_dir = lib.get_path(args['data']['path'])
stats = lib.load_json(output / 'stats.json')
stats.update({'dataset': dataset_dir.name, 'algorithm': Path(__file__).stem})

D = lib.Dataset.from_dir(dataset_dir)
X = D.build_X(
    normalization=args['data'].get('normalization'),
    num_nan_policy='mean',
    cat_nan_policy='new',
    cat_policy=args['data'].get('cat_policy', 'indices'),
    cat_min_frequency=args['data'].get('cat_min_frequency', 0.0),
    seed=args['seed'],
)
zero.set_randomness(args['seed'])
Y, y_info = D.build_y(args['data'].get('y_policy'))
lib.dump_pickle(y_info, output / 'y_info.pickle')

model_kwargs = args['model']

if args['data'].get('cat_policy') == 'indices':
    assert isinstance(X, tuple)
    N, C = X
    n_num_features = 0 if N is None else N[lib.TRAIN].shape[1]
    n_cat_features = 0 if C is None else C[lib.TRAIN].shape[1]
    n_features = n_num_features + n_cat_features
    if N is None:
        assert C is not None
        X = {x: pd.DataFrame(C[x], columns=range(n_features)) for x in C}
    elif C is None:
        assert N is not None
        X = {x: pd.DataFrame(N[x], columns=range(n_features)) for x in N}
    else:
        X = {
            k: pd.concat(
                [
                    pd.DataFrame(N[k], columns=range(n_num_features)),
                    pd.DataFrame(C[k], columns=range(n_num_features, n_features)),
                ],
                axis=1,
            )
            for k in N.keys()
        }

    model_kwargs['cat_features'] = list(range(n_num_features, n_features))


if model_kwargs['task_type'] == 'GPU':
    model_kwargs['devices'] = '0'
if D.is_regression:
    model = CatBoostRegressor(**model_kwargs)
    predict = model.predict
else:
    model = CatBoostClassifier(**model_kwargs, eval_metric='Accuracy')
    predict = (
        model.predict_proba
        if D.is_multiclass
        else lambda x: model.predict_proba(x)[:, 1]  # type: ignore[code]
    )

timer = zero.Timer()
timer.run()
model.fit(
    X[lib.TRAIN],
    Y[lib.TRAIN],
    **args['fit'],
    eval_set=(X[lib.VAL], Y[lib.VAL]),
)
if Path('catboost_info').exists():
    shutil.rmtree('catboost_info')

model.save_model(str(output / 'model.cbm'))
np.save(output / 'feature_importances.npy', model.get_feature_importance())
stats['metrics'] = {}
for part in X:
    p = predict(X[part])
    stats['metrics'][part] = lib.calculate_metrics(
        D.info['task_type'], Y[part], p, 'probs', y_info
    )
    np.save(output / f'p_{part}.npy', p)
stats['time'] = lib.format_seconds(timer())
lib.dump_stats(stats, output, True)
lib.backup_output(output)
