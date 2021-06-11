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
    cat_policy=args['data'].get('cat_policy', 'counter'),
    cat_min_frequency=args['data'].get('cat_min_frequency', 0.0),
    seed=args['seed'],
)
zero.set_randomness(args['seed'])
Y, y_info = D.build_y(args['data'].get('y_policy'))
lib.dump_pickle(y_info, output / 'y_info.pickle')

model_kwargs = args['model']

if args['data'].get('cat_policy') == 'indices':
    N, C = X
    n_num_features = D.info['n_num_features']
    n_features = D.n_features

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
    model = CatBoostClassifier(
        **model_kwargs, eval_metric='AUC' if D.is_binclass else 'Accuracy'
    )
    predict = (
        model.predict_proba
        if D.is_multiclass
        else lambda x: model.predict_proba(x)[:, 1]  # type: ignore[code]
    )

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
lib.dump_stats(stats, output, True)
lib.backup_output(output)
