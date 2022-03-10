import os
import shutil
from dataclasses import InitVar, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import zero
from catboost import CatBoostClassifier, CatBoostRegressor

import lib


# %%
@dataclass
class Config:
    @dataclass
    class Data:
        path: str
        T: lib.Transformations = field(default_factory=lib.Transformations)
        T_cache: bool = False

    seed: int
    data: Data
    catboost: dict[str, Any] = field(default_factory=dict)
    catboost_fit: dict[str, Any] = field(default_factory=dict)
    validate: InitVar[bool] = True

    def __post_init__(self, validate):
        assert 'random_seed' not in self.catboost
        assert (
            self.catboost.get('l2_leaf_reg', 3.0) > 0
        )  # CatBoost fails on multiclass problems with l2_leaf_reg=0; 3.0 is the default value
        self.catboost = {
            'iterations': 2000,
            'early_stopping_rounds': 50,
            'od_pval': 1e-3,
            'task_type': 'CPU',
            'thread_count': 1,
        } | self.catboost
        self.catboost_fit = {'logging_level': 'Verbose'} | self.catboost_fit

        if validate:
            assert (
                'task_type' in self.catboost
            ), 'task_type significantly affects performance, so must be set explicitly'
            if self.catboost['task_type'] == 'GPU':
                assert os.environ.get('CUDA_VISIBLE_DEVICES')


# %%
C, output, report = lib.start(Config)

# %%
zero.improve_reproducibility(C.seed)
D = lib.build_dataset(C.data.path, C.data.T, C.data.T_cache)
lib.dump_pickle(D.y_info, output / 'y_info.pickle')
report['prediction_type'] = None if D.is_regression else 'probs'

if D.X_num is None:
    assert D.X_cat is not None
    X = {k: pd.DataFrame(v, columns=range(D.n_features)) for k, v in D.X_cat.items()}
elif D.X_cat is None:
    assert D.X_num is not None
    X = {k: pd.DataFrame(v, columns=range(D.n_features)) for k, v in D.X_num.items()}
else:
    X = {
        part: pd.concat(
            [
                pd.DataFrame(D.X_num[part], columns=range(D.n_num_features)),
                pd.DataFrame(
                    D.X_cat[part],
                    columns=range(D.n_num_features, D.n_features),
                ),
            ],
            axis=1,
        )
        for part in D.y.keys()
    }
if D.X_cat is not None:
    C.catboost['cat_features'] = list(range(D.n_num_features, D.n_features))

# %%
if C.catboost['task_type'] == 'GPU':
    C.catboost['devices'] = '0'
C.catboost['train_dir'] = output / 'catboost_info'
if D.is_regression:
    model = CatBoostRegressor(**C.catboost, random_seed=C.seed)
    predict = model.predict
else:
    model = CatBoostClassifier(**C.catboost, random_seed=C.seed, eval_metric='Accuracy')
    predict = (
        model.predict_proba
        if D.is_multiclass
        else lambda x: model.predict_proba(x)[:, 1]  # type: ignore[code]
    )

# %%
timer = lib.Timer.launch()
model.fit(
    X['train'],
    D.y['train'],
    eval_set=(X['val'], D.y['val']),
    **C.catboost_fit,
)
shutil.rmtree(C.catboost['train_dir'])

# %%
model.save_model(output / 'model.cbm')
lib.dump_json(
    {
        x: getattr(model, x + '_')
        for x in ['tree_count', 'evals_result', 'best_iteration', 'best_score']
    },
    output / 'model_info.json',
    indent=None,
)
np.save(output / 'feature_importances.npy', model.get_feature_importance())

predictions = {k: predict(v) for k, v in X.items()}
report['metrics'] = D.calculate_metrics(predictions, report['prediction_type'])
lib.dump_predictions(predictions, output)
report['time'] = str(timer)
lib.finish(output, report)
