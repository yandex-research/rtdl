import os
from dataclasses import InitVar, dataclass, field
from typing import Any

import numpy as np
import zero
from xgboost import XGBClassifier, XGBRegressor

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
    xgboost: dict[str, Any] = field(default_factory=dict)
    xgboost_fit: dict[str, Any] = field(default_factory=dict)
    validate: InitVar[bool] = True

    def __post_init__(self, validate):
        assert 'random_state' not in self.xgboost
        self.xgboost = {
            'booster': 'gbtree',
            'n_estimators': 2000,
            'n_jobs': 1,
        } | self.xgboost
        self.xgboost_fit = {
            'early_stopping_rounds': 50,
            'verbose': True,
        } | self.xgboost_fit

        if validate:
            assert (
                'early_stopping_rounds' in self.xgboost_fit
            ), 'XGBoost does not automatically use the best model, so early stopping must be used'
            use_gpu = self.xgboost.get('tree_method') == 'gpu_hist'
            if use_gpu:
                assert os.environ.get('CUDA_VISIBLE_DEVICES')


# %%
C, output, report = lib.start(Config)

# %%
zero.improve_reproducibility(C.seed)
D = lib.build_dataset(C.data.path, C.data.T, C.data.T_cache)
assert D.X_num is not None and D.X_cat is None, (
    "XGBoost does not support categorical features."
    " Set `[data][features]cat_encoding = 'one-hot' or 'counter'`"
    " in the config to fix this issue."
)
lib.dump_pickle(D.y_info, output / 'y_info.pickle')
report['prediction_type'] = None if D.is_regression else 'probs'

# %%
if D.is_regression:
    model = XGBRegressor(**C.xgboost, random_state=C.seed)
    predict = model.predict
else:
    model = XGBClassifier(
        **C.xgboost, random_state=C.seed, disable_default_eval_metric=True
    )
    if D.is_multiclass:
        predict = model.predict_proba
        C.xgboost_fit['eval_metric'] = 'merror'
    else:
        predict = lambda x: model.predict_proba(x)[:, 1]  # type: ignore[code]  # noqa
        C.xgboost_fit['eval_metric'] = 'error'

# %%
timer = zero.Timer()
timer.run()
model.fit(
    D.X_num['train'],
    D.y['train'],
    eval_set=[(D.X_num['val'], D.y['val'])],
    **C.xgboost_fit,
)

# %%
model.save_model(str(output / "model.xgbm"))
np.save(output / "feature_importances.npy", model.feature_importances_)

predictions = {k: predict(v) for k, v in D.X_num.items()}
report['metrics'] = D.calculate_metrics(predictions, report['prediction_type'])
lib.dump_predictions(predictions, output)
report['time'] = str(timer)
lib.finish(output, report)
