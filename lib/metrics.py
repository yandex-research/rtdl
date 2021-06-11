import typing as ty

import numpy as np
import scipy.special
import sklearn.metrics as skm

from . import util


def calculate_metrics(
    task_type: str,
    y: np.ndarray,
    prediction: np.ndarray,
    classification_mode: str,
    y_info: ty.Optional[ty.Dict[str, ty.Any]],
) -> ty.Dict[str, float]:
    if task_type == util.REGRESSION:
        del classification_mode
        rmse = skm.mean_squared_error(y, prediction) ** 0.5  # type: ignore[code]
        if y_info:
            if y_info['policy'] == 'mean_std':
                rmse *= y_info['std']
            else:
                assert False
        return {'rmse': rmse, 'score': -rmse}
    else:
        assert task_type in (util.BINCLASS, util.MULTICLASS)
        labels = None
        if classification_mode == 'probs':
            probs = prediction
        elif classification_mode == 'logits':
            probs = (
                scipy.special.expit(prediction)
                if task_type == util.BINCLASS
                else scipy.special.softmax(prediction, axis=1)
            )
        else:
            assert classification_mode == 'labels'
            probs = None
            labels = prediction
        if labels is None:
            labels = (
                np.round(probs).astype('int64')
                if task_type == util.BINCLASS
                else probs.argmax(axis=1)  # type: ignore[code]
            )

        result = skm.classification_report(y, labels, output_dict=True)  # type: ignore[code]
        if task_type == util.BINCLASS:
            result['roc_auc'] = skm.roc_auc_score(y, labels)  # type: ignore[code]
        result['score'] = result[  # type: ignore[code]
            'roc_auc' if task_type == util.BINCLASS else 'accuracy'
        ]
    return result  # type: ignore[code]


def make_summary(metrics: ty.Dict[str, ty.Any]) -> str:
    precision = 3
    summary = {}
    for k, v in metrics.items():
        if k.isdigit():
            continue
        k = {
            'score': 'SCORE',
            'accuracy': 'acc',
            'roc_auc': 'roc_auc',
            'macro avg': 'm',
            'weighted avg': 'w',
        }.get(k, k)
        if isinstance(v, float):
            v = round(v, precision)
            summary[k] = v
        else:
            v = {
                {'precision': 'p', 'recall': 'r', 'f1-score': 'f1', 'support': 's'}.get(
                    x, x
                ): round(v[x], precision)
                for x in v
            }
            for item in v.items():
                summary[k + item[0]] = item[1]

    s = [f'score = {summary.pop("SCORE"):.3f}']
    for k, v in summary.items():
        if k not in ['mp', 'mr', 'wp', 'wr']:  # just to save screen space
            s.append(f'{k} = {v}')
    return ' | '.join(s)
