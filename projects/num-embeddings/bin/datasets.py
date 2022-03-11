# %%
import argparse
import enum
import json
import math
import random
import shutil
import sys
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, cast
from urllib.request import urlretrieve

import catboost.datasets
import geopy.distance
import numpy as np
import pandas as pd
import pyarrow.csv
import sklearn.datasets
import sklearn.utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

ArrayDict = dict[str, np.ndarray]
Info = dict[str, Any]

DATA_DIR = Path.home() / 'repositories' / 'a' / 'data'
SEED = 0
CAT_MISSING_VALUE = '__nan__'

EXPECTED_FILES = {
    'eye': [],
    'gas': [],
    'gesture': [],
    'house': [],
    'higgs-small': [],
    # Run `kaggle competitions download -c santander-customer-transaction-prediction`
    'santander': ['santander-customer-transaction-prediction.zip'],
    # Run `kaggle competitions download -c otto-group-product-classification-challenge`
    'otto': ['otto-group-product-classification-challenge.zip'],
    # Run `kaggle competitions download -c rossmann-store-sales`
    'rossmann': ['rossmann-store-sales.zip'],
    # Source: https://www.kaggle.com/shrutimechlearn/churn-modelling
    'churn': ['Churn_Modelling.csv'],
    # Source: https://www.kaggle.com/neomatrix369/nyc-taxi-trip-duration-extended
    'taxi': ['train_extended.csv.zip'],
    # Source: https://archive.ics.uci.edu/ml/machine-learning-databases/00363/Dataset.zip
    'fb-comments': ['Dataset.zip'],
    'california': [],
    'covtype': [],
    'adult': [],
    # Source: https://www.dropbox.com/s/572rj8m5f9l2nz5/MSLR-WEB10K.zip?dl=1
    # This is literally the official data, but reuploded to Dropbox.
    'microsoft': ['MSLR-WEB10K.zip'],
}
EXPECTED_FILES['wd-taxi'] = EXPECTED_FILES['taxi']
EXPECTED_FILES['fb-c'] = EXPECTED_FILES['wd-fb-comments'] = EXPECTED_FILES[
    'fb-comments'
]


class TaskType(enum.Enum):
    REGRESSION = 'regression'
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'


# %%
def _set_random_seeds():
    random.seed(SEED)
    np.random.seed(SEED)


def _download_file(url: str, path: Path):
    assert not path.exists()
    try:
        print(f'Downloading {url} ...', end='', flush=True)
        urlretrieve(url, path)
    except Exception:
        if path.exists():
            path.unlink()
        raise
    finally:
        print()


def _unzip(path: Path, members: Optional[list[str]] = None) -> None:
    with zipfile.ZipFile(path) as f:
        f.extractall(path.parent, members)


def _start(dirname: str) -> tuple[Path, list[Path]]:
    print(f'>>> {dirname}')
    _set_random_seeds()
    dataset_dir = DATA_DIR / dirname
    expected_files = EXPECTED_FILES[dirname]
    if expected_files:
        assert dataset_dir.exists()
        assert set(expected_files) == set(x.name for x in dataset_dir.iterdir())
    else:
        assert not dataset_dir.exists()
        dataset_dir.mkdir()
    return dataset_dir, [dataset_dir / x for x in expected_files]


def _fetch_openml(data_id: int) -> sklearn.utils.Bunch:
    bunch = cast(
        sklearn.utils.Bunch,
        sklearn.datasets.fetch_openml(data_id=data_id, as_frame=True),
    )
    assert not bunch['categories']
    return bunch


def _get_sklearn_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    get_data = getattr(sklearn.datasets, f'load_{name}', None)
    if get_data is None:
        get_data = getattr(sklearn.datasets, f'fetch_{name}', None)
    assert get_data is not None, f'No such dataset in scikit-learn: {name}'
    return get_data(return_X_y=True)


def _encode_classification_target(y: np.ndarray) -> np.ndarray:
    assert not str(y.dtype).startswith('float')
    if str(y.dtype) not in ['int32', 'int64', 'uint32', 'uint64']:
        y = LabelEncoder().fit_transform(y)
    else:
        labels = set(map(int, y))
        if sorted(labels) != list(range(len(labels))):
            y = LabelEncoder().fit_transform(y)
    return y.astype(np.int64)


def _make_split(size: int, stratify: Optional[np.ndarray], n_parts: int) -> ArrayDict:
    # n_parts == 3:      all -> train & val & test
    # n_parts == 2: trainval -> train & val
    assert n_parts in (2, 3)
    all_idx = np.arange(size, dtype=np.int64)
    a_idx, b_idx = train_test_split(
        all_idx,
        test_size=0.2,
        stratify=stratify,
        random_state=SEED + (1 if n_parts == 2 else 0),
    )
    if n_parts == 2:
        return cast(ArrayDict, {'train': a_idx, 'val': b_idx})
    a_stratify = None if stratify is None else stratify[a_idx]
    a1_idx, a2_idx = train_test_split(
        a_idx, test_size=0.2, stratify=a_stratify, random_state=SEED + 1
    )
    return cast(ArrayDict, {'train': a1_idx, 'val': a2_idx, 'test': b_idx})


def _apply_split(data: ArrayDict, split: ArrayDict) -> dict[str, ArrayDict]:
    return {k: {part: v[idx] for part, idx in split.items()} for k, v in data.items()}


def _save(
    dataset_dir: Path,
    name: str,
    task_type: TaskType,
    *,
    X_num: Optional[ArrayDict],
    X_cat: Optional[ArrayDict],
    y: ArrayDict,
    idx: Optional[ArrayDict],
    id_: Optional[str] = None,
    id_suffix: str = '--default',
) -> None:
    if id_ is not None:
        assert id_suffix == '--default'
    assert (
        X_num is not None or X_cat is not None
    ), 'At least one type of features must be presented.'
    if X_num is not None:
        X_num = {k: v.astype(np.float32) for k, v in X_num.items()}
    if X_cat is not None:
        X_cat = {k: v.astype(str) for k, v in X_cat.items()}
    if idx is not None:
        idx = {k: v.astype(np.int64) for k, v in idx.items()}
    y = {
        k: v.astype(np.float32 if task_type == TaskType.REGRESSION else np.int64)
        for k, v in y.items()
    }
    if task_type != TaskType.REGRESSION:
        y_unique = {k: set(v.tolist()) for k, v in y.items()}
        assert y_unique['train'] == set(range(max(y_unique['train']) + 1))
        for x in ['val', 'test']:
            assert y_unique[x] <= y_unique['train']
        del x

    info = {
        'name': name,
        'id': (dataset_dir.name + id_suffix) if id_ is None else id_,
        'task_type': task_type.value,
        'n_num_features': (0 if X_num is None else next(iter(X_num.values())).shape[1]),
        'n_cat_features': (0 if X_cat is None else next(iter(X_cat.values())).shape[1]),
    } | {f'{k}_size': len(v) for k, v in y.items()}
    if task_type == TaskType.MULTICLASS:
        info['n_classes'] = len(set(y['train']))
    (dataset_dir / 'info.json').write_text(json.dumps(info, indent=4))

    for data_name in ['X_num', 'X_cat', 'y', 'idx']:
        data = locals()[data_name]
        if data is not None:
            for k, v in data.items():
                np.save(dataset_dir / f'{data_name}_{k}.npy', v)
    (dataset_dir / 'READY').touch()
    print('Done\n')


# %%
def eye_movements():
    dataset_dir, _ = _start('eye')
    bunch = _fetch_openml(1044)

    X_num_all = bunch['data'].drop(columns=['lineNo']).values.astype(np.float32)
    y_all = _encode_classification_target(bunch['target'].cat.codes.values)
    idx = _make_split(len(X_num_all), y_all, 3)

    _save(
        dataset_dir,
        'Eye Movements',
        TaskType.MULTICLASS,
        **_apply_split({'X_num': X_num_all, 'y': y_all}, idx),
        X_cat=None,
        idx=idx,
    )


def gesture_phase():
    dataset_dir, _ = _start('gesture')
    bunch = _fetch_openml(4538)

    X_num_all = bunch['data'].values.astype(np.float32)
    y_all = _encode_classification_target(bunch['target'].cat.codes.values)
    idx = _make_split(len(X_num_all), y_all, 3)

    _save(
        dataset_dir,
        'Gesture Phase',
        TaskType.MULTICLASS,
        **_apply_split({'X_num': X_num_all, 'y': y_all}, idx),
        X_cat=None,
        idx=idx,
    )


def house_16h():
    dataset_dir, _ = _start('house')
    bunch = _fetch_openml(574)

    X_num_all = bunch['data'].values.astype(np.float32)
    y_all = bunch['target'].values.astype(np.float32)
    idx = _make_split(len(X_num_all), None, 3)

    _save(
        dataset_dir,
        'House 16H',
        TaskType.REGRESSION,
        **_apply_split({'X_num': X_num_all, 'y': y_all}, idx),
        X_cat=None,
        idx=idx,
    )


def higgs_small():
    dataset_dir, _ = _start('higgs-small')
    bunch = _fetch_openml(23512)

    X_num_all = bunch['data'].values.astype(np.float32)
    y_all = _encode_classification_target(bunch['target'].cat.codes.values)
    nan_mask = np.isnan(X_num_all)
    valid_objects_mask = ~(nan_mask.any(1))
    # There is just one object with nine(!) missing values; let's drop it
    assert valid_objects_mask.sum() + 1 == len(X_num_all) and nan_mask.sum() == 9
    X_num_all = X_num_all[valid_objects_mask]
    y_all = y_all[valid_objects_mask]
    idx = _make_split(len(X_num_all), y_all, 3)

    _save(
        dataset_dir,
        'Higgs Small',
        TaskType.BINCLASS,
        **_apply_split({'X_num': X_num_all, 'y': y_all}, idx),
        X_cat=None,
        idx=idx,
    )


def santander_customer_transactions():
    dataset_dir, files = _start('santander')
    _unzip(files[0])

    df = pd.read_csv(dataset_dir / 'train.csv')
    df.drop(columns=['ID_code'], inplace=True)
    y_all = _encode_classification_target(df.pop('target').values)  # type: ignore[code]
    X_num_all = df.values.astype(np.float32)
    idx = _make_split(len(X_num_all), y_all, 3)

    _save(
        dataset_dir,
        'Santander Customer Transactions',
        TaskType.BINCLASS,
        **_apply_split({'X_num': X_num_all, 'y': y_all}, idx),
        X_cat=None,
        idx=idx,
    )


def otto_group_products():
    dataset_dir, files = _start('otto')
    _unzip(files[0])

    df = pd.read_csv(dataset_dir / 'train.csv')
    df.drop(columns=['id'], inplace=True)
    y_all = _encode_classification_target(
        df.pop('target').map(lambda x: int(x.split('_')[-1]) - 1).values  # type: ignore[code]
    )
    X_num_all = df.values.astype(np.float32)
    idx = _make_split(len(X_num_all), y_all, 3)

    _save(
        dataset_dir,
        'Otto Group Products',
        TaskType.MULTICLASS,
        **_apply_split({'X_num': X_num_all, 'y': y_all}, idx),
        X_cat=None,
        idx=idx,
    )


def churn_modelling():
    # Get the file here: https://www.kaggle.com/shrutimechlearn/churn-modelling
    dataset_dir, files = _start('churn')
    df = pd.read_csv(files[0])

    df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
    df['Gender'] = df['Gender'].astype('category').cat.codes.values.astype(np.int64)
    y_all = df.pop('Exited').values.astype(np.int64)
    num_columns = [
        'CreditScore',
        'Gender',
        'Age',
        'Tenure',
        'Balance',
        'NumOfProducts',
        'EstimatedSalary',
        'HasCrCard',
        'IsActiveMember',
        'EstimatedSalary',
    ]
    cat_columns = ['Geography']
    assert set(num_columns) | set(cat_columns) == set(df.columns.tolist())
    X_num_all = df[num_columns].astype(np.float32).values
    X_cat_all = df[cat_columns].astype(str).values
    idx = _make_split(len(df), y_all, 3)

    _save(
        dataset_dir,
        'Churn Modelling',
        TaskType.BINCLASS,
        **_apply_split(
            {'X_num': X_num_all, 'X_cat': X_cat_all, 'y': y_all},
            idx,
        ),
        idx=idx,
    )


def facebook_comments_volume(keep_derived: bool):
    # This is our preprocessing. The difference with Wide & Deep:
    # - (let columns be: [c0, c1, ..., c51, c52, target])
    # - c3 is the only categorical feature (as described at the UCI page)
    # - c14 is removed (it contains three unique values with the following distribution: [157631, 4, 3])
    # - c37 is removed (it contains one unique value)
    # - if keep_derived is False, then [c4, c5, ..., c28] are removed
    dataset_dir, files = _start('fb-comments' if keep_derived else 'fb-c')
    csv_path = 'Dataset/Training/Features_Variant_5.csv'
    _unzip(files[0], [csv_path])

    df = pd.read_csv(
        dataset_dir / csv_path, names=[f'c{i}' for i in range(53)] + ['target']
    )
    extra_columns = {'c14', 'c37'}
    if not keep_derived:
        extra_columns.update(f'c{i}' for i in range(4, 29))
    df.drop(columns=sorted(extra_columns), inplace=True)

    seed = 2
    dfs = {}
    dfs['train'], dfs['test'] = train_test_split(df, random_state=seed, test_size=0.2)
    dfs['val'], dfs['test'] = train_test_split(
        dfs['test'], random_state=seed, test_size=0.5
    )
    max_target_value = dfs['train']['target'].quantile(0.99)  # type: ignore[code]
    dfs = {k: v[v['target'] <= max_target_value] for k, v in dfs.items()}

    cat_columns = ['c3']
    num_columns = dfs['train'].columns.difference(cat_columns + ['target'])
    _save(
        dataset_dir,
        'Facebook Comments Volume'
        + ('' if keep_derived else ' (without derived features)'),
        TaskType.REGRESSION,
        X_num={k: v[num_columns].astype(np.float32).values for k, v in dfs.items()},
        X_cat={k: v[cat_columns].astype(str).values for k, v in dfs.items()},
        y={k: v['target'].astype(np.float32).values for k, v in dfs.items()},
        idx=None,
        id_='fb-comments--'
        + ('default' if keep_derived else 'without-derived-features'),
    )


def california_housing():
    dataset_dir, _ = _start('california')

    X_num_all, y_all = _get_sklearn_dataset('california_housing')
    idx = _make_split(len(X_num_all), None, 3)

    _save(
        dataset_dir,
        'California Housing',
        TaskType.REGRESSION,
        **_apply_split({'X_num': X_num_all, 'y': y_all}, idx),
        X_cat=None,
        idx=idx,
    )


def covtype():
    dataset_dir, _ = _start('covtype')

    X_num_all, y_all = _get_sklearn_dataset('covtype')
    idx = _make_split(len(X_num_all), y_all, 3)
    data = _apply_split({'X_num': X_num_all, 'y': y_all}, idx)
    data['y'] = {k: v - 1 for k, v in data['y'].items()}

    _save(
        dataset_dir,
        'Covertype',
        TaskType.MULTICLASS,
        **data,
        X_cat=None,
        idx=idx,
    )


def adult():
    dataset_dir, _ = _start('adult')

    df_trainval, df_test = catboost.datasets.adult()
    df_trainval = cast(pd.DataFrame, df_trainval)
    df_test = cast(pd.DataFrame, df_test)
    assert (df_trainval.dtypes == df_test.dtypes).all()
    assert (df_trainval.columns == df_test.columns).all()
    categorical_mask = cast(pd.Series, df_trainval.dtypes != np.float64)

    def get_Xy(df: pd.DataFrame):
        y = (df.pop('income') == '>50K').values.astype('int64')
        return {
            'X_num': df.loc[:, ~categorical_mask].values,
            'X_cat': df.loc[:, categorical_mask].values,
            'y': y,
        }

    data = {k: {'test': v} for k, v in get_Xy(df_test).items()} | {
        'idx': {
            'test': np.arange(
                len(df_trainval), len(df_trainval) + len(df_test), dtype=np.int64
            )
        }
    }
    trainval_data = get_Xy(df_trainval)
    train_val_idx = _make_split(len(df_trainval), trainval_data['y'], 2)
    data['idx'].update(train_val_idx)
    for x in data['X_cat'].values():
        x[x == 'nan'] = CAT_MISSING_VALUE
    for k, v in _apply_split(trainval_data, train_val_idx).items():
        data[k].update(v)

    _save(dataset_dir, 'Adult', TaskType.BINCLASS, **data)


def mslr_web10k():
    dataset_dir, files = _start('microsoft')
    _unzip(files[0], ['Fold1/test.txt', 'Fold1/train.txt', 'Fold1/vali.txt'])
    fold1_dir = dataset_dir / 'Fold1'

    def parse_file(path):
        with open(path) as f:
            rows = []
            for line in f:
                line = line.split()
                rows.append(
                    np.fromiter(
                        (float(item.split(':', 1)[-1]) for item in line),
                        np.float32,
                        len(line),
                    )
                )
            rows = np.array(rows)
        return rows[:, 2:], rows[:, 0], rows[:, 1].astype(np.int64)

    X_num = {}
    y = {}
    groups = {}
    for part in ['train', 'val', 'test']:
        print(f'Parsing {part}...')
        filename = f"{'vali' if part == 'val' else part}.txt"
        X_num[part], y[part], groups[part] = parse_file(fold1_dir / filename)
    shutil.rmtree(fold1_dir)
    for k, v in groups.items():
        np.save(dataset_dir / f'groups_{k}.npy', v)
    _save(
        dataset_dir,
        'MSLR-WEB10K (Fold 1)',
        TaskType.REGRESSION,
        X_num=X_num,
        X_cat=None,
        y=y,
        idx=None,
    )


# %%
def main(argv):
    assert DATA_DIR.exists()
    _set_random_seeds()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Remove everything except for the expected files.',
    )
    args = parser.parse_args(argv[1:])

    if args.clear:
        for dirname, filenames in EXPECTED_FILES.items():
            dataset_dir = DATA_DIR / dirname
            for x in dataset_dir.iterdir():
                if x.name not in filenames:
                    if x.is_dir():
                        shutil.rmtree(x)
                    else:
                        x.unlink()
                    print(f'Removed: {x}')
            if not list(dataset_dir.iterdir()):
                dataset_dir.rmdir()
                print(f'Removed: {dataset_dir}')
        return

    # Below, datasets are grouped by file sources that we use, not by original sources.

    # OpenML
    # eye_movements()
    # gesture_phase()
    # house_16h()
    # higgs_small()

    # Kaggle
    # santander_customer_transactions()
    # otto_group_products()
    # churn_modelling()

    # UCI
    # facebook_comments_volume(True)

    # Python packages
    # california_housing()  # Scikit-Learn
    # covtype()  # Scikit-Learn
    # adult()  # CatBoost

    # Other
    # mslr_web10k()

    print('-----')
    print('Done!')


if __name__ == '__main__':
    sys.exit(main(sys.argv))
