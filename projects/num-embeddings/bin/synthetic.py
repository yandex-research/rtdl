# %%
import dataclasses as dc
import math
import typing as ty
from pathlib import Path

import numpy as np
import zero

import lib
from lib import ArrayDict

# %%
AnyDict = ty.Dict[str, ty.Any]


@dc.dataclass
class Dataset:
    X: ArrayDict
    y: ArrayDict
    info: AnyDict
    basename: dc.InitVar[str]

    def __post_init__(self, basename):
        split = 0
        self.info.update(
            {
                'basename': basename,
                'split': split,
                'name': f"{basename}-{split}",
                'task_type': 'regression',
                'n_num_features': self.X['train'].shape[1],
                'n_cat_features': 0,
                **{f"{k}_size": len(v) for k, v in self.X.items()},
            }
        )


def save_dataset(dataset: Dataset, path: Path):
    path.mkdir()
    lib.dump_json(dataset.info, path / 'info.json')
    for name, arrays in [('X_num', dataset.X), ('y', dataset.y)]:
        for part in ['train', 'val', 'test']:
            np.save(path / f'{name}_{part}.npy', arrays[part])


def generate_X(train_size: int, n_features: int, seed: int) -> lib.ArrayDict:
    rng = np.random.default_rng(seed)

    def generate(fraction):
        return rng.standard_normal(
            (int(fraction * train_size), n_features), dtype=np.float32
        )

    return {'train': generate(1.0), 'val': generate(0.2), 'test': generate(0.3)}


@dc.dataclass(frozen=True)
class ObliviousForestTargetConfig:
    n_trees: int
    tree_depth: int
    seed: int


def generate_oblivious_forest_target(X: ArrayDict, config: ObliviousForestTargetConfig):
    zero.improve_reproducibility(config.seed)

    rng = np.random.default_rng(config.seed)
    X_all = np.concatenate([X['train'], X['val'], X['test']])
    n_features = X_all.shape[1]
    bounds = np.column_stack([X_all.min(0), X_all.max(0)])

    n_splits = config.n_trees * config.tree_depth
    tree_feature_indices = list(range(n_features)) * math.ceil(n_splits / n_features)
    tree_feature_indices = tree_feature_indices[:n_splits]
    tree_feature_indices = rng.permutation(tree_feature_indices)
    tree_feature_indices = tree_feature_indices.reshape(
        config.n_trees, config.tree_depth
    )
    tree_feature_indices = tree_feature_indices.tolist()

    trees = []
    for feature_indices in tree_feature_indices:
        thresholds = []
        for feature_index in feature_indices:
            threshold = bounds[feature_index][0]
            while (
                threshold <= bounds[feature_index][0]
                or threshold >= bounds[feature_index][1]
            ):
                threshold = rng.standard_normal()
            thresholds.append(threshold)
        thresholds = np.array(thresholds)

        leaf_values = rng.standard_normal((2,) * config.tree_depth, dtype='float32')
        targets = {
            k: leaf_values[
                tuple((v[:, feature_indices] < thresholds[None]).astype('int64').T)
            ]
            for k, v in X.items()
        }
        trees.append((feature_indices, thresholds, leaf_values, targets))
    return (
        {part: np.stack([tree[-1][part]]).mean(0) for part in X for tree in trees},
        {'tree_feature_indices': tree_feature_indices},
    )


# %%
def main():
    seed = 0
    for train_size in [100_000]:
        n_features = 8
        synthetic_path = lib.PROJ / 'data' / 'synthetic'
        synthetic_path.mkdir(parents=True, exist_ok=True)

        X_path = synthetic_path / 'X.npz'
        X = generate_X(train_size, n_features, seed)
        np.savez(X_path, **X)

        config = ObliviousForestTargetConfig(16, 6, seed)
        y, info = generate_oblivious_forest_target(X, config)
        dataset = Dataset(
            X,
            y,
            {**dc.asdict(config), **info},
            f"oblivious_forest_{train_size}_{config.n_trees}_{config.tree_depth}",
        )
        save_dataset(dataset, synthetic_path / dataset.info['basename'])


main()
