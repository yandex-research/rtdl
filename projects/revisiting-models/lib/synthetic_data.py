"Code used to generate data for experiments with synthetic data"
import math
import typing as ty

import numba
import numpy as np
import torch
import torch.nn as nn
from numba.experimental import jitclass
from tqdm.auto import tqdm


class MLP(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        d_layers: ty.List[int],
        d_out: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(d_layers[i - 1] if i else d_in, x, bias=bias)
                for i, x in enumerate(d_layers)
            ]
        )
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    torch.nn.init.uniform_(m.bias, -bound, bound)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)
        x = self.head(x)
        x = x.squeeze(-1)
        return x


@jitclass(
    spec=[
        ('left_children', numba.int64[:]),
        ('right_children', numba.int64[:]),
        ('feature', numba.int64[:]),
        ('threshold', numba.float32[:]),
        ('value', numba.float32[:]),
        ('is_leaf', numba.int64[:]),
    ]
)
class Tree:
    "Randomly initialized decision tree"

    def __init__(self, n_features, n_nodes, max_depth):
        assert (2 ** np.arange(max_depth + 1)).sum() >= n_nodes, "Too much nodes"

        self.left_children = np.ones(n_nodes, dtype=np.int64) * -1
        self.right_children = np.ones(n_nodes, dtype=np.int64) * -1
        self.feature = np.random.randint(0, n_features, (n_nodes,))
        self.threshold = np.random.randn(n_nodes).astype(np.float32)
        self.value = np.random.randn(n_nodes).astype(np.float32)
        depth = np.zeros(n_nodes, dtype=np.int64)

        # Root is 0
        self.is_leaf = np.zeros(n_nodes, dtype=np.int64)
        self.is_leaf[0] = 1

        # Keep adding nodes while we can (new node must have 2 children)
        while True:
            idx = np.flatnonzero(self.is_leaf)[np.random.choice(self.is_leaf.sum())]
            if depth[idx] < max_depth:
                unused = np.flatnonzero(
                    (self.left_children == -1)
                    & (self.right_children == -1)
                    & ~self.is_leaf
                )
                if len(unused) < 2:
                    break

                lr_child = unused[np.random.permutation(unused.shape[0])[:2]]
                self.is_leaf[lr_child] = 1
                self.is_leaf[lr_child] = 1
                depth[lr_child] = depth[idx] + 1
                self.left_children[idx] = lr_child[0]
                self.right_children[idx] = lr_child[1]
                self.is_leaf[idx] = 0

    def apply(self, x):
        y = np.zeros(x.shape[0])

        for i in range(x.shape[0]):
            idx = 0

            while not self.is_leaf[idx]:
                if x[i, self.feature[idx]] < self.threshold[idx]:
                    idx = self.left_children[idx]
                else:
                    idx = self.right_children[idx]

            y[i] = self.value[idx]

        return y


class TreeEnsemble:
    "Combine multiple trees"

    def __init__(self, *, n_trees, n_features, n_nodes, max_depth):
        self.trees = [
            Tree(n_features=n_features, n_nodes=n_nodes, max_depth=max_depth)
            for _ in range(n_trees)
        ]

    def apply(self, x):
        return np.mean([t.apply(x) for t in tqdm(self.trees)], axis=0)
