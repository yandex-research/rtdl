from copy import deepcopy

import numba
import numpy as np
import torch
import torch.nn.functional as F
import zero
from catboost import CatBoostRegressor
from tqdm.auto import tqdm

import lib
from bin.ft_transformer import Transformer
from bin.resnet import ResNet
from lib.synthetic_data import MLP, TreeEnsemble

device = lib.get_device()
print(device)


def get_data(args):
    "Generate two datasets X, RandomMLP(X) and X, RandomForest(X)"

    zero.set_randomness(args["seed"])
    ptr = numba._helperlib.rnd_get_np_state_ptr()
    ints, index = np.random.get_state()[1:3]
    numba._helperlib.rnd_set_state(ptr, (index, [int(x) for x in ints]))

    mlp_generator = MLP(
        d_in=args["n_informative"],
        d_out=1,
        d_layers=[args["mlp_d_layer"]] * args["mlp_n_layers"],
        bias=args["mlp_bias"],
    ).to(device)

    tree_generator = TreeEnsemble(
        n_features=args["n_informative"],
        n_trees=args["tree_n_trees"],
        n_nodes=args["tree_n_nodes"],
        max_depth=args["tree_max_depth"],
    )
    X = {}
    Y_mlp = {}
    Y_tree = {}

    for part in lib.PARTS:
        X[part] = torch.randn(args["dataset_size"][part], args["n_features"])
        with torch.no_grad():
            Y_mlp[part] = mlp_generator(
                X[part][:, : args["n_informative"]].to(device)
            ).cpu()
            Y_tree[part] = torch.from_numpy(
                tree_generator.apply(X[part][:, : args["n_informative"]].numpy())
            ).float()

    Y_tree = {
        k: (v - Y_tree[lib.TRAIN].mean()) / Y_tree[lib.TRAIN].std()
        + torch.randn(len(v)) * args["noise_std"]
        for k, v in Y_tree.items()
    }

    Y_mlp = {
        k: (v - Y_mlp[lib.TRAIN].mean()) / Y_mlp[lib.TRAIN].std()
        + torch.randn(len(v)) * args["noise_std"]
        for k, v in Y_mlp.items()
    }

    return X, Y_tree, Y_mlp


def train_eval_gbdt(model, x, y):
    model.fit(
        x[lib.TRAIN].numpy(),
        y[lib.TRAIN].numpy(),
        eval_set=(x[lib.VAL].numpy(), y[lib.VAL].numpy()),
    )
    metrics = {}

    for part in lib.PARTS:
        metrics[part] = lib.calculate_metrics(
            "regression",
            y[part].numpy(),
            model.predict(x[part].numpy()),
            'logits',
            y_info={"mean": 0, "std": 1, "policy": "mean_std"},
        )['rmse']

    train_log = None
    if getattr(model, "evals_result_", None) is not None:
        train_log = {
            lib.TRAIN: np.array(model.evals_result_['learn']['RMSE']),
            lib.VAL: np.array(model.evals_result_['validation']['RMSE']),
        }

    return model, train_log, metrics


def train_eval_net(model, X, Y, lr=1e-4):
    def apply_model(x):
        return model(x, x_cat=None)

    @torch.no_grad()
    def evaluate(parts):
        model.eval()
        metrics = {}
        predictions = {}

        for part in parts:
            # compute train loss for visualization only
            if part == lib.TRAIN:
                x_eval = X[part][:50_000]
                y_eval = Y[part][:50_000]
            else:
                x_eval = X[part]
                y_eval = Y[part]

            predictions[part] = (
                torch.cat(
                    [
                        apply_model(x_eval.to(device)[idx])
                        for idx in lib.IndexLoader(len(x_eval), 1024, False, device)
                    ]
                )
                .cpu()
                .numpy()
            )

            metrics[part] = lib.calculate_metrics(
                'regression',
                y_eval.numpy(),
                predictions[part],
                'logits',
                y_info={"std": 1, "mean": 0, "policy": "mean_std"},
            )['rmse']
        return metrics

    tqdm._instances.clear()
    train_size = len(X[lib.TRAIN])
    batch_size = 256
    loss_fn = F.mse_loss

    device = next(model.parameters()).device
    X_device = {part: X[part].to(device) for part in X.keys()}
    Y_device = {part: Y[part].to(device) for part in Y.keys()}

    optimizer = lib.make_optimizer('adamw', model.parameters(), lr, 1e-5)
    stream = zero.Stream(lib.IndexLoader(train_size, batch_size, True, device))
    progress = zero.ProgressTracker(10, min_delta=0.001)

    training_log = {lib.TRAIN: [], lib.VAL: []}
    best_model = None

    timer = zero.Timer()
    metrics = evaluate([lib.TRAIN, lib.VAL])
    for k, v in metrics.items():
        training_log[k].append(v)

    timer.run()
    for epoch in stream.epochs(1000):
        model.train()

        for batch_idx in epoch:
            optimizer.zero_grad()
            loss = loss_fn(
                apply_model(X_device[lib.TRAIN][batch_idx]),
                Y_device[lib.TRAIN][batch_idx],
            )
            loss.backward()
            optimizer.step()

        metrics = evaluate([lib.TRAIN, lib.VAL])
        for k, v in metrics.items():
            training_log[k].append(v)

        progress.update(metrics[lib.VAL]['score'])
        print(
            "Epoch done;",
            ' '.join([f"{p} {metrics[p]['rmse']:.5f}" for p in [lib.TRAIN, lib.VAL]]),
        )
        if progress.success:
            best_model = deepcopy(model)
            print("New best")

        elif progress.fail:
            break

    model = best_model
    metrics = evaluate(lib.PARTS)

    return best_model, training_log, metrics


if __name__ == "__main__":
    args, output = lib.load_config()
    stats = lib.load_json(output / "stats.json")

    x, y_tree, y_mlp = get_data(args)

    torch.save(
        {
            "x": x,
            "y_tree": y_tree,
            "y_mlp": y_mlp,
        },
        output / "data.pt",
    )

    stats["metrics"] = []

    for i, a in tqdm(enumerate(args["coefs"])):
        out = {}

        y_combined = {k: y_tree[k] * a + y_mlp[k] * (1 - a) for k in y_mlp.keys()}
        y_combined = {
            k: (v - y_combined[lib.TRAIN].mean()) / y_combined[lib.TRAIN].std()
            for k, v in y_combined.items()
        }

        if args["model"] == "gbdt":
            model = CatBoostRegressor(**args["model_config"])
        elif args["model"] == "resnet":
            model = ResNet(
                d_numerical=args["n_features"],
                categories=None,
                d_embedding=None,
                d_out=1,
                **args["model_config"],
            ).to(device)
        elif args["model"] == "transformer":
            model = Transformer(
                d_numerical=args["n_features"],
                d_out=1,
                categories=None,
                kv_compression=None,
                kv_compression_sharing=None,
                **args["model_config"],
            ).to(device)

        if args["model"] in ["resnet", "transformer"]:
            model, log, metrics = train_eval_net(model, x, y_combined)
            out["model"] = model.state_dict()
        else:
            model, log, metrics = train_eval_gbdt(model, x, y_combined)
            model.save_model(str(output / f"model_{i}.cbm"))

        out["log"] = log
        out["metrics"] = metrics

        stats["metrics"].append(metrics)

        torch.save(out, output / f"outputs_{i}.pt")
    lib.dump_stats(stats, output, final=True)
