"models from https://github.com/sbadirli/GrowNet/"
# %%

import argparse
import math
import typing as ty
from enum import Enum
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import zero
from tqdm.auto import tqdm

import lib


# %%
class ForwardType(Enum):
    SIMPLE = 0
    STACKED = 1
    CASCADE = 2
    GRADIENT = 3


class DynamicNet:
    def __init__(
        self,
        c0,
        lr,
        categories: ty.Optional[ty.List[int]],
        d_embedding: ty.Optional[int],
    ):
        self.models = []
        self.c0 = c0
        self.lr = lr
        self.boost_rate = nn.Parameter(
            torch.tensor(lr, requires_grad=True, device="cuda")
        )
        if categories is not None:
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.category_offsets = category_offsets
            self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')
        else:
            self.category_embeddings = None
            self.category_offsets = None

    def add(self, model):
        self.models.append(model)

    def parameters(self):
        params = []
        for m in self.models:
            params.extend(m.parameters())

        if self.category_embeddings is not None:
            params.extend(self.category_embeddings.parameters())
            params.append(self.boost_rate)
        return params

    def zero_grad(self):
        for m in self.models:
            m.zero_grad()

    def to_cuda(self):
        for m in self.models:
            m.cuda()
        if self.category_embeddings is not None:
            self.category_embeddings = self.category_embeddings.cuda()
            self.category_offsets = self.category_offsets.cuda()

    def to_eval(self):
        for m in self.models:
            m.eval()

    def to_train(self):
        for m in self.models:
            m.train(True)

    def embed_input(self, x_num, x_cat):
        if x_cat is not None:
            x_cat = self.category_embeddings(x_cat + self.category_offsets[None])
            x = torch.cat([x_num, x_cat.view(x_cat.size(0), -1)], dim=-1)
        else:
            x = x_num
        return x

    def forward(self, x_num, x_cat):
        if len(self.models) == 0:
            return None, self.c0
        middle_feat_cum = None
        prediction = None
        with torch.no_grad():
            for m in self.models:
                if middle_feat_cum is None:
                    middle_feat_cum, prediction = m(
                        self.embed_input(x_num, x_cat), middle_feat_cum
                    )
                else:
                    middle_feat_cum, pred = m(
                        self.embed_input(x_num, x_cat), middle_feat_cum
                    )
                    prediction += pred
        return middle_feat_cum, self.c0 + self.boost_rate * prediction

    def forward_grad(self, x_num, x_cat):
        if len(self.models) == 0:
            return None, self.c0
        # at least one model
        middle_feat_cum = None
        prediction = None
        for m in self.models:
            if middle_feat_cum is None:
                middle_feat_cum, prediction = m(
                    self.embed_input(x_num, x_cat), middle_feat_cum
                )
            else:
                middle_feat_cum, pred = m(
                    self.embed_input(x_num, x_cat), middle_feat_cum
                )
                prediction += pred
        return middle_feat_cum, self.c0 + self.boost_rate * prediction

    @classmethod
    def from_file(cls, path, builder):
        d = torch.load(path)
        net = DynamicNet(d['c0'], d['lr'], categories=None, d_embedding=None)
        net.boost_rate = d['boost_rate']
        if 'category_embeddings' in d:
            net.category_embeddings = d['category_embeddings']
            net.category_offsets = d['category_offsets']
        for stage, m in enumerate(d['models']):
            submod = builder(stage)
            submod.load_state_dict(m)
            net.add(submod)
        return net

    def to_file(self, path):
        models = [m.state_dict() for m in self.models]
        d = {
            'models': models,
            'c0': self.c0,
            'lr': self.lr,
            'boost_rate': self.boost_rate,
            'category_embeddings': self.category_embeddings,
            'category_offsets': self.category_offsets,
        }
        torch.save(d, path)


class SpLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = (input.t().mm(grad_output)).t()
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias


splinear = SpLinearFunc.apply


class SpLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(SpLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)
        # TODO write a default initialization
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return splinear(input, self.weight, self.bias)


class MLP_2HL(nn.Module):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, sparse=False, bn=True):
        super(MLP_2HL, self).__init__()
        self.in_layer = (
            SpLinear(dim_in, dim_hidden1) if sparse else nn.Linear(dim_in, dim_hidden1)
        )
        self.dropout_layer = nn.Dropout(0.0)
        self.lrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.hidden_layer = nn.Linear(dim_hidden1, dim_hidden2)
        self.out_layer = nn.Linear(dim_hidden2, 1)
        self.bn = nn.BatchNorm1d(dim_hidden1)
        self.bn2 = nn.BatchNorm1d(dim_in)

    def forward(self, x, lower_f):
        if lower_f is not None:
            x = torch.cat([x, lower_f], dim=1)
            x = self.bn2(x)
        out = self.lrelu(self.in_layer(x))
        out = self.bn(out)
        out = self.hidden_layer(out)
        return out, self.out_layer(self.relu(out)).squeeze()

    @classmethod
    def get_model(cls, stage, opt):
        if stage == 0:
            dim_in = opt.feat_d
        else:
            dim_in = opt.feat_d + opt.hidden_d
        model = MLP_2HL(dim_in, opt.hidden_d, opt.hidden_d, opt.sparse)
        return model


def init_gbnn(y):
    positive = negative = 0
    for i in range(len(y)):
        if y[i] > 0:
            positive += 1
        else:
            negative += 1
    blind_acc = max(positive, negative) / (positive + negative)
    print(f'Blind accuracy: {blind_acc}')
    return float(np.log(positive / negative))


# %%

args, output = lib.load_config()

zero.set_randomness(args['seed'])
dataset_dir = lib.get_path(args['data']['path'])
stats: ty.Dict[str, ty.Any] = {
    'dataset': dataset_dir.name,
    'algorithm': Path(__file__).stem,
    **lib.load_json(output / 'stats.json'),
}
timer = zero.Timer()
timer.run()

D = lib.Dataset.from_dir(dataset_dir)
X = D.build_X(
    normalization=args['data'].get('normalization'),
    num_nan_policy='mean',
    cat_nan_policy='new',
    cat_policy=args['data'].get('cat_policy', 'counter'),
    cat_min_frequency=args['data'].get('cat_min_frequency', 0.0),
    seed=args['seed'],
)
if not isinstance(X, tuple):
    X = (X, None)

zero.set_randomness(args['seed'])
Y, y_info = D.build_y(args['data'].get('y_policy'))
lib.dump_pickle(y_info, output / 'y_info.pickle')
X = tuple(None if x is None else lib.to_tensors(x) for x in X)
Y = lib.to_tensors(Y)
device = lib.get_device()
if device.type != 'cpu':
    X = tuple(None if x is None else {k: v.to(device) for k, v in x.items()} for x in X)
    Y_device = {k: v.to(device) for k, v in Y.items()}
else:
    Y_device = Y
X_num, X_cat = X
if not D.is_multiclass:
    Y_device = {k: v.float() for k, v in Y_device.items()}

train_size = len(X_num[lib.TRAIN])

args["model"]["feat_d"] = (
    X_num[lib.TRAIN].shape[1] + 0
    if X_cat is None
    else X_cat[lib.TRAIN].shape[1] * args["model"].get("d_embedding", 0)
)

batch_size, epoch_size = (
    stats['batch_size'],
    stats['epoch_size'],
) = lib.get_epoch_parameters(train_size, args['training'].get('batch_size', 'v3'))


loss_f1 = nn.MSELoss(reduction='none') if D.is_binclass else nn.MSELoss()
loss_f2 = nn.BCEWithLogitsLoss() if D.is_binclass else nn.MSELoss()

if D.is_binclass:
    c0 = init_gbnn(Y_device[lib.TRAIN])
else:
    c0 = Y_device[lib.TRAIN].mean()

net_ensemble = DynamicNet(
    c0,
    float(args["model"]["boost_rate"]),
    categories=lib.get_categories(X_cat),
    d_embedding=args["model"].get("d_embedding", None),
)
if device.type != 'cpu':
    net_ensemble.to_cuda()
print(device)

progress = zero.ProgressTracker(args["patience"])


@torch.no_grad()
def predict(part):
    loader = lib.IndexLoader(
        len(X_num[part]), args['training']['eval_batch_size'], False, device
    )
    preds = []
    for idx in loader:
        _, out = net_ensemble.forward(
            X_num[part][idx], None if X_cat is None else X_cat[part][idx]
        )
        preds.append(out)
    return torch.cat(preds).cpu()


@torch.no_grad()
def evaluate(parts):
    metrics = {}
    predictions = {}
    for part in parts:
        predictions[part] = predict(part).numpy()
        metrics[part] = lib.calculate_metrics(
            D.info['task_type'],
            Y[part].numpy(),  # type: ignore[code]
            predictions[part],  # type: ignore[code]
            'logits',
            y_info,
        )

    for part, part_metrics in metrics.items():
        print(f'[{part:<5}]', lib.make_summary(part_metrics))

    return metrics, predictions


for s in range(args["model"]["num_nets"]):
    m = MLP_2HL.get_model(s, argparse.Namespace(**args["model"])).to(device)
    opt = torch.optim.Adam(
        m.parameters(),
        args["training"]["lr"],
        weight_decay=args["training"]["weight_decay"],
    )
    net_ensemble.to_train()

    loader = lib.IndexLoader(train_size, batch_size, True, device)

    timer = zero.Timer()
    timer.run()
    # Train small model step
    for e in range(args["model"]["epochs_per_stage"]):
        for batch_idx in tqdm(loader, leave=False):
            x_num, x_cat = (
                X_num[lib.TRAIN][batch_idx],
                None if X_cat is None else X_cat[lib.TRAIN][batch_idx],
            )
            y = Y_device[lib.TRAIN][batch_idx].unsqueeze(1)

            middle_feat, out = net_ensemble.forward(x_num, x_cat)
            out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)

            # Classification
            if D.is_binclass:
                h = 1 / ((1 + torch.exp(y * out)) * (1 + torch.exp(-y * out)))
                grad_direction = y * (1.0 + torch.exp(-y * out))
                out = torch.as_tensor(out)
                nwtn_weights = (torch.exp(out) + torch.exp(-out)).abs()
                _, out = m(net_ensemble.embed_input(x_num, x_cat), middle_feat)

                out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
                loss = loss_f1(net_ensemble.boost_rate * out, grad_direction)  # T
                loss = loss * h
                loss = loss.mean()
            # Regression
            else:
                grad_direction = -(out - y)
                _, out = m(net_ensemble.embed_input(x_num, x_cat), middle_feat)
                out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
                loss = loss_f1(net_ensemble.boost_rate * out, grad_direction)  # T

            m.zero_grad()
            loss.backward()
            opt.step()
            break

        break
    net_ensemble.add(m)
    print(f"Time to train one model: {zero.format_seconds(timer())}")

    # Corrective step
    timer.reset()
    timer.run()
    if s > 0:
        lr_scaler = args["training"]["lr_scaler"]

        if s % 15 == 0:
            args["training"]["lr"] /= 2
            args["training"]["weight_decay"] /= 2

        opt = torch.optim.Adam(
            net_ensemble.parameters(),
            args["training"]["lr"] / lr_scaler,
            weight_decay=args["training"]["weight_decay"],
        )

        for e in range(args["model"]["correct_epoch"]):
            for batch_idx in tqdm(loader, leave=False):
                x_num, x_cat = (
                    X_num[lib.TRAIN][batch_idx],
                    None if X_cat is None else X_cat[lib.TRAIN][batch_idx],
                )
                y = Y_device[lib.TRAIN][batch_idx].unsqueeze(1)

                # Binclass
                if D.is_binclass:
                    _, out = net_ensemble.forward_grad(x_num, x_cat)
                    out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
                    loss = loss_f2(out, y)
                else:
                    _, out = net_ensemble.forward_grad(x_num, x_cat)
                    out = torch.as_tensor(out, dtype=torch.float32).cuda().view(-1, 1)
                    loss = loss_f2(out, y)

                opt.zero_grad()
                loss.backward()
                opt.step()
    print(f"Time for corrective step: {zero.format_seconds(timer())}")
    # Evaluations and printing

    timer.reset()
    timer.run()

    net_ensemble.to_file(output / "ensemble.pt")
    net_ensemble = DynamicNet.from_file(
        output / "ensemble.pt",
        lambda stage: MLP_2HL.get_model(stage, argparse.Namespace(**args["model"])),
    )
    if device.type != 'cpu':
        net_ensemble.to_cuda()

    net_ensemble.to_eval()
    print(f"Done with stage {s};")
    metrics, predictions = evaluate([lib.VAL, lib.TEST])
    print(f"Evaluation time {zero.format_seconds(timer())}")

    progress.update(metrics[lib.VAL]["score"])

    if progress.success:
        print("New best stage")
        net_ensemble.to_file(output / "final.pt")
        stats["metrics"] = metrics
    elif progress.fail:
        print("Early stopping")
        break


net_ensemble = DynamicNet.from_file(
    output / "final.pt",
    lambda stage: MLP_2HL.get_model(stage, argparse.Namespace(**args["model"])),
)
if device.type != 'cpu':
    net_ensemble.to_cuda()

stats['metrics'], predictions = evaluate(lib.PARTS)
lib.dump_stats(stats, output, final=True)
for k, v in predictions.items():
    np.save(output / f'p_{k}.npy', v)
lib.backup_output(output)

# %%
