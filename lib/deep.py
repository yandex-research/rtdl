from __future__ import absolute_import, division, print_function

import math
import os
import typing as ty
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import zero
from torch import Tensor


class IndexLoader:
    def __init__(
        self, train_size: int, batch_size: int, shuffle: bool, device: torch.device
    ) -> None:
        self._train_size = train_size
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._device = device

    def __len__(self) -> int:
        return math.ceil(self._train_size / self._batch_size)

    def __iter__(self):
        indices = list(
            zero.iloader(self._train_size, self._batch_size, shuffle=self._shuffle)
        )
        return iter(torch.cat(indices).to(self._device).split(self._batch_size))


class Lambda(nn.Module):
    def __init__(self, f: ty.Callable) -> None:
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


# Source: https://github.com/bzhangGo/rmsnorm
# NOTE: eps is changed to 1e-5
class RMSNorm(nn.Module):
    def __init__(self, d, p=-1.0, eps=1e-5, bias=False):
        """Root Mean Square Layer Normalization

        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


class ScaleNorm(nn.Module):
    """
    Sources:
    - https://github.com/tnq177/transformers_without_tears/blob/25026061979916afb193274438f7097945acf9bc/layers.py#L132
    - https://github.com/tnq177/transformers_without_tears/blob/6b2726cd9e6e642d976ae73b9f696d9d7ff4b395/layers.py#L157
    """

    def __init__(self, d: int, eps: float = 1e-5, clamp: bool = False) -> None:
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(d ** 0.5))
        self.eps = eps
        self.clamp = clamp

    def forward(self, x):
        norms = torch.norm(x, dim=-1, keepdim=True)
        norms = norms.clamp(min=self.eps) if self.clamp else norms + self.eps
        return self.scale * x / norms


def reglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


class ReGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)


class GEGLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)


def make_optimizer(
    optimizer: str,
    parameter_groups,
    lr: float,
    weight_decay: float,
) -> optim.Optimizer:
    Optimizer = {
        'adabelief': AdaBelief,
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'radam': RAdam,
        'sgd': optim.SGD,
    }[optimizer]
    momentum = (0.9,) if Optimizer is optim.SGD else ()
    return Optimizer(parameter_groups, lr, *momentum, weight_decay=weight_decay)


def make_lr_schedule(
    optimizer: optim.Optimizer,
    lr: float,
    epoch_size: int,
    lr_schedule: ty.Optional[ty.Dict[str, ty.Any]],
) -> ty.Tuple[
    ty.Optional[optim.lr_scheduler._LRScheduler],
    ty.Dict[str, ty.Any],
    ty.Optional[int],
]:
    if lr_schedule is None:
        lr_schedule = {'type': 'constant'}
    lr_scheduler = None
    n_warmup_steps = None
    if lr_schedule['type'] in ['transformer', 'linear_warmup']:
        n_warmup_steps = (
            lr_schedule['n_warmup_steps']
            if 'n_warmup_steps' in lr_schedule
            else lr_schedule['n_warmup_epochs'] * epoch_size
        )
    elif lr_schedule['type'] == 'cyclic':
        lr_scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=lr,
            max_lr=lr_schedule['max_lr'],
            step_size_up=lr_schedule['n_epochs_up'] * epoch_size,
            step_size_down=lr_schedule['n_epochs_down'] * epoch_size,
            mode=lr_schedule['mode'],
            gamma=lr_schedule.get('gamma', 1.0),
            cycle_momentum=False,
        )
    return lr_scheduler, lr_schedule, n_warmup_steps


def get_activation_fn(name: str) -> ty.Callable[[Tensor], Tensor]:
    return (
        reglu
        if name == 'reglu'
        else geglu
        if name == 'geglu'
        else torch.sigmoid
        if name == 'sigmoid'
        else getattr(F, name)
    )


def get_nonglu_activation_fn(name: str) -> ty.Callable[[Tensor], Tensor]:
    return (
        F.relu
        if name == 'reglu'
        else F.gelu
        if name == 'geglu'
        else get_activation_fn(name)
    )


def load_swa_state_dict(model: nn.Module, swa_model: optim.swa_utils.AveragedModel):
    state_dict = deepcopy(swa_model.state_dict())
    del state_dict['n_averaged']
    model.load_state_dict({k[len('module.') :]: v for k, v in state_dict.items()})


def get_epoch_parameters(
    train_size: int, batch_size: ty.Union[int, str]
) -> ty.Tuple[int, int]:
    if isinstance(batch_size, str):
        if batch_size == 'v3':
            batch_size = (
                256 if train_size < 50000 else 512 if train_size < 100000 else 1024
            )
        elif batch_size == 'v1':
            batch_size = (
                16
                if train_size < 1000
                else 32
                if train_size < 10000
                else 64
                if train_size < 50000
                else 128
                if train_size < 100000
                else 256
                if train_size < 200000
                else 512
                if train_size < 500000
                else 1024
            )
        elif batch_size == 'v2':
            batch_size = (
                512 if train_size < 100000 else 1024 if train_size < 500000 else 2048
            )
    return batch_size, math.ceil(train_size / batch_size)  # type: ignore[code]


def get_linear_warmup_lr(lr: float, n_warmup_steps: int, step: int) -> float:
    assert step > 0, "1-based enumeration of steps is expected"
    return min(lr, step / (n_warmup_steps + 1) * lr)


def get_manual_lr(schedule: ty.List[float], epoch: int) -> float:
    assert epoch > 0, "1-based enumeration of epochs is expected"
    return schedule[min(epoch, len(schedule)) - 1]


def get_transformer_lr(scale: float, d: int, n_warmup_steps: int, step: int) -> float:
    return scale * d ** -0.5 * min(step ** -0.5, step * n_warmup_steps ** -1.5)


def learn(model, optimizer, loss_fn, step, batch, star) -> ty.Tuple[Tensor, ty.Any]:
    model.train()
    optimizer.zero_grad()
    out = step(batch)
    loss = loss_fn(*out) if star else loss_fn(out)
    loss.backward()
    optimizer.step()
    return loss, out


def _learn_with_virtual_batch(
    model, optimizer, loss_fn, step, batch, chunk_size
) -> Tensor:
    batch_size = len(batch)
    if chunk_size >= batch_size:
        return learn(model, optimizer, loss_fn, step, batch, True)[0]
    model.train()
    optimizer.zero_grad()
    total_loss = None
    for chunk in zero.iter_batches(batch, chunk_size):
        loss = loss_fn(*step(chunk))
        loss = loss * len(chunk)
        loss.backward()
        if total_loss is None:
            total_loss = loss.detach()
        else:
            total_loss += loss.detach()
    for x in model.parameters():
        if x.grad is not None:
            x.grad /= batch_size
    optimizer.step()
    return total_loss / batch_size


def learn_with_auto_virtual_batch(
    model,
    optimizer,
    loss_fn,
    step,
    batch,
    batch_size_hint: int,
    chunk_size: ty.Optional[int],
) -> ty.Tuple[Tensor, ty.Optional[int]]:
    """This is just an overcomplicated version of `train_with_auto_virtual_batch`."""
    random_state = zero.get_random_state()
    while chunk_size != 0:
        try:
            zero.set_random_state(random_state)
            return (
                _learn_with_virtual_batch(
                    model,
                    optimizer,
                    loss_fn,
                    step,
                    batch,
                    chunk_size or batch_size_hint,
                ),
                chunk_size,
            )
        except RuntimeError as err:
            if not is_oom_exception(err):
                raise
            if chunk_size is None:
                chunk_size = batch_size_hint
            chunk_size //= 2
    raise RuntimeError('Not enough memory even for batch_size=1')


def train_with_auto_virtual_batch(
    optimizer,
    loss_fn,
    step,
    batch,
    chunk_size: int,
) -> ty.Tuple[Tensor, int]:
    batch_size = len(batch)
    random_state = zero.get_random_state()
    while chunk_size != 0:
        try:
            zero.set_random_state(random_state)
            optimizer.zero_grad()
            if batch_size <= chunk_size:
                loss = loss_fn(*step(batch))
                loss.backward()
            else:
                loss = None
                for chunk in zero.iter_batches(batch, chunk_size):
                    chunk_loss = loss_fn(*step(chunk))
                    chunk_loss = chunk_loss * (len(chunk) / batch_size)
                    chunk_loss.backward()
                    if loss is None:
                        loss = chunk_loss.detach()
                    else:
                        loss += chunk_loss.detach()
        except RuntimeError as err:
            if not is_oom_exception(err):
                raise
            chunk_size //= 2
        else:
            break
    if not chunk_size:
        raise RuntimeError('Not enough memory even for batch_size=1')
    optimizer.step()
    return loss, chunk_size  # type: ignore[code]


def tensor(x) -> torch.Tensor:
    assert isinstance(x, torch.Tensor)
    return ty.cast(torch.Tensor, x)


def get_n_parameters(m: nn.Module):
    return sum(x.numel() for x in m.parameters() if x.requires_grad)


def get_mlp_n_parameters(units: ty.List[int]):
    x = 0
    for a, b in zip(units, units[1:]):
        x += a * b + b
    return x


def get_lr(optimizer: optim.Optimizer) -> float:
    return next(iter(optimizer.param_groups))['lr']


def set_lr(optimizer: optim.Optimizer, lr: float) -> None:
    for x in optimizer.param_groups:
        x['lr'] = lr


def get_device() -> torch.device:
    return torch.device('cuda:0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu')


@torch.no_grad()
def get_gradient_norm_ratios(m: nn.Module):
    return {
        k: v.grad.norm() / v.norm()
        for k, v in m.named_parameters()
        if v.grad is not None
    }


def is_oom_exception(err: RuntimeError) -> bool:
    return any(
        x in str(err)
        for x in [
            'CUDA out of memory',
            'CUBLAS_STATUS_ALLOC_FAILED',
            'CUDA error: out of memory',
        ]
    )


# Source: https://github.com/LiyuanLucasLiu/RAdam
class RAdam(optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        degenerated_to_sgd=True,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if (
            isinstance(params, (list, tuple))
            and len(params) > 0
            and isinstance(params[0], dict)
        ):
            for param in params:
                if 'betas' in param and (
                    param['betas'][0] != betas[0] or param['betas'][1] != betas[1]
                ):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            buffer=[[None, None, None] for _ in range(10)],
        )
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t)
                            * (N_sma - 4)
                            / (N_sma_max - 4)
                            * (N_sma - 2)
                            / N_sma
                            * N_sma_max
                            / (N_sma_max - 2)
                        ) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(
                            -group['weight_decay'] * group['lr'], p_data_fp32
                        )
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(
                            -group['weight_decay'] * group['lr'], p_data_fp32
                        )
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


version_higher = torch.__version__ >= "1.5.0"


# Source: https://github.com/juntang-zhuang/Adabelief-Optimizer
class AdaBelief(optim.Optimizer):
    r"""Implements AdaBelief algorithm. Modified from Adam in PyTorch
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-16)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        weight_decouple (boolean, optional): ( default: True) If set as True, then
            the optimizer uses decoupled weight decay as in AdamW
        fixed_decay (boolean, optional): (default: False) This is used when weight_decouple
            is set as True.
            When fixed_decay == True, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay$.
            When fixed_decay == False, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay \times lr$. Note that in this case, the
            weight decay ratio decreases with learning rate (lr).
        rectify (boolean, optional): (default: True) If set as True, then perform the rectified
            update similar to RAdam
        degenerated_to_sgd (boolean, optional) (default:True) If set as True, then perform SGD update
            when variance of gradient is high
        print_change_log (boolean, optional) (default: True) If set as True, print the modifcation to
            default hyper-parameters
    reference: AdaBelief Optimizer, adapting stepsizes by the belief in observed gradients, NeurIPS 2020
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-16,
        weight_decay=0,
        amsgrad=False,
        weight_decouple=True,
        fixed_decay=False,
        rectify=True,
        degenerated_to_sgd=True,
        print_change_log=True,
    ):

        # ------------------------------------------------------------------------------
        # Print modifications to default arguments
        if print_change_log:
            print(
                'Please check your arguments if you have upgraded adabelief-pytorch from version 0.0.5.'
            )
            print('Modifications to default arguments:')
            default_table = [
                ['eps', 'weight_decouple', 'rectify'],
                ['adabelief-pytorch=0.0.5', '1e-8', 'False', 'False'],
                ['>=0.1.0 (Current 0.2.0)', '1e-16', 'True', 'True'],
            ]
            print(default_table)

            recommend_table = [
                [
                    'SGD better than Adam (e.g. CNN for Image Classification)',
                    'Adam better than SGD (e.g. Transformer, GAN)',
                ],
                ['Recommended eps = 1e-8', 'Recommended eps = 1e-16'],
            ]
            print(recommend_table)

            print('For a complete table of recommended hyperparameters, see')
            print('https://github.com/juntang-zhuang/Adabelief-Optimizer')

            print(
                'You can disable the log message by setting "print_change_log = False", though it is recommended to keep as a reminder.'
            )
        # ------------------------------------------------------------------------------

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if (
            isinstance(params, (list, tuple))
            and len(params) > 0
            and isinstance(params[0], dict)
        ):
            for param in params:
                if 'betas' in param and (
                    param['betas'][0] != betas[0] or param['betas'][1] != betas[1]
                ):
                    param['buffer'] = [[None, None, None] for _ in range(10)]

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            buffer=[[None, None, None] for _ in range(10)],
        )
        super(AdaBelief, self).__init__(params, defaults)

        self.degenerated_to_sgd = degenerated_to_sgd
        self.weight_decouple = weight_decouple
        self.rectify = rectify
        self.fixed_decay = fixed_decay
        if self.weight_decouple:
            print('Weight decoupling enabled in AdaBelief')
            if self.fixed_decay:
                print('Weight decay fixed')
        if self.rectify:
            print('Rectification enabled in AdaBelief')
        if amsgrad:
            print('AMSGrad enabled in AdaBelief')

    def __setstate__(self, state):
        super(AdaBelief, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                amsgrad = group['amsgrad']

                # State initialization
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = (
                    torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if version_higher
                    else torch.zeros_like(p.data)
                )

                # Exponential moving average of squared gradient values
                state['exp_avg_var'] = (
                    torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if version_higher
                    else torch.zeros_like(p.data)
                )

                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_var'] = (
                        torch.zeros_like(p.data, memory_format=torch.preserve_format)
                        if version_higher
                        else torch.zeros_like(p.data)
                    )

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # cast data type
                half_precision = False
                if p.data.dtype == torch.float16:
                    half_precision = True
                    p.data = p.data.float()
                    p.grad = p.grad.float()

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'AdaBelief does not support sparse gradients, please consider SparseAdam instead'
                    )
                amsgrad = group['amsgrad']

                state = self.state[p]

                beta1, beta2 = group['betas']

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = (
                        torch.zeros_like(p.data, memory_format=torch.preserve_format)
                        if version_higher
                        else torch.zeros_like(p.data)
                    )
                    # Exponential moving average of squared gradient values
                    state['exp_avg_var'] = (
                        torch.zeros_like(p.data, memory_format=torch.preserve_format)
                        if version_higher
                        else torch.zeros_like(p.data)
                    )
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_var'] = (
                            torch.zeros_like(
                                p.data, memory_format=torch.preserve_format
                            )
                            if version_higher
                            else torch.zeros_like(p.data)
                        )

                # perform weight decay, check if decoupled weight decay
                if self.weight_decouple:
                    if not self.fixed_decay:
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        p.data.mul_(1.0 - group['weight_decay'])
                else:
                    if group['weight_decay'] != 0:
                        grad.add_(p.data, alpha=group['weight_decay'])

                # get current state variable
                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Update first and second moment running average
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(
                    grad_residual, grad_residual, value=1 - beta2
                )

                if amsgrad:
                    max_exp_avg_var = state['max_exp_avg_var']
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(
                        max_exp_avg_var,
                        exp_avg_var.add_(group['eps']),
                        out=max_exp_avg_var,
                    )

                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(
                        group['eps']
                    )
                else:
                    denom = (
                        exp_avg_var.add_(group['eps']).sqrt()
                        / math.sqrt(bias_correction2)
                    ).add_(group['eps'])

                # update
                if not self.rectify:
                    # Default update
                    step_size = group['lr'] / bias_correction1
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

                else:  # Rectified update, forked from RAdam
                    buffered = group['buffer'][int(state['step'] % 10)]
                    if state['step'] == buffered[0]:
                        N_sma, step_size = buffered[1], buffered[2]
                    else:
                        buffered[0] = state['step']
                        beta2_t = beta2 ** state['step']
                        N_sma_max = 2 / (1 - beta2) - 1
                        N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                        buffered[1] = N_sma

                        # more conservative since it's an approximated value
                        if N_sma >= 5:
                            step_size = math.sqrt(
                                (1 - beta2_t)
                                * (N_sma - 4)
                                / (N_sma_max - 4)
                                * (N_sma - 2)
                                / N_sma
                                * N_sma_max
                                / (N_sma_max - 2)
                            ) / (1 - beta1 ** state['step'])
                        elif self.degenerated_to_sgd:
                            step_size = 1.0 / (1 - beta1 ** state['step'])
                        else:
                            step_size = -1
                        buffered[2] = step_size

                    if N_sma >= 5:
                        denom = exp_avg_var.sqrt().add_(group['eps'])
                        p.data.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                    elif step_size > 0:
                        p.data.add_(exp_avg, alpha=-step_size * group['lr'])

                if half_precision:
                    p.data = p.data.half()
                    p.grad = p.grad.half()

        return loss
