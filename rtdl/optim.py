"""Optimization utilities."""

from typing import Any, Dict, List

import torch.nn as nn
from torch.nn.parameter import Parameter

from ._utils import INTERNAL_ERROR_MESSAGE
from .nn import CatEmbeddings, CLSEmbedding, LinearEmbeddings, PeriodicEmbeddings


def default_no_weight_decay_condition(
    module_name: str,
    module: nn.Module,
    parameter_name: str,
    parameter: Parameter,
) -> bool:
    """The default condition to decide whether a parameter needs weight decay.

    This function is used as the default condition in `get_parameter_groups`. Generally,
    the function returns ``True`` for normalization layer parameters, for biases in
    linear layers and for embedding layers.
    """
    del module_name, parameter
    return (
        isinstance(
            module,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.GroupNorm,
                nn.SyncBatchNorm,
                nn.InstanceNorm1d,
                nn.InstanceNorm2d,
                nn.InstanceNorm3d,
                nn.LayerNorm,
            ),
        )
        or (isinstance(module, nn.Linear) and parameter_name == 'bias')
        or isinstance(
            module, (CatEmbeddings, CLSEmbedding, LinearEmbeddings, PeriodicEmbeddings)
        )
    )


def get_parameter_groups(
    module: nn.Module, no_weight_decay_condition=default_no_weight_decay_condition
) -> List[Dict[str, Any]]:
    """Prepare parameter groups for an optimizer (instead of ``model.parameters()``).

    TL;DR::

        # before
        optimizer = SomeOptimizer(model.parameters(), ...)
        # after
        optimizer = SomeOptimizer(get_parameter_groups(model)

    The function returns two parameter groups, one of them has ``weight_decay`` set to
    ``0.0`` (i.e. the ``weight_decay`` parameter passed to the optimizer will NOT affect
    the parameters from this group).

    Args:
        module: the module
        no_weight_decay_condition: if this function returns ``True`` for a given
            parameter, then the corresponding parameter will be assigned to the
            group with ``weight_decay=0.0``. The signature must be the same as that
            of `default_no_weight_decay_condition`. Note that the function is called
            multiple times for the same parameter, since one parameter is a parameter of
            all its parent modules (see the example below). If it retuns ``True`` at
            least once, then the corresponding parameter will be assigned to the
            group with ``weight_decay=0.0``.
    Returns:
        parameter groups

    Examples:

        In this example, weight decay is set to zero only to biases in linear layers.
        It also demonstrates why the condition is called multiple times for the same
        parameter.

        .. testcode::

            def no_wd_condition(m_name, m, p_name, p):
                print(m_name or '-', p_name)
                return isinstance(m, nn.Linear) and p_name == 'bias'

            a = nn.ModuleList([nn.ModuleDict({'b': nn.Linear(1, 1)})])
            optimizer = optim.SGD(get_parameter_groups(a), 1e-2)

            for group in get_parameter_groups(a):
                for param in group['params']:
                    if param is a[0]['b'].weight:
                        assert 'weight_decay' not in group
                    elif param is a[0]['b'].bias:
                        assert group['weight_decay'] == 0.0

        .. testoutput::

            - 0.b.weight
            - 0.b.bias
            0 b.weight
            0 b.bias
            0.b weight
            0.b bias
    """
    wd_index = 0
    no_wd_index = 1
    parameter_groups: List[Dict[str, Any]] = [
        {'params': []},
        {'params': [], 'weight_decay': 0.0},
    ]
    assert (
        parameter_groups[no_wd_index].get('weight_decay') == 0.0
    ), INTERNAL_ERROR_MESSAGE

    no_wd: Dict[str, bool] = {}
    for m_name, m in module.named_modules():
        for p_name, p in m.named_parameters():
            full_p_name = f'{m_name}.{p_name}' if m_name else p_name
            if no_wd.get(full_p_name):
                continue
            no_wd[full_p_name] = no_weight_decay_condition(m_name, m, p_name, p)
    for full_p_name, p in module.named_parameters():
        group_index = no_wd_index if no_wd[full_p_name] else wd_index
        parameter_groups[group_index]['params'].append(p)
    return parameter_groups
