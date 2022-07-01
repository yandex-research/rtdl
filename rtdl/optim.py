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
