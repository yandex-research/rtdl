import itertools
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import torch
import torch.nn as nn
import torch.optim

from .._utils import INTERNAL_ERROR_MESSAGE
from ..optim import get_parameter_groups
from ._backbones import MLP, ResNet, Transformer
from ._embeddings import CatEmbeddings, LinearEmbeddings


class _Lambda(nn.Module):
    def __init__(self, fn: Callable):
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


MainModule = TypeVar('MainModule', bound=nn.Module)


class SimpleModel(nn.Module, Generic[MainModule]):
    """

    Warning:
        Do not instantiate this class directly, use `make_simple_model` instead.
    """

    def __init__(
        self,
        input: Dict[str, Union[Tuple, List[Tuple]]],
        main: MainModule,
        main_input_ndim: int,
        output: Optional[Dict[str, nn.Module]] = None,
    ) -> None:
        # let's check this important condition again
        assert main_input_ndim in (2, 3), INTERNAL_ERROR_MESSAGE

        input_modules = {}
        input_args: Dict[str, Union[Tuple[str, ...], List[Tuple[str, ...]]]] = {}
        for name, spec in input.items():
            if isinstance(spec, list):
                input_modules[name] = nn.ModuleList()
                input_args[name] = []
                for module, *args in spec:
                    input_modules[name].append(module)
                    assert isinstance(input_args[name], list)
                    cast(list, input_args[name]).append(tuple(args))
            else:
                input_modules[name] = spec[0]
                input_args[name] = spec[1:]

        self.input = nn.ModuleDict(input_modules)
        self._input_args = input_args
        self.main = main
        self._main_input_ndim = main_input_ndim
        self.output = None if output is None else nn.ModuleDict(output)

    def _get_forward_kwarg_names(self) -> Set[str]:
        kwargs: Set[str] = set()
        for args in self._input_args.values():
            if isinstance(args, tuple):
                args = [args]
            kwargs.update(itertools.chain.from_iterable(args))
        return kwargs

    def usage(self) -> str:
        return f'forward(*, {", ".join(self._get_forward_kwarg_names())})'

    def forward(self, **kwargs) -> Any:
        required_kwarg_names = self._get_forward_kwarg_names()
        if required_kwarg_names != set(kwargs):
            raise TypeError(
                f'The expected arguments are: {required_kwarg_names}.'
                f' The provided arguments are: {set(kwargs)}.'
            )

        input_results = []
        for name in self.input:
            module = self.input[name]
            input_args = self._input_args[name]

            if isinstance(module, nn.ModuleList):
                assert isinstance(input_args, list), INTERNAL_ERROR_MESSAGE
                outputs = []
                for i_mod, (mod, args) in enumerate(zip(module, input_args)):
                    out = mod(**{arg: kwargs[arg] for arg in args})
                    if out.ndim != 3:
                        raise RuntimeError(
                            f'The output of the input module {name}[{i_mod}] has'
                            f' {out.ndim} dimensions, but when there are multiple input modules'
                            ' under the same name, they must output three-dimensional tensors'
                        )
                    outputs.append(out)
                    if outputs[-1].shape[:2] != outputs[0].shape[:2]:
                        raise RuntimeError(
                            f'The input modules {name}[{0}] and {name}[{i_mod}] produced'
                            ' tensors with different two dimensions:'
                            f' {outputs[-1].shape} VS  {outputs[0].shape}'
                        )
                input_results.append(torch.cat(outputs, 2))
            else:
                assert isinstance(input_args, tuple), INTERNAL_ERROR_MESSAGE
                output = module(**{arg: kwargs[arg] for arg in input_args})
                if output.ndim < 2 or output.ndim > 3:
                    raise RuntimeError(
                        f'The input module {name} produced tensor with {output.ndim}'
                        ' dimensions, but it must be 2 or 3.'
                    )
                input_results.append(output)

        # let's check this important condition for the last time
        assert self._main_input_ndim in (2, 3), INTERNAL_ERROR_MESSAGE
        # for main_input_ndim = 2, the concatenation happens along the last dimension
        # for main_input_ndim = 3, the concatenation happens along the feature dimension
        # in both cases, dim=1
        x = torch.cat(
            [t.flatten(self._main_input_ndim - 1, -1) for t in input_results], dim=1
        )
        x = self.main(x)
        return x if self.output is None else {k: v(x) for k, v in self.output.items()}


# here, "Tuple" means Tuple[Union[Callable, nn.Module], str, ...]
_SimpleInputModule = Union[Callable, nn.Module, Tuple]


def make_simple_model(
    input: Dict[str, Union[_SimpleInputModule, List[_SimpleInputModule]]],
    main: MainModule,
    *,
    main_input_ndim: Optional[int] = None,
    output: Optional[Dict[str, nn.Module]] = None,
) -> SimpleModel[MainModule]:
    """Make a simple model (N input modules + 1 main module + M output modules).

    Examples:
        .. testcode::

            # data info
            batch_size = 3
            n_num_features = 4
            n_cat_features = 2
            cat_cardinalities = [3, 4]
            d_num_embedding = 5

            # inputs
            x_num = torch.randn(batch_size, n_num_features)
            x_cat = torch.tensor([[0, 1], [2, 3], [2, 0]])
            assert len(x_cat) = batch_size

            # modules
            m_num = rtdl.nn.make_plr_embeddings(d_num_embedding, n_num_features, 0.1)
            m_cat = rtdl.nn.OneHotEncoder(cat_cardinalities)
            m_main = rtdl.nn.MLP.make_baseline(
                d_in=n_num_features * d_num_embedding + sum(cat_cardinalities),
                d_out=1,
                n_blocks=1,
                d_layer=2,
                dropout=0.1,
            )

            model = make_simple_model(
                dict(
                    hello=m_num,
                    world=m_cat,
                ),
                m_main,
            )
            y_pred = model(hello=x_num, world=x_cat)

    The line `hello=m_num,` in the dictionary above is equivalent to
    `hello=(m_num, 'hello')`. The general pattern for the input modules is as follows::

        model = make_simple_model(
            dict(
                a=module_0,  # -> a=(module_0, 'a'),
                b=(module_1, 'arg_1'),
                c=(module_2, 'arg_2', 'arg_3'),
                d=[module_3, module_4]  # -> (batch_size, x4.shape[1], d3 + d4)
            ),
            m_main
        )
        assert model.input['a'] is module_0
        assert model.input['b'] is module_1
        assert model.input['c'] is module_2
        assert isinstance()
        y_pred = model(a=x0, arg_1=x1, arg_2=x2, arg_3=x3, d=x4)

    """
    if main_input_ndim is None:
        if isinstance(main, (MLP, ResNet)):
            main_input_ndim = 2
        elif isinstance(main, Transformer):
            main_input_ndim = 3
        else:
            raise ValueError(
                'If the main module is none of rtdl.nn.{MLP,ResNet,Transformer},'
                ' than main_input_ndim must be an integer'
            )
    if main_input_ndim not in (2, 3):
        raise ValueError('main_input_ndim must be either 2 or 3')

    def to_tuple(name, spec) -> Tuple:
        if callable(spec) and not isinstance(spec, nn.Module):
            spec = _Lambda(spec)
        if isinstance(spec, nn.Module):
            spec = (spec, name)
        return spec

    normalized_input = {
        name: (
            [to_tuple(name, s) for s in spec]
            if isinstance(spec, list)
            else to_tuple(name, spec)
        )
        for name, spec in input.items()
    }
    for spec in normalized_input.values():
        for spec_tuple in spec if isinstance(spec, list) else [spec]:
            if (
                len(spec_tuple) < 2
                or not isinstance(spec_tuple[0], nn.Module)
                or any(not isinstance(x, str) for x in spec_tuple[1:])
            ):
                raise ValueError('The argument `input` is invalid')

    if output is None:
        normalized_output = None
    else:
        normalized_output = {}
        for name, module_fn in output.items():
            normalized_output[name] = (
                module_fn if isinstance(module_fn, nn.Module) else _Lambda(module_fn)
            )

    return SimpleModel(
        normalized_input,  # type: ignore
        main,
        main_input_ndim,
        normalized_output,
    )


_FT_TRANSFORMER_ACTIVATION = 'ReGLU'


def make_ft_transformer(
    n_num_features: int,
    cat_cardinalities: List[int],
    d_out: Optional[int],
    d_embedding: int,
    n_blocks: int,
    attention_dropout: float,
    ffn_d_hidden: int,
    ffn_dropout: float,
    residual_dropout: float,
    last_block_pooling_token_only: bool = True,
    linformer_compression_ratio: Optional[float] = None,
    linformer_sharing_policy: Optional[str] = None,
) -> SimpleModel:
    if not n_num_features and not cat_cardinalities:
        raise ValueError(
            f'At least one kind of features must be presented. The provided arguments are:'
            f' n_num_features={n_num_features}, cat_cardinalities={cat_cardinalities}.'
        )

    input_modules: Dict[str, nn.Module] = {}
    if n_num_features:
        input_modules['x_num'] = LinearEmbeddings(n_num_features, d_embedding)
    if cat_cardinalities:
        input_modules['x_cat'] = CatEmbeddings(
            cat_cardinalities, d_embedding, stack=True, bias=True
        )
    m_main = Transformer.make_baseline(
        d_embedding=d_embedding,
        d_out=d_out,
        n_blocks=n_blocks,
        attention_n_heads=8,
        attention_dropout=attention_dropout,
        ffn_d_hidden=ffn_d_hidden,
        ffn_dropout=ffn_dropout,
        activation=_FT_TRANSFORMER_ACTIVATION,
        residual_dropout=residual_dropout,
        pooling='cls',
        last_block_pooling_token_only=last_block_pooling_token_only,
        linformer_compression_ratio=linformer_compression_ratio,
        linformer_sharing_policy=linformer_sharing_policy,
        n_tokens=(
            None
            if linformer_compression_ratio is None
            else 1 + n_num_features + len(cat_cardinalities)  # +1 because of CLS
        ),
    )
    return make_simple_model(
        input_modules,  # type: ignore
        m_main,
    )


def make_ft_transformer_default(
    *,
    n_num_features: int,
    cat_cardinalities: List[int],
    d_out: Optional[int],
    n_blocks: int = 3,
    last_block_pooling_token_only: bool = True,
    linformer_compression_ratio: Optional[float] = None,
    linformer_sharing_policy: Optional[str] = None,
) -> Tuple[SimpleModel, torch.optim.Optimizer]:
    if not (1 <= n_blocks <= 6):
        raise ValueError('n_blocks must be in the range from 1 to 6 (inclusive)')

    from ._backbones import _is_reglu

    default_value_index = n_blocks - 1
    d_embedding = [96, 128, 192, 256, 320, 384][default_value_index]
    ffn_d_hidden_factor = (4 / 3) if _is_reglu(_FT_TRANSFORMER_ACTIVATION) else 2.0
    model = make_ft_transformer(
        n_num_features=n_num_features,
        cat_cardinalities=cat_cardinalities,
        d_out=d_out,
        d_embedding=d_embedding,
        n_blocks=n_blocks,
        attention_dropout=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35][default_value_index],
        ffn_dropout=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25][default_value_index],
        ffn_d_hidden=int(d_embedding * ffn_d_hidden_factor),
        residual_dropout=0.0,
        last_block_pooling_token_only=last_block_pooling_token_only,
        linformer_compression_ratio=linformer_compression_ratio,
        linformer_sharing_policy=linformer_sharing_policy,
    )
    optimizer = torch.optim.AdamW(get_parameter_groups(model), 1e-4, weight_decay=1e-5)
    return model, optimizer
