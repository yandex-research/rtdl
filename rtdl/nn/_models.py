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
        super().__init__()
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

        super().__init__()
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
    """Make a simple model (N input modules + 1 main module [+ M output modules]).

    See the tutorial below.

    Args:
        input: the input modules. If the main module is
            `rtdl.nn.Transformer` or ``main_input_ndim == 3``, then the input modules must
            produce three dimensional tensors. Otherwise, the input modules are allowed
            to produce two and three dimensional tensors.
        main: the main module. See the tutorial below.
        main_input_ndim: the number of dimensions of the main module's input. The outputs
            of all input modules are merged into a tensor with ``main_input_ndim`` dimensions.
            If the main module is one of {`rtdl.nn.MLP`, `rtdl.nn.ResNet`, `rtdl.nn.Transformer`},
            then ``main_input_ndim`` must be `None` and it will be set to the correct value
            under the hood (2, 2 and 3 respectively). Otherwise, it must be
            either ``2`` or ``3``. For ``3`` (i.e. for transformer-like main modules),
            the merge (concatenation) happens along the dimension ``1``.
        output: the output modules. See the tutorial below.

    .. rubric:: Tutorial

    The basic usage is demonstrated in the following example:

    .. testcode::

        # data info
        batch_size = 3
        n_num_features = 4  # numerical features
        n_cat_features = 2  # categorical features
        # first categorical feature takes 3 unique values
        # second categorical feature takes 4 unique values
        cat_cardinalities = [3, 4]

        # inputs
        x_num = torch.randn(batch_size, n_num_features)
        x_cat = torch.tensor([[0, 1], [2, 3], [2, 0]])
        asssert x_cat.shape == (batch_size, n_cat_features)

        # (1) the module for numerical features
        # no transformations for numerical features:
        m_num = nn.Identity()
        # for a fancy model with embeddings for numerical features, it would be:
        # m_num = make_fancy_num_embedding_module(...)

        # (2) the module for categorical features
        m_cat = rtdl.nn.OneHotEncoder(cat_cardinalities)

        # (3) the main module (backbone)
        m_main = rtdl.nn.MLP.make_baseline(
            d_in=n_num_features * d_num_embedding + sum(cat_cardinalities),
            d_out=1,
            n_blocks=1,
            d_layer=2,
            dropout=0.1,
        )

        model = make_simple_model(
            dict(  # any number of input modules
                hello=m_num,  # m_num is accessible as model.input['hello']
                world=m_cat,  # m_cat is accessible as model.input['world']
            ),
            # the outputs of the input modules are merged into one tensor and passed
            # to the main module
            m_main,  # m_main is accessible as model.main
        )

        # x_num is the input for the module 'hello' (m_num)
        # x_cat is the input for the module 'world' (m_cat)
        y_pred = model(hello=x_num, world=x_cat)

    Of course, in practice, you would use better names for the input modules, for example::

        model = make_simple_model(dict(x_num=m_num, x_cat=m_cat), m_main)
        ...
        y_pred = model(x_num=x_num, x_cat=x_cat)

    Optionally, you can set output modules. For example, if you want the model to have
    two heads (one for the downstream task and another one for the auxiliary input
    reconstruction task serving as a regularization), then the code can look like this::

        model = make_simple_model(
            dict(x_num=m_num, x_cat=m_cat),
            m_main,
            output=dict(y_pred=m_pred, x_rec=m_reconstruction)
        )
        # model(x_num=..., x_cat=...) produces a dictionary:
        # {
        #     'y_pred': m_pred(<output of the main module>),
        #     'x_rec': m_reconstruction(<output of the main module>),
        # }

    .. rubric:: Advanced usage

    Let's consider this model::

        model = make_simple_model(dict(hello=m_hello, world=m_world), m_main)
        # usage: model(hello=..., world=...)

    In fact, the snippet above is equivalent to the following::

        model = make_simple_model(
            dict(
                # to the left of "=", the module name is given (hello)
                # to the right of "=", the module (m_hello) and its inputs are given ('hello')
                hello=(m_hello, 'hello'),
                world=(m_world, 'world'),
            )
        )
        # usage: model(hello=..., world=...)

    Let's change the last snippet to the following::

        model = make_simple_model(
            dict(
                hello=(m_hello, 'a'),
                world=(m_world, 'b'),
            )
        )
        # usage: model(a=..., b=...)

    So, the line ``hello=(m_hello, 'a'),`` means that:

        * the module m_hello can be accessed as ``model.input['hello']``
        * this module requires the input ``a``

    The general pattern is as follows::

        model = make_simple_model(
            dict(
                a=ma,  # this is transformed to: a=(m0, 'a'),
                b=(mb, 'arg_1'),
                c=(mc, 'arg_2', 'arg_3'),  # the module will be called as mc(arg_2, arg_3)
                # one of the inputs for the module 'd' is the same as for the module 'b',
                # this is allowed!
                d=(md, 'arg_1', 'arg_4')
            )
        )
        # usage: model(a=, arg_1=..., arg_2=..., arg_3=..., arg_4=...)

    The last advanced technique is demonstrated in the following example::

        m_num_plr = rtdl.nn.make_plr_embeddings(...)
        m_num_ple_lr = rtdl.nn.make_ple_lr_embeddings(...)
        m_cat = nn.OneHotEncoder(...)
        m_main = rtdl.nn.Transformer.make_baseline(...)
        model = make_simple_model(
            dict(
                x_num=[m_num_plr, m_num_ple_lr],
                x_cat=x_cat,
            ),
            m_main,
        )
        y_pred = model(x_num=..., x_cat=...)

    To understand it, let's simplify the notation:

        model = make_simple_model(
            dict(
                x_num=[m0, m1],  # m0/m1 are accessible as model.input['x_num'][0/1]
                x_cat=m2
            )
        )
        # usage: model(x_num=..., x_cat=...)

    Now, ``model.input['x_num']`` is not a module, but a group of modules. E.g. ``m0``
    can be accessed as ``model.input['x_num'][0]``. Each module in a group can be
    a module or a tuple as in the examples above.

    Each module in a module group must produce three dimensional tensor as its output.
    The outputs of all modules in a group are contatenated along the last dimension.
    So, if ``m0(x_num)`` has the shape ``(batch_size, n_features, d0)`` and
    ``m1(x_num)`` has the shape ``(batch_size, n_features, d1)``, then the output of
    the group ``'x_num'`` has the shape ``(batch_size, n_features, d0 + d1)``. This can
    be useful to combine different kinds of embeddings and pass them to backbones that
    require three dimensional inputs (e.g. to Transformer).
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


def make_default_ft_transformer(
    *,
    n_num_features: int,
    cat_cardinalities: List[int],
    d_out: Optional[int],
    n_blocks: int = 3,
    linformer_compression_ratio: Optional[float] = None,
    linformer_sharing_policy: Optional[str] = None,
) -> Tuple[SimpleModel, torch.optim.Optimizer]:
    """Create the default FT-Transformer and the optimizer.

    The function creates:

        * FT-Transformer with the default hyperparameters
        * the default optimizer for this model

    as described in [1].

    This function is useful if you want to quickly try a fancy model without investing
    in hyperparameter tuning. For a zero-config solution, the average performance
    can be considered as decent (especially if you have an ensemble of default
    FT-Transformers). That said, for achieving the best results on a given task, more
    customized solutions should be used.

    Args:
        n_num_features: the number of numerical features
        cat_cardinalities: the cardinalities of categorical features
            (``cat_cardinalities[i]`` is the number of unique values of the i-th
            categorical feature)
        d_out: the output size. If `None`, then the model is backbone-only.
        n_blocks: the number of blocks. Other hyperparameters are determined based
            ``n_blocks``.
        linformer_compression_ratio: the option for the fast linear attention.
            See `rtdl.nn.MultiheadAttention` for details.
        linformer_sharing_policy: the option for the fast linear attention.
            See `rtdl.nn.MultiheadAttention` for details.

    References:
        * [1] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021

    Examples:
        .. testcode::

            model, optimizer = make_ft_transformer_default()
    """
    if not n_num_features and not cat_cardinalities:
        raise ValueError(
            f'At least one kind of features must be presented. The provided arguments are:'
            f' n_num_features={n_num_features}, cat_cardinalities={cat_cardinalities}.'
        )
    if not (1 <= n_blocks <= 6):
        raise ValueError('n_blocks must be in the range from 1 to 6 (inclusive)')

    from ._backbones import _is_reglu

    default_value_index = n_blocks - 1
    d_embedding = [96, 128, 192, 256, 320, 384][default_value_index]
    ffn_d_hidden_factor = (4 / 3) if _is_reglu(_FT_TRANSFORMER_ACTIVATION) else 2.0

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
        attention_dropout=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35][default_value_index],
        ffn_d_hidden=int(d_embedding * ffn_d_hidden_factor),
        ffn_dropout=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25][default_value_index],
        ffn_activation=_FT_TRANSFORMER_ACTIVATION,
        residual_dropout=0.0,
        pooling='cls',
        linformer_compression_ratio=linformer_compression_ratio,
        linformer_sharing_policy=linformer_sharing_policy,
        n_tokens=(
            None
            if linformer_compression_ratio is None
            else 1 + n_num_features + len(cat_cardinalities)  # +1 because of CLS
        ),
    )
    model = make_simple_model(
        input_modules,  # type: ignore
        m_main,
    )

    optimizer = torch.optim.AdamW(get_parameter_groups(model), 1e-4, weight_decay=1e-5)
    return model, optimizer
