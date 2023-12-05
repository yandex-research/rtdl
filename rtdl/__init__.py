"""Research on tabular deep learning."""

__version__ = '0.0.14.dev7'

import warnings

warnings.warn(
    'The rtdl package is deprecated. See the GitHub repository to learn more.',
    DeprecationWarning,
)

from . import data  # noqa: F401
from .functional import geglu, reglu  # noqa: F401
from .modules import (  # noqa: F401
    GEGLU,
    MLP,
    CategoricalFeatureTokenizer,
    CLSToken,
    FeatureTokenizer,
    FTTransformer,
    MultiheadAttention,
    NumericalFeatureTokenizer,
    ReGLU,
    ResNet,
    Transformer,
)
