"""[EXPERIMENTAL] Tools for data (pre)processing."""

import numpy as np
import scipy.sparse
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.utils import check_random_state


class NoisyQuantileTransformer(QuantileTransformer):
    """A variation of `sklearn.preprocessing.QuantileTransformer`.

    This transformer can be considered as one of the default preprocessing strategies
    for tabular data problems (in addition to more popular ones such as
    `sklearn.preprocessing.StandardScaler`).

    TODO: explain the difference with `sklearn.preprocessing.QuantileTransformer`

    As of now, no default parameter values are provided. However, a good starting
    point is the configuration used in some papers on tabular deep learning [1,2]:

    - `n_quantiles=min(train_size // 30, 1000)` where `train_size` is the number of
        objects passed to the `.fit()` method. This heuristic rule was tested on
        datasets with `train_size >= 5000`.
    - `output_distribution='normal'`
    - `subsample=10**9`
    - `noise_std=1e-3`

    References:
        * [1] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", NeurIPS 2021
        * [2] Yury Gorishniy, Ivan Rubachev, Artem Babenko, "On Embeddings for Numerical Features in Tabular Deep Learning", arXiv 2022
    """

    def __init__(
        self,
        *,
        n_quantiles: int,
        output_distribution: str,
        subsample: int,
        noise_std: float,
        **kwargs,
    ) -> None:
        """TODO"""
        if noise_std <= 0.0:
            raise ValueError(
                'noise_std must be positive. Note that with noise_std=0 the transformer'
                ' is equivalent to `sklearn.preprocessing.QuantileTransformer`'
            )
        super().__init__(
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            subsample=subsample,
            **kwargs,
        )
        self.noise_std = noise_std

    def fit(self, X, y=None):
        exception = ValueError('X must be either `numpy.ndarray` or `pandas.DataFrame`')
        if isinstance(X, np.ndarray):
            X_ = X
        elif hasattr(X, 'values'):
            try:
                import pandas

                if not isinstance(X, pandas.DataFrame):
                    raise exception
            except ImportError:
                raise exception
            X_ = X.values
        else:
            raise exception
        if scipy.sparse.issparse(X_):
            raise ValueError(
                'rtdl.data.NoisyQuantileTransformer does not support sparse input'
            )

        self.scaler_ = StandardScaler(with_mean=False)
        X_ = self.scaler_.fit_transform(X_)
        random_state = check_random_state(self.random_state)
        return super().fit(X_ + random_state.normal(0.0, self.noise_std, X_.shape), y)

    def transform(self, X):
        return super().transform(self.scaler_.transform(X))
