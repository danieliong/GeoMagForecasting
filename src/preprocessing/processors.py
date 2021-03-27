import pandas as pd
import numpy as np
import logging
import functools
import itertools

# from loguru import logger
from pandas.tseries.frequencies import to_offset
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from omegaconf import OmegaConf, open_dict
from hydra.utils import to_absolute_path

logger = logging.getLogger(__name__)


class Resampler(TransformerMixin):
    def __init__(
        self,
        freq="T",
        label="right",
        func="mean",
        time_col="times",
        verbose=True,
        **kwargs,
    ):
        """Scikit-learn Wrapper for Pandas resample method


        Parameters
        ----------
        freq : DateOffset, Timedelta or str, default="T"
            The offset string or object representing target conversion
        label : {'right','left'}, default="right"
            Which bin edge label to label bucket with. The default is ‘left’ for
            all frequency offsets except for ‘M’, ‘A’, ‘Q’, ‘BM’, ‘BA’, ‘BQ’,
            and ‘W’ which all have a default of ‘right’.
        func : function or str, default="max"
            Function to apply to time-aggregated data.
        kwargs : Keyword arguments for pd.DataFrame.resample

        """

        self.freq = freq
        self.label = label
        self.func = func
        self.time_col = time_col
        self.verbose = verbose
        self.kwargs = kwargs

    # TODO: Replace with the one in utils later.
    def _get_freq_multi_idx(self, X):
        def _get_freq_one_storm(x):
            # Infer freq for one storm
            times = x.index.get_level_values(self.time_col)
            return to_offset(pd.infer_freq(times))

        # Infer frequency within each storm and get unique frequencies
        freqs = X.groupby(level=0).apply(_get_freq_one_storm).unique()

        # If there is only one unique frequency
        if len(freqs) == 1:
            return freqs[0]
        else:
            return None

    def _get_freq(self, X):
        # HACK: Better way might be to resample before splitting but then the resampler
        # will not be part of the pipeline
        # TODO: Figure this out later.
        if isinstance(X.index, pd.MultiIndex):
            freq = self._get_freq_multi_idx(X)
        else:
            freq = to_offset(pd.infer_freq(X.index))

        return freq

    def fit(self, X, y=None):

        if self.func is None:
            self.func = "mean"

        self.freq = to_offset(self.freq)
        self.X_freq_ = self._get_freq(X)

        return self

    def transform(self, X, y=None):

        X_freq = self._get_freq(X)
        logger.debug("Frequency: %s", X_freq)
        assert X_freq == self.X_freq_, "X does not have the correct frequency."

        if X_freq is None or self.freq > X_freq:
            if isinstance(X.index, pd.MultiIndex):
                X = (
                    X.groupby(level="storm")
                    .resample(
                        self.freq, label=self.label, level=self.time_col, **self.kwargs
                    )
                    .apply(self.func)
                )
            else:
                X = X.resample(self.freq, label=self.label, **self.kwargs).apply(
                    self.func
                )
        else:
            if self.verbose:
                logger.debug(
                    f"Specified frequency ({self.freq}) is <= data frequency ({X_freq}). Resampling is ignored."
                )

        return X

    def inverse_transform(self, X):
        return X


class Interpolator(TransformerMixin):
    def __init__(
        self, method="linear", axis=0, limit_direction="both", limit=15, **kwargs
    ):
        """Scikit-learn wrapper for Pandas interpolate method
        """

        self.method = method
        self.axis = axis
        self.limit_direction = limit_direction
        self.limit = limit
        self.kwargs = kwargs

    def fit(self, X, y=None):
        # For compatibility only
        return self

    def _transform_multi_idx(self, X):

        if isinstance(X, pd.DataFrame):
            _interpolate = pd.DataFrame.interpolate
        elif isinstance(X, pd.Series):
            _interpolate = pd.Series.interpolate

        X = X.groupby(level="storm").apply(
            _interpolate,
            method=self.method,
            axis=self.axis,
            limit_direction=self.limit_direction,
            **self.kwargs,
        )

        return X

    def transform(self, X, y=None):

        if isinstance(X.index, pd.MultiIndex):
            X = self._transform_multi_idx(X)
        else:
            X = X.interpolate(
                method=self.method,
                axis=self.axis,
                limit_direction=self.limit_direction,
                **self.kwargs,
            )

        return X

    # NOTE: This is for inverse transforming the pipeline when computing metrics later.
    # The only thing that needs to be inversed is the scaler.
    # Find better way to do this?
    def inverse_transform(self, X):
        return X


class PandasTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer=None, **transformer_params):
        self.transformer = transformer
        self.transformer_params = transformer_params

    def fit(self, X, y=None, **fit_params):

        self.type_ = type(X)

        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns
        elif isinstance(X, pd.Series):
            self.name_ = X.name
        else:
            logger.warning("X is not a pandas object.")

        self.transformer.fit(X, **fit_params)

        return self

    def transform(self, X):
        check_is_fitted(self)

        assert isinstance(X, self.type_)

        if isinstance(X, pd.Series):
            assert X.name == self.name_
            X_ = self.transformer.transform(X.to_numpy().reshape(-1, 1))
            X_pd = pd.Series(X_.flatten(), name=self.name_, index=X.index)
        elif isinstance(X, pd.DataFrame):
            assert X.columns == self.columns_
            X_ = self.transformer.transform(X)
            X_pd = pd.DataFrame(X_, columns=self.columns_, index=X.index)

        return X_pd

    def inverse_transform(self, X):
        check_is_fitted(self)
        assert isinstance(X, self.type_)

        if isinstance(X, pd.Series):
            assert X.name == self.name_
            X_ = self.transformer.inverse_transform(X.to_numpy().reshape(-1, 1))
            X_pd = pd.Series(X_.flatten(), name=self.name_, index=X.index)
        elif isinstance(X, pd.DataFrame):
            assert X.columns == self.columns_
            X_ = self.transformer.inverse_transform(X)
            X_pd = pd.DataFrame(X_, columns=self.columns_, index=X.index)

        return X_pd


# INCOMPLETE
class SolarWindPropagator(BaseEstimator, TransformerMixin):
    def __init__(self, position_cols=["x_gse", "y_gse", "z_gse"], delete_cols=True):
        self.position_cols = position_cols
        self.delete_cols = delete_cols

    def fit(self, X, y=None):
        # TODO: Check if satellite positions are in columns
        return self

    def transform(self, X):
        # TODO

        if self.delete_cols:
            logger.debug("Dropping satellite position columns...")
            X = X.drop(columns=self.position_cols)

        return X


# TODO: Add this to pipeline
def limited_change(x, factor=1.3):
    """Apply limited relative change to density and temperature by a set factor"""

    out = x.copy()

    # treat 0 as nan for density and temperature
    out.loc[out.values == 0] = np.nan

    for i, (d1, d2) in enumerate(zip(out[:-1], out[1:])):
        if np.isnan(d2):
            out[i + 1] = d1  # assume persistence
        else:
            out[i + 1] = np.nanmax((np.nanmin((d2, d1 * factor)), d1 / factor))
    return out


# TODO: Add this to pipeline
def limited_change_speed(speed, density, change_down=30, change_ups=[50, 10]):
    """Apply limited relative change to speed according to density"""

    # QUESTION: Which velocity do I use?

    out = speed.copy()
    for i, (d1, d2) in enumerate(zip(out[:-1], out[1:])):
        if np.isnan(d2):
            out[i + 1] = d1  # assume persistence
        else:
            if density[i + 1] > density[i]:
                out[i + 1] = np.nanmax(
                    (np.nanmin((d2, d1 + change_ups[0])), d1 - change_down)
                )
            else:
                out[i + 1] = np.nanmax(
                    (np.nanmin((d2, d1 + change_ups[1])), d1 - change_down)
                )
    return out


def _get_callable(obj_str):
    # TODO: Modify to allow scaler_str to be more general
    # TODO: Validation

    obj = eval(obj_str)

    return obj


def _delete_df_cols(X, cols, errors="ignore", **kwargs):
    return X.drop(columns=cols, errors=errors)


class HydraPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, cfg):
        # TODO: Change this to take general arguments to make it more general.
        self.cfg = cfg
        self.pipeline_list = []

        # NOTE: Add new method for each processing step

    def _param_in_cfg(self, param):
        with open_dict(self.cfg):
            return not OmegaConf.is_none(self.cfg, param)

    def add_step(self, name, transformer):
        self.pipeline_list.append((name, transformer))

    # TODO: Modify pipeline steps below to return transformer
    def pipeline_step(param, return_transformer=True):
        # Decorator for pipeline step helper functions
        # param should be the param in config file
        def check_param_exists(func):
            # Ignores func and returns None if param is not in self.cfg
            @functools.wraps(func)
            def wrapped_func(inst):
                if inst._param_in_cfg(param):
                    with open_dict(inst.cfg):
                        logger.debug(f"Adding {param} to pipeline...")
                        # Get params from cfg and add it as first argument in func
                        params = OmegaConf.select(inst.cfg, param)
                        if return_transformer:
                            transformer = func(inst, params)
                            inst.add_step(param, transformer)
                        else:
                            return func(inst, params)
                else:
                    logger.debug(f"{param} not in configs")
                    return None

            return wrapped_func

        return check_param_exists

    @pipeline_step("delete_cols")
    def _add_delete_cols(self, params):

        cols_to_delete = list(itertools.chain(*params))
        delete_cols_transformer = FunctionTransformer(
            _delete_df_cols, kw_args={"cols": cols_to_delete}
        )

        if len(cols_to_delete) == 0:
            return None
        else:
            return delete_cols_transformer

    @pipeline_step("interpolate")
    def _add_interpolate(self, params):

        if isinstance(params, bool):
            if params:
                return Interpolator()
                # self.pipeline_list.append(("interpolator", Interpolator()))
        elif OmegaConf.is_config(params):
            return Interpolator(**params)
            # self.pipeline_list.append(("interpolator", Interpolator(**params)))

    @pipeline_step("resample")
    def _add_resample(self, params):

        if OmegaConf.is_config(params):
            return Resampler(**params)
            # self.pipeline_list.append(("resampler", Resampler(**params)))

    @pipeline_step("filter", return_transformer=False)
    def _add_filter(self, params):

        if params.type == "limited_change":
            logger.debug("Filter type: %s", params.type)
            # INCOMPLETE
            pass

    @pipeline_step("propagate")
    def _add_propagate(self, params):
        # INCOMPLETE

        kwargs = self.cfg.propagate if self.cfg.propagate is not None else {}
        return SolarWindPropagator(**kwargs)
        # self.pipeline_list.append(("propagator", SolarWindPropagator(**kwargs)))

    def _add_scaler_func(self):
        # NOTE: Can't use pipeline_step decorator on this

        if not self._param_in_cfg("scaler") and not self._param_in_cfg("func"):
            return None

        # NOTE: scaler ignored when func is specified
        func = self.cfg.func.func
        inverse_func = self.cfg.func.inverse_func
        scaler = self.cfg.scaler.scaler

        if func is not None:
            logger.debug(f"Scaling using function {func}...")

            func = _get_callable(func)
            inverse_func = _get_callable(inverse_func)
            func_transformer = FunctionTransformer(
                func=func, inverse_func=inverse_func, **self.cfg.func.kwargs
            )
            transformer = PandasTransformer(transformer=func_transformer)
            self.add_step("func", transformer)

        elif scaler is not None:
            logger.debug(f"Scaling using scaler {scaler}...")

            scaler_callable = _get_callable(scaler)
            transformer = PandasTransformer(
                transformer=scaler_callable(), **self.cfg.scaler.kwargs
            )
            self.add_step("scaler", transformer)
        else:
            logger.debug("Scaler or func was not specified.")

    def create_pipeline(self):
        logger.debug("Creating pipeline...")

        # NOTE: Choose order here.
        # params is added as argument in pipeline_steps decorator
        self._add_resample()
        self._add_filter()
        self._add_interpolate()
        self._add_propagate()
        self._add_delete_cols()
        self._add_scaler_func()

        if self.pipeline_list is not None:
            pipeline = Pipeline(self.pipeline_list)
            return pipeline
        else:
            logger.debug("Pipeline is empty.")
            return None

    def fit(self, X, y=None):
        self.pipeline_ = self.create_pipeline()

        if self.pipeline_ is not None:
            self.pipeline_.fit(X, y)

        return self

    def transform(self, X):
        if self.pipeline_ is None:
            return X
        else:
            return self.pipeline_.transform(X)
