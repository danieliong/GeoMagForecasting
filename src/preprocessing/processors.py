import functools
import itertools
import logging

import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, open_dict

# from loguru import logger
from pandas.tseries.frequencies import to_offset
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted

from src.utils import get_freq
from src.storm_utils import (
    StormIndexAccessor,
    StormAccessor,
    iterate_storms_method,
    has_storm_index,
)

logger = logging.getLogger(__name__)

# For SolarWindPropagator
KM_PER_RE = 6371
X0_RE = 10


def _get_callable(obj_str):
    # TODO: Modify to allow scaler_str to be more general
    # TODO: Validation

    obj = eval(obj_str)

    return obj


def _delete_df_cols(X, cols, errors="ignore", **kwargs):
    # Flatten list
    cols = list(itertools.chain(*cols))
    logger.debug(f"Deleting columns: {', '.join(cols)} ")

    return X.drop(columns=cols, errors=errors)


class Resampler(BaseEstimator, TransformerMixin):
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

    def fit(self, X, y=None):

        if self.freq is None:
            self.freq = get_freq(X)
        else:
            self.freq = to_offset(self.freq)
        # self.X_freq_ = self._get_freq(X)

        return self

    @iterate_storms_method(drop_storms=True)
    def transform(self, X, y=None):
        # TODO: Use iterate_storms_method here

        X_freq = get_freq(X)

        # # Checking frequency is probably unnecessary
        # logger.debug("Inferred Frequency: %s", X_freq)
        # assert X_freq == self.X_freq_, "X does not have the correct frequency."

        if X_freq is None or self.freq > X_freq:
            X = X.resample(self.freq, label=self.label, **self.kwargs).apply(self.func)
        else:
            if self.verbose:
                logger.debug(
                    f"Specified frequency ({self.freq}) is <= data frequency ({X_freq}). Resampling is ignored."
                )

        return X

    def inverse_transform(self, X):
        return X


class Interpolator(BaseEstimator, TransformerMixin):
    def __init__(
        self, method="linear", axis=0, limit_direction="both", limit=15, **kwargs,
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

    @iterate_storms_method(drop_storms=True)
    def _transform(self, X):
        return X.interpolate(
            method=self.method,
            axis=self.axis,
            limit_direction=self.limit_direction,
            **self.kwargs,
        )

    def _transform_time(self, X):
        X_ = self._transform(X.reset_index(level="storm"))
        X_.set_index(["storm", X_.index], inplace=True)
        return X_

    def transform(self, X, y=None):

        if self.method == "time":
            return self._transform_time(X).squeeze()
        else:
            return self._transform(X).squeeze()

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


class ValuesFilter(BaseEstimator, TransformerMixin):
    def __init__(
        self, bad_values=[-9999, -999], value_ranges=None, copy=True, **replace_kwargs
    ):
        self.bad_values = bad_values
        self.value_ranges = value_ranges
        self.copy = copy
        self.replace_kwargs = replace_kwargs

    def fit(self, X, y=None):
        return self

    def _filter_ranges(self, X):

        for col in self.value_ranges.keys():
            if col in X.columns:
                value_min, value_max = self.value_ranges[col]
                logger.debug(
                    f"Converting values in {col} that are out of ({value_min}, {value_max}) to NA..."
                )
                cond = np.logical_and(X[col] >= value_min, X[col] <= value_max)
                X[col].where(cond=cond, other=np.nan, inplace=True)

    def _filter_bad_values(self, X):
        logger.debug(f"Converting values that are in {self.bad_values} to NA...")
        X.replace(
            to_replace=self.bad_values,
            value=np.nan,
            inplace=True,
            **self.replace_kwargs,
        )

    @iterate_storms_method(drop_storms=True)
    def transform(self, X):

        if self.bad_values is None and self.value_ranges is None:
            # Do nothing
            return X

        if self.copy:
            X_ = X.copy()
        else:
            X_ = X

        if self.value_ranges is not None:
            self._filter_ranges(X_)

        if self.bad_values is not None:
            self._filter_bad_values(X_)

        return X_

    def inverse_transform(self, X):
        return X


# REVIEW


class SolarWindPropagator(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        x_coord_col="x",
        speed_col="speed",
        time_level="times",
        freq=None,
        delete_cols=True,
        force=False,
    ):
        self.x_coord_col = x_coord_col
        self.speed_col = speed_col
        self.time_level = time_level
        self.freq = freq
        self.delete_cols = delete_cols
        self.force = force
        # TODO: Load positions from position_path if it is not None

    def fit(self, X, y=None):

        if self.force:
            assert self.x_coord_col in X.columns
            assert self.speed_col in X.columns

            if self.freq is None:
                self.freq = get_freq(X)

        return self

    @staticmethod
    def _make_time_nondecreasing(times):

        if not times.is_monotonic_increasing:
            current_min = pd.Timestamp.max
            for i, time in enumerate(times[::-1]):
                if time >= current_min:
                    times[~i] = np.nan
                if not pd.isna(time):
                    current_min = min(current_min, time)

        return times

    @classmethod
    @iterate_storms_method(storm_params=["times"], drop_storms=True)
    def compute_propagated_times(cls, x_coord, speed, times, freq):

        assert isinstance(times, pd.DatetimeIndex)

        delta_x = (x_coord - X0_RE) * KM_PER_RE
        time_delay = pd.to_timedelta(delta_x / speed, unit="sec")
        propagated_time = cls._make_time_nondecreasing(times + time_delay)

        return propagated_time.round(freq)

    def transform(self, X):

        required_cols = [self.x_coord_col, self.speed_col]
        if any(col not in X.columns for col in required_cols):
            if self.force:
                raise ValueError(
                    f"X doesn't contain the columns {', '.join(required_cols)}."
                )
            else:
                return X

        X_ = X.copy()
        x_coord = X_[self.x_coord_col]
        speed = X_[self.speed_col]
        # times = X_.index.get_level_values(self.time_level)

        # NOTE: Couldn't use iterate_storms_method here because we want to
        # interpolate x_coord as a whole
        if x_coord.isna().any():
            x_coord = Interpolator(method="time").transform(x_coord)

        # QUESTION: Should I interpolate, drop, or fill NAs in speed?
        if speed.isna().any():
            speed = Interpolator().transform(x_coord)

        # Set propagated time as new time index
        X_[self.time_level] = self.compute_propagated_times(
            x_coord, speed, times=X_.index, freq=self.freq
        )
        # iterate_storms_method will convert X_.index to time index if it is a
        # MultiIndex

        na_mask = ~X_[self.time_level].isna()
        X_.where(na_mask, np.nan, inplace=True)

        if has_storm_index(X_):
            X_.set_index([X_.storms.index, self.time_level], inplace=True)
            X_ = X_.storms.resample(self.freq, level=self.time_level).mean()
        else:
            X_.set_index(self.time_level, inplace=True)
            # X_.dropna(subset=["propagated_time"], inplace=True)
            X_ = X_.resample(self.freq).mean()

        if self.delete_cols:
            # logger.debug("Dropping x position column...")
            X_.drop(columns=[self.x_coord_col], inplace=True)

        return X_


# INCOMPLETE
class LimitedChangeFilter(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        change_down=30,
        change_ups=[50, 10],
        factor=1.3,
        speed_col="speed",
        density_col="speed",
        temp_col="temperature",
        copy=True,
    ):
        self.change_down = change_down
        self.change_ups = change_ups
        self.factor = factor
        self.speed_col = speed_col
        self.density_col = density_col
        self.temp_col = temp_col
        self.copy = copy

    def fit(self, X, y=None):
        # For compatibility only
        return self

    def _limit_change_speed(self, d1, d2, density_increasing):
        if density_increasing:
            return max(min(d2, d1 + self.change_up[0]), d1 - self.change_down)
        else:
            return max(min(d2, d1 + self.change_up[1]), d1 - self.change_down)

    def _limit_change_other(self, d1, d2):
        return max(min(d2, d1 * self.factor), (d1 / self.factor))

    def limit_change_speed(self, speed, density):
        for i, (d1, d2) in enumerate(zip(speed[:-1], speed[1:])):
            if np.isnan(d2):
                speed[i + 1] = d1
            else:
                density_increasing = density[i + 1] > density[i]
                speed = self.limit_change_speed(d1, d2, density_increasing)

        return speed

    def limit_change_other(self, x):
        # Should be satisfied if used ValuesFilter
        assert all(x > 0)

        for i, (d1, d2) in enumerate(zip(x[:-1], x[1:])):
            if np.isnan(d2):
                x[i + 1] = d1
            else:
                x[i + 1] = self._limit_change_other(d1, d2)

        return x

    @iterate_storms_method()
    def transform(self, X):

        if self.copy:
            X_ = X.copy()
        else:
            X_ = X

        X_[self.speed_col] = self.limit_change_speed(
            speed=X_[self.speed_col], density=X_[self.density_col]
        )

        X_[self.density_col] = self.limit_change_other(X_[self.density_col])
        X_[self.temp_col] = self.limit_change_other(X_[self.temp_col])

        return X_

    def inverse_transform(self, X):
        return X


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
        # params should be a list of columns to delete

        if len(params) == 0:
            return None

        delete_cols_transformer = FunctionTransformer(
            _delete_df_cols, kw_args={"cols": params}
        )
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

    @pipeline_step("values_filter")
    def _add_values_filter(self, params):
        if OmegaConf.is_config(params):
            return ValuesFilter(**params)

    @pipeline_step("filter", return_transformer=False)
    def _add_filter(self, params):
        assert OmegaConf.is_dict(params) or isinstance(params, dict)

        if OmegaConf.is_dict(params):
            params = OmegaConf.to_container(params)

        filter_type = params.pop("type")
        logger.debug("Filter type: %s", filter_type)

        return LimitedChangeFilter(**params[filter_type])

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
        logger.info("Creating pipeline...")

        for step in self.cfg.order:
            step_method_name = f"_add_{step}"
            if step_method_name in dir(self):
                step_method = eval(f"self.{step_method_name}")
                step_method()
            else:
                logger.debug(f"Pipeline step {step} is not available in HydraPipeline.")

        # NOTE: Choose order here.
        # params is added as argument in pipeline_steps decorator
        # self._add_values_filter()
        # self._add_resample()
        # self._add_filter()
        # # QUESTION: Should I propagate before interpolating?
        # self._add_propagate()
        # self._add_delete_cols()
        # self._add_interpolate()
        # self._add_scaler_func()

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

            # This won't work because we are repeatedly fitting pipeline with different storms
            # if has_storm_index(X):
            #     apply_storms(self.pipeline_.fit, X=X, y=y)
            # else:
            #     self.pipeline_.fit(X, y)

        return self

    def transform(self, X):
        if self.pipeline_ is None:
            return X
        else:
            return self.pipeline_.transform(X)

    def inverse_transform(self, X):
        if self.pipeline_ is None:
            return X
        else:
            return self.pipeline_.inverse_transform(X)
