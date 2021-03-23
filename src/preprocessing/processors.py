import pandas as pd
import numpy as np
import logging

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
    def __init__(self,
                 freq="T",
                 label="right",
                 func="mean",
                 verbose=True,
                 **kwargs):
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
        self.verbose = verbose
        self.kwargs = kwargs

    @staticmethod
    def _get_freq(X):
        return to_offset(pd.infer_freq(X.index))

    def fit(self, X, y=None):

        if self.func is None:
            self.func = "mean"

        self.freq = to_offset(self.freq)
        self.X_freq_ = self._get_freq(X)

        return self

    def transform(self, X, y=None):

        X_freq = self._get_freq(X)
        assert X_freq == self.X_freq_, "X does not have the correct frequency."

        if X_freq is None:
            X = X.resample(self.freq, label=self.label,
                           **self.kwargs).apply(self.func)
        elif self.freq > X_freq:
            X = X.resample(self.freq, label=self.label,
                           **self.kwargs).apply(self.func)
        else:
            if self.verbose:
                logger.debug(f"Specified frequency ({self.freq}) is <= data "
                             + f"frequency ({X_freq}). Resampling is ignored.")

        return X

    def inverse_transform(self, X):
        return X


class Interpolator(TransformerMixin):
    def __init__(self,
                 method="linear",
                 axis=0,
                 limit_direction="both",
                 limit=15,
                 **kwargs):
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

    def transform(self, X, y=None):

        X = X.interpolate(method=self.method,
                          axis=self.axis,
                          limit_direction=self.limit_direction,
                          **self.kwargs)

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
            X_ = self.transformer.inverse_transform(X.to_numpy().reshape(
                -1, 1))
            X_pd = pd.Series(X_.flatten(), name=self.name_, index=X.index)
        elif isinstance(X, pd.DataFrame):
            assert X.columns == self.columns_
            X_ = self.transformer.inverse_transform(X)
            X_pd = pd.DataFrame(X_, columns=self.columns_, index=X.index)

        return X_pd


class StormSubsetter(BaseEstimator, TransformerMixin):
    def __init__(self, times_path):
        self.times_path = times_path


    def _subset_storm(self, X, row):

        storm_num, storm = row

        if isinstance(X, pd.Series):
            X = X.to_frame()

        X_ = X.truncate(before=storm['start_time'],
                        after=storm['end_time'])
        X_['storm'] = storm_num

        return X_

    def fit(self, X, y=None):
        path = to_absolute_path(self.times_path)
        self.storm_times_ = pd.read_csv(path, index_col=0)

        return self

    def transform(self, X, y=None):
        storm_iter = self.storm_times_.iterrows()
        X_storms_iter = (self._subset_storm(X, row) for row in storm_iter)

        X_storms = pd.concat(X_storms_iter)
        X_storms.set_index([X_storms['storm'], X_storms.index], inplace=True)

        return X_storms


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


def limited_change_speed(speed, density, change_down=30, change_ups=[50, 10]):
    """Apply limited relative change to speed according to density"""

    # QUESTION: Which velocity do I use?

    out = speed.copy()
    for i, (d1, d2) in enumerate(zip(out[:-1], out[1:])):
        if np.isnan(d2):
            out[i + 1] = d1  # assume persistence
        else:
            if density[i + 1] > density[i]:
                out[i + 1] = np.nanmax((np.nanmin(
                    (d2, d1 + change_ups[0])), d1 - change_down))
            else:
                out[i + 1] = np.nanmax((np.nanmin(
                    (d2, d1 + change_ups[1])), d1 - change_down))
    return out


def _get_callable(obj_str):
    # TODO: Modify to allow scaler_str to be more general
    # TODO: Validation

    obj = eval(obj_str)

    return obj


class HydraPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, cfg):
        self.cfg = cfg
        self.pipeline_list = []

        # NOTE: Add new method for each processing step

    def _add_interpolate(self):
        if OmegaConf.is_none(self.cfg, "interpolate"):
            return None
        else:
            logger.debug("Interpolating...")

        if isinstance(self.cfg.interpolate, bool):
            if self.cfg.interpolate:
                self.pipeline_list.append(("interpolator", Interpolator()))
        elif OmegaConf.is_config(self.cfg.interpolate):
            self.pipeline_list.append(
                ("interpolator", Interpolator(**self.cfg.interpolate)))

    def _add_resample(self):
        if OmegaConf.is_none(self.cfg, "resample"):
            return None
        else:
           logger.debug("Resampling...")

        if OmegaConf.is_config(self.cfg.resample):
            self.pipeline_list.append(
                ("resampler", Resampler(**self.cfg.resample)))

    def _add_scaler_func(self):

        if (OmegaConf.is_none(self.cfg, "scaler")
                and OmegaConf.is_none(self.cfg, "func")):
            return None

        # NOTE: scaler ignored when func is specified
        func = self.cfg.func.func
        inverse_func = self.cfg.func.inverse_func
        scaler = self.cfg.scaler.scaler

        if func is not None:
            logger.debug(f"Scaling using function {func}...")

            func = _get_callable(func)
            inverse_func = _get_callable(inverse_func)
            func_transformer = FunctionTransformer(func=func,
                                                   inverse_func=inverse_func,
                                                   **self.cfg.func.kwargs)
            self.pipeline_list.append(
                ("func", PandasTransformer(transformer=func_transformer)))

        elif scaler is not None:
            logger.debug(f"Scaling using scaler {scaler}...")

            scaler_callable = _get_callable(scaler)
            self.pipeline_list.append(
                ("scaler",
                 PandasTransformer(transformer=scaler_callable(),
                                   **self.cfg.scaler.kwargs)))
        else:
            logger.debug("scaler or func was not specified.")

    def _add_filter(self):

        if OmegaConf.is_none(self.cfg, "filter"):
            return None

        if self.cfg.filter.type == "limited_change":
            logger.debug("Filtering using filter {self.filter.type}...")
            # TODO
            pass

    def _add_subset_storms(self):

        if OmegaConf.is_none(self.cfg, "subset_storms"):
            return None

        logger.debug("Adding storm subsetter to pipeline...")
        storm_subsetter = StormSubsetter(self.cfg.subset_storms.times_path)
        self.pipeline_list.append(("storm_subsetter", storm_subsetter))


    def create_pipeline(self):
        # NOTE: Choose order here.

        logger.debug("Creating pipeline...")
        with open_dict(self.cfg):
            self._add_resample()
            self._add_interpolate()
            self._add_subset_storms()
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
