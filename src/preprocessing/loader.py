#!/usr/bin/env python

import logging
import re
import src.preprocessing.load

from functools import partialmethod
from hydra.utils import to_absolute_path
from src import utils


logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(
        self,
        start="2010-01-01",
        end="2019-12-31",
        features="omni",
        target="symh",
        data_dir=to_absolute_path("data/"),
        load_module=src.preprocessing.load,
    ):
        self.start = start
        self.end = end
        self.features = features
        self.target = target
        self.data_dir = data_dir
        self.load_module = load_module

        logger.debug("Initiating Loader...")
        logger.debug(f"Start={start}, End={end}")

    def _load_func(self, input_type, name):
        # Get function with name load_{input_type}_{name}
        r = re.compile(f"^load_{input_type}_{name}")
        load_func = None
        for name, func in vars(self.load_module).items():
            if r.match(name):
                load_func = func
                break

        assert (
            load_func is not None
        ), f"Couldn't find function load_{input_type}_{name} in {self.load_module.__name__}"

        return load_func

    def _load(self, input_type, name, **kwargs):
        logger.debug(f"Loading {input_type}: {name}")

        # TODO: Input Validation
        # if input_type not in self.load_func_dict.keys():
        #     raise utils.NotSupportedError(input_type, "Input type")

        # if name not in self.load_func_dict[input_type].keys():
        #     raise utils.NotSupportedError(name, input_type)

        load_func = self._load_func(input_type, name)
        return load_func(start=self.start, end=self.end, **kwargs)

    load_features = partialmethod(_load, input_type="features")
    load_target = partialmethod(_load, input_type="target")
