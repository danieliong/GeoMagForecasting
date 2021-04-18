#!/usr/bin/env python

from ._models import HydraXGB, HydraEBM

# Dictionary with model name as keys and HydraModel class as value
MODELS_DICT = {"xgboost": HydraXGB, "ebm": HydraEBM}


def get_model(name):
    # API for getting model specified in config files
    return MODELS_DICT[name]
