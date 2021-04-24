#!/usr/bin/env python

from . import xgb, ebm

# Dictionary with model name as keys and HydraModel class as value
MODELS = {"xgboost": xgb.HydraXGB, "ebm": ebm.HydraEBM}


def get_model(name):
    # API for getting model specified in config files
    return MODELS[name]
