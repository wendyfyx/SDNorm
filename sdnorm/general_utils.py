import os
import logging
import json

import numpy as np


def custom_json_serializer(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float_, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int_, np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(data, fname):
    '''Save data to json file'''
    with open(fname, 'w') as fp:
        json.dump(data, fp, default=custom_json_serializer, indent=2)
        logging.info(f"Saved json data to {fname}.")


def load_json(fname):
    '''Load data from json file'''
    with open(fname, 'r') as f:
        logging.debug(f"Loading json data from {fname}.")
        return json.load(f)


def get_rng(rng=None):
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    elif isinstance(rng, np.random.Generator):
        rng = rng
    else:
        rng = np.random.default_rng(0)
    return rng


def random_select(data, n_sample=1, rng=0):
    '''
        Randomly select samples from data, returns index
        Example usage: data[random_select(data)]
    '''
    rng = get_rng(rng)
    return rng.choice(len(data), size=n_sample, replace=False)