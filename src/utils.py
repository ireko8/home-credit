"""
Home Credit Default Risk
DenoisingAutoEncoder's code

this code defines utility functions

writer: Ireko8
"""

import random
from datetime import datetime
import tensorflow as tf
import numpy as np
import pandas as pd


def set_seed(seed):
    """Set seed for random functions."""
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def remove_all_zero(X_train, X_test, var_limit=0, corr_limit=1):
    """Remove columns which values are zero."""
    train_len = X_train.shape[0]
    df = pd.concat([X_train, X_test])

    v = df.sum(axis=0)
    not_zero_cols = v[v != 0].index

    df = df[not_zero_cols]

    X_train, X_test = df.iloc[:train_len], df.iloc[train_len:]

    return X_train, X_test
