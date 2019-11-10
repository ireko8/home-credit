"""
Home Credit Default Risk
DenoisingAutoEncoder's code

this code defines how to preprocess the data.
RankGauss normalization and onehot encoding.

this code is based on OsciiArt's DAE kernel
https://www.kaggle.com/osciiart/denoising-autoencoder

writer: Ireko8
"""

import numpy as np
import pandas as pd
from scipy.special import erfinv


def rankgauss(df):
    """Rankgauss Normalization.

    you can learn more infomation about rankgauss from the links below.
    https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
    http://fastml.com/preparing-continuous-features-for-neural-networks-with-rankgauss/
    """
    df = df.replace(np.inf, np.nan)
    df = df.replace(-np.inf, np.nan)
    num_df = df
    num_df = num_df.replace(np.nan, 0)
    print('ranking...')
    num_df = num_df.rank(axis=0) / num_df.shape[0]
    print('scaling...')
    num_df = 2 * num_df - 1
    print('replace min/max values')
    num_df = num_df.replace(-1, -0.99999)
    num_df = num_df.replace(1, 0.99999)
    print('erfinv')
    num_df = erfinv(num_df)
    return num_df


def make_onehot(df, cat_cols=[]):
    """Preprocess of data to onehot and rankgaussed data."""
    num_cols = df.columns[~df.columns.isin(cat_cols)]
    if len(cat_cols) > 0:
        df_cat = df[cat_cols].astype(object)
        df_num = rankgauss(df[num_cols])
        onehot = pd.get_dummies(df_cat, dummy_na=True)
        df = pd.concat([df_num, onehot], axis=1)
    else:
        df = rankgauss(df[num_cols])

    return df
