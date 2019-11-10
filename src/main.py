"""
Home Credit Default Risk
DenoisingAutoEncoder's code

this code defines how to execute DAE and dump output of hidden layers.

some code is based on OsciiArt's DAE kernel.
https://www.kaggle.com/osciiart/denoising-autoencoder

writer: Ireko8
"""

import gc
from pprint import pformat
from pathlib import Path
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from neural_net import get_DAE, mix_generator
from utils import set_seed, remove_all_zero
from preprocess import make_onehot
import config


def proc_autoencoder(X, ae_dir):
    """Fit DAE(DenoisingAutoEncoder) to data."""
    print(f"hidden size {config.ae_hls}")
    print(f"swap_rate {config.swap_rate}")

    gen = mix_generator(X.values, config.batch_size, swaprate=config.swap_rate)
    dae, encoders = get_DAE(X.shape[1])

    with open(ae_dir / 'AutoEncoder.txt', 'w') as fp:
        dae.summary(print_fn=lambda x: fp.write(x + '\n'))

    callbacks = [
        EarlyStopping(
            monitor='loss',
            patience=10,
            verbose=1,
            min_delta=0.0001,
            mode='min'),
        ReduceLROnPlateau(
            monitor='loss',
            factor=0.1,
            patience=5,
            verbose=1,
            min_delta=0.0001,
            mode='min')
    ]
    dae.fit_generator(
        generator=gen,
        steps_per_epoch=np.ceil(X.shape[0] / config.batch_size),
        epochs=config.num_epoch,
        verbose=1,
        callbacks=callbacks)
    return dae, encoders


def main(X_train, X_test, dir_path):
    """Convert the feature to autoencoder's outputs of each hidden layer."""
    dir_path = Path(dir_path)
    dir_path.mkdir(exist_ok=True, parents=True)

    df = pd.concat([X_train, X_test])
    train_len = X_train.shape[0]

    print("make onehot")

    df = make_onehot(df, cat_cols=config.cat_cols_804)
    cols = df.columns
    print(cols)
    print("done")

    print('autoencoder start')
    dae, encoders = proc_autoencoder(df, dir_path)
    print("done")

    print("encoding...")
    aes = []

    for e in encoders:
        aes.append(e.predict(df))
    df = np.hstack(aes)

    gc.collect()
    print("done")

    new_cols = [f'AE_{i}' for i in range(df.shape[1])]
    X_train = pd.DataFrame(df[:train_len], columns=new_cols)
    X_test = pd.DataFrame(df[train_len:], columns=new_cols)

    print("save hidden outputs")

    X_train, X_test = remove_all_zero(X_train, X_test)

    print(pformat(X_train.columns))
    fname = f'AE_{config.swap_rate}_{config.ae_hls}_all_X'
    X_train.to_feather(dir_path / f'{fname}_train_reduction.ftr')
    X_test.to_feather(dir_path / f'{fname}_test_reduction.ftr')

    print("done")


def load_data(fpath):
    """Load data."""
    print("data load")

    onodera_path = Path(fpath)
    train = pd.read_feather(onodera_path / 'X_train_LB0.804.f')
    test = pd.read_feather(onodera_path / 'X_test_LB0.804.f')

    print("base done")

    print("data load done")
    gc.collect()

    return train, test


if __name__ == '__main__':

    seed = config.seed

    set_seed(seed)
    print(f"seed = {seed}")

    ae_dir_name = 'output'
    fpath = 'input/'
    train, test = load_data(fpath)
    main(train, test, ae_dir_name)
