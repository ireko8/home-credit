"""
Home Credit Default Risk
DenoisingAutoEncoder's code

this code defines architecture of DAE and swapnoise generator.

this code is based on OsciiArt's DAE kernel, but some funcions is modified.
https://www.kaggle.com/osciiart/denoising-autoencoder

writer: Ireko8
"""

import numpy as np
import keras.layers as kl
from keras import Model
import config


def get_DAE(num_size):
    """Keras architecture of DenoisingAutoEncoder."""
    inputs = kl.Input((num_size, ))
    ls = config.ae_hls

    e1 = kl.Dense(ls)(inputs)
    e1 = kl.Activation('relu')(e1)

    e2 = kl.Dense(ls)(e1)
    e2 = kl.Activation('relu')(e2)

    e3 = kl.Dense(ls)(e2)
    e3 = kl.Activation('relu')(e3)

    outputs = kl.Dense(num_size)(e3)
    dae = Model(inputs=inputs, outputs=outputs)
    encoders = []
    for e in [e1, e2, e3]:
        encoders.append(Model(inputs=inputs, outputs=e))
    dae.compile(optimizer='adam', loss='mse')

    return dae, encoders


def x_generator(X, batch_size, shuffle=True):
    """Generate batch of input."""
    batch_index = 0
    n = X.shape[0]
    while True:
        if batch_index == 0:
            index_array = np.arange(n)
            if shuffle:
                index_array = np.random.permutation(n)

        current_index = (batch_index * batch_size) % n
        if n >= current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = n - current_index
            batch_index = 0

        batch_x = X[index_array[current_index:current_index +
                                current_batch_size]]

        yield batch_x


def mix_generator(X, batch_size, swaprate=0.15, shuffle=True):
    """Generate Swapnoised input and output."""
    gen1 = x_generator(X, batch_size, shuffle)
    num_cols = X.shape[1]
    while True:
        batch1 = next(gen1)
        new_batch = batch1.copy()
        batch1_size = batch1.shape[0]
        num_swap = int(swaprate * batch1_size)
        for i in range(num_cols):
            # swapnoise part
            swap_idx = np.random.choice(
                batch1_size, 2 * num_swap, replace=False)
            new_batch[swap_idx[:num_swap], i] = batch1[swap_idx[num_swap:], i]

        yield (new_batch, batch1)
