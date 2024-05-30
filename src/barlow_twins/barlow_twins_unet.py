

import random

import matplotlib.pyplot as plt
import tensorflow as tf

import functools
import argparse
import numpy as np
import cv2 as cv2

from keras.models import Model
from keras import backend as K
from keras.regularizers import l2

'''
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import Adam
'''

from shapely import hausdorff_distance

from sklearn.model_selection import train_test_split
import glob
from keras.src import layers
from keras.src.layers import *
from keras.src.metrics import *
from keras.src.callbacks import *
from keras.src.optimizers import *




def expend_as(tensor, rep):
    return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                  arguments={'repnum': rep})(tensor)

def double_conv_layer(input, filter_size, kernel_size, dropout, batch_norm=False):
    print(filter_size)
    input = Conv2D(filter_size, kernel_size=kernel_size, activation='relu', padding='same')(input)
    input = Conv2D(filter_size, kernel_size=kernel_size, activation='relu', padding='same')(input)
    return input

def encoder(inputs):
    num_filters = [16, 32, 64, 128]
    skip_connections = []
    x = inputs

    for i, f in enumerate(num_filters):

        a = double_conv_layer(x, f, 3, 0.1, True)
        skip_connections.append(a)
        x = MaxPooling2D(pool_size=(2, 2))(a)

    return x, skip_connections

def bottleneck(inputs):
    x = inputs
    f = 256
    x = Conv2D(f, 1, activation = 'relu')(x)
    x = Conv2D(f, 1, activation = 'relu')(x)

    # x3 = double_conv_layer(x, 3, f, 0.1, True)

    return x

def decoder(inputs, skip_connections):
    num_filters = [128, 64, 32, 16]
    skip_connections.reverse()
    x = inputs
    batch_norm = True

    for i, f in enumerate(num_filters):

        x_up = UpSampling2D(size=(2, 2), data_format="channels_last")(x)
        x_att = concatenate([x_up, skip_connections[i]], axis=-1)

        x = double_conv_layer(x_att, 3, f, 0.1, True)
    return x

def output(inputs):
    x = Conv2D(1, kernel_size=(1 ,1))(inputs)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)

    return x



def projection_head(x, hidden_dim=128):
    """Constructs the projection head."""
    for i in range(2):
        x = Dense(
            hidden_dim,
            name=f"projection_layer_{i}",
            kernel_regularizer=l2(0.0001),
        )(x)
        x = Activation("relu")(x)
    outputs = Dense(hidden_dim, name="projection_output")(x)
    return outputs


def build_encoder(shape, hidden_dim=128):
    inputs = Input(shape)
    s = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
    # s = inputs
    x, skip_1 = encoder(s)

    # x = bottleneck(x)
    f = 256
    x = Conv2D(f, 1, activation='relu')(x)
    x = Conv2D(f, 1, activation='relu')(x)

    # Projections
    trunk_output = GlobalAvgPool2D()(x)
    projection_outputs = projection_head(trunk_output, hidden_dim=hidden_dim)

    model = Model(inputs, projection_outputs)
    return model, skip_1



PROJECT_DIM = IMG_HEIGHT/2
BATCH_SIZE = 8
EPOCHS = 100
WEIGHT_DECAY = 5e-4
TRAIN_FLG = 0 # 0 - No Training, 1 - Training
val_split = 0.3

STEPS_PER_EPOCH = len(X_train) // BATCH_SIZE
TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS
WARMUP_EPOCHS = int(EPOCHS * 0.1)
WARMUP_STEPS = int(WARMUP_EPOCHS * STEPS_PER_EPOCH)

unet_enc, skip_connetions = build_encoder((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), hidden_dim=256)