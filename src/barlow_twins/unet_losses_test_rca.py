from keras.models import Model, Sequential
from keras.layers import Activation, Dense, BatchNormalization, Dropout, Conv2D, concatenate, Conv2DTranspose, MaxPooling2D, UpSampling2D, Input, Reshape
from keras import backend as K
#from keras.layers.core import SpatialDropout2D
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
from tensorflow.keras.metrics import *
from tensorflow.keras.layers import *
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import *
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
from hausdorff import hausdorff_distance
from tensorflow.keras.regularizers import l2
import keras

import os
import random
import glob
import numpy as np
import cv2 as cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.io import imread,imshow
from skimage.morphology import label
from skimage.transform import resize
import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd
from tensorflow.keras.metrics import *
import glob
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split
from keras.losses import BinaryCrossentropy


class Dataset:
    def __init__(self, images_path, masks_path, train_val_split, shape):
        images_path += "/*.png"
        masks_path += "/*.png"
        self.images = [cv2.resize(cv2.imread(fname), shape).astype(np.float16) / 255 for fname in
                       glob.glob(images_path)]
        self.masks = [cv2.resize(cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2GRAY), shape).astype(np.float16) / 255
                      for
                      fname in glob.glob(masks_path)]
        self.train_val_split = train_val_split
        self.__split()

    def __split(self):
        self.train_images, self.val_images, self.train_masks, self.val_masks = train_test_split(self.images, self.masks,
                                                                                                test_size=self.train_val_split)

    def get_splitted(self):
        return np.array(self.train_images).astype(np.float32),\
               np.array(self.val_images).astype(np.float32),\
               np.array(self.train_masks).astype(np.float32),\
               np.array(self.val_masks).astype(np.float32)


'''

Losses

'''


def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def jacard_similarity(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum((y_true_f + y_pred_f) - (y_true_f * y_pred_f))
    return intersection / union


def dice_coef(y_true, y_pred, smooth=100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def average_precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def mean_average_precision(y_true, y_pred):
    average_precisions = K.map_fn(lambda x: average_precision(x[0], x[1]), (y_true, y_pred), dtype=K.floatx())
    return K.mean(average_precisions)

def jacard_loss(y_true, y_pred, smooth=100):
    return 1 - jacard_similarity(y_true, y_pred)


def dice_loss(y_true, y_pred, smooth=100):
    loss = 1 - dice_coef(y_true, y_pred, smooth)
    return loss


def bce_dice_loss(y_true, y_pred, smooth=100):
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) + \
           dice_loss(y_true, y_pred, smooth)
    return loss / 2.0


def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred, smooth=100):
    weight_a = alpha * (1 - y_pred) ** gamma * targets
    weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

    return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(
        -logits)) * (weight_a + weight_b) + logits * weight_b


def focal_loss(y_true, y_pred, smooth=100):
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
                              1 - tf.keras.backend.epsilon())
    logits = tf.math.log(y_pred / (1 - y_pred))

    loss = focal_loss_with_logits(logits=logits, targets=y_true,
                                  alpha=0.25, gamma=2, y_pred=y_pred)

    return tf.reduce_mean(loss)


'''

Models

'''


def double_conv_layer(x, filter_size, size, dropout, batch_norm=False):
    axis = 3
    conv = SeparableConv2D(size, (filter_size, filter_size), padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    conv = SeparableConv2D(size, (filter_size, filter_size), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = Dropout(dropout)(conv)

    shortcut = Conv2D(size, kernel_size=(1, 1), padding='same')(x)
    if batch_norm is True:
        shortcut = BatchNormalization(axis=axis)(shortcut)

    res_path = add([shortcut, conv])
    return res_path


def encoder(inputs):
    num_filters = [16, 32, 64, 128]
    skip_connections = []
    x = inputs

    for i, f in enumerate(num_filters):
        a = double_conv_layer(x, 3, f, 0.1, True)
        skip_connections.append(a)
        x = MaxPooling2D(pool_size=(2, 2))(a)

    return x, skip_connections


def bottleneck(inputs):
    x = inputs
    f = 256

    x3 = double_conv_layer(x, 3, f, 0.1, True)

    return x3


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
    x = Conv2D(1, kernel_size=(1, 1))(inputs)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)

    return x

def build_encoder(shape, hidden_dim=128):
    inputs = Input(shape)
    s = keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
    # s = inputs
    x, skip_1 = encoder(s)

    x = bottleneck(x)

    trunk_output = GlobalAvgPool2D()(x)

    model = Model(inputs, trunk_output)
    return model, skip_1


def train_model(model, X_train, Y_train, X_val, Y_val, epochs = 10, batch_size = 16):
    model.fit(X_train, Y_train, validation_data= (X_val, Y_val), batch_size=batch_size, epochs=epochs)


def compute_metrics(model, model_name, X, Y, path):

    print(X[0])

    print("---------------------------------------------------")
    print("Metrics of model")
    print()
    print("delimeter")
    metrics = {"IoU": 0,
               "Acc": 0,
               "Recall": 0,
               "Precision": 0,
               "Dice": 0,
               "mAP": 0}


    for ind in range(X.shape[0]):
        prediction = (model.predict(np.expand_dims(X[ind], axis=0)) > 0.5).astype(np.float32)

        iou = MeanIoU(num_classes=2)
        iou.update_state(np.squeeze(Y[ind]), np.squeeze(prediction))
        iou = iou.result().numpy()
        metrics["IoU"] += iou / X.shape[0]

        acc = Accuracy()
        acc.update_state(np.squeeze(Y[ind]), np.squeeze(prediction))
        acc = acc.result().numpy()
        metrics["Acc"] += acc / X.shape[0]

        precision = Precision()
        precision.update_state(np.squeeze(Y[ind]), np.squeeze(prediction))
        precision = precision.result().numpy()
        metrics["Precision"] += precision / X.shape[0]

        dice = dice_coeff(np.squeeze(Y[ind]), np.squeeze(prediction))
        metrics["Dice"] += (dice / X.shape[0]).numpy()

        mAP = mean_average_precision(np.squeeze(Y[ind]), np.squeeze(prediction))
        metrics["mAP"] += ( mAP / X.shape[0]).numpy()

        res = Image.fromarray(cv2.cvtColor((np.squeeze(prediction) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))

        res.save(path + model_name + f"/{ind}.png")

    print(metrics)
    print()
    print("---------------------------------------------------")


def compile_model(model, optimizer, metrics):
    model.compile(
        loss=focal_loss,
        optimizer=optimizer,
        metrics=metrics
    )

def main():
    # dice_loss jacard_loss
    losses = {
        "bce_dice_loss": bce_dice_loss,
        "focal_loss": focal_loss,
        "bce": tf.keras.losses.BinaryCrossentropy()
    }

    images_path = "X_PATH"
    masks_path = "Y_PATH"


    X_train, X_val, Y_train, Y_val = Dataset(images_path=images_path, masks_path=masks_path, train_val_split=0.2,
                                             shape=(512, 512)).get_splitted()

    print(X_train[0].shape)

    epochs = 100
    batch_size = 16

    print(Y_val.shape, Y_train.shape)

    for loss_name in losses.keys():
        loss = losses[loss_name]

        print("----------------------------------------------------------------------")
        print(loss_name, "Unet")
        print("----------------------------------------------------------------------")
        unet_enc, skip_connetions = build_encoder((512, 512, 3),
                                                  hidden_dim=256)

        backbone = tf.keras.Model(
            unet_enc.input, unet_enc.layers[-9].output
        )

        new_skip_connections = [backbone.get_layer(index=11).output,
                                backbone.get_layer(index=22).output,
                                backbone.get_layer(index=33).output,
                                backbone.get_layer(index=44).output]

        backbone.trainable = True
        x = backbone.output
        # x = bottleneck(x)
        x = decoder(x, new_skip_connections)
        outputs = output(x)
        # print(np.array(X_train).shape, np.array(Y_train).shape)
        unet = Model(unet_enc.input, outputs)

        optimizer = Adam(learning_rate=0.0001)
        metrics = ['accuracy', Precision(), MeanIoU(num_classes=2), Recall(), dice_coeff,
                   MeanAbsoluteError()]  # ], my_iou_metric]
        compile_model(unet, optimizer, metrics)

        train_model(unet, X_train, Y_train, X_val, Y_val, epochs=epochs, batch_size=batch_size)

        compute_metrics(unet, loss_name, X_val, Y_val, "PATH_TO_YOUR_RESULTS")

        unet.save(f"{loss_name}.keras")

    '''
    for loss_name in losses.keys():
        loss = losses[loss_name]

        print("----------------------------------------------------------------------")
        print(loss_name, "Unet Plus Plus")
        print("----------------------------------------------------------------------")

        unetpp = unet_plus_plus(loss)
        train_model(unetpp, X_train, Y_train, X_val, Y_val, epochs=epochs, batch_size=batch_size)

        unetpp.save(f"/home/mrumjanceva/ssl/barlow_twins/unet/weights_losses/unetpp_{loss_name}.keras")
    '''

if __name__ == "__main__":
    main()