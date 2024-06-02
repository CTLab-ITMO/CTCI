import numpy as np
from keras.models import Model, Sequential
from keras.layers import Activation, Dense, BatchNormalization, Dropout, Conv2D, concatenate, Conv2DTranspose, MaxPooling2D, UpSampling2D, Input, Reshape
from keras import backend as K
#from keras.layers.core import SpatialDropout2D
import tensorflow as tf
import glob
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from keras.src.layers import *
from keras.src.metrics import *
from keras.src.callbacks import *
from keras.src.optimizers import *
from sklearn.model_selection import train_test_split
from keras.losses import BinaryCrossentropy

class Dataset:
    def __init__(self, images_path, masks_path, train_val_split, shape):
        images_path += "/*.png"
        masks_path += "/*.png"
        self.images = [cv2.resize(cv2.imread(fname), shape).astype(np.uint8) / 255 for fname in glob.glob(images_path)[:16]]
        self.masks = [cv2.resize(cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2GRAY), shape).astype(np.uint8) / 255 for
                      fname in glob.glob(masks_path)[:16]]
        self.train_val_split = train_val_split
        self.__split()

    def __split(self):
        self.train_images, self.val_images, self.train_masks, self.val_masks = train_test_split(self.images, self.masks,
                                                                                                test_size=self.train_val_split)

    def get_splitted(self):
        return np.array(self.train_images).astype(np.float32), np.array(self.val_images).astype(np.float32),\
               np.array(self.train_masks).astype(np.float32), np.array(
            self.val_masks).astype(np.float32)

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

def dice_coef(y_true, y_pred, smooth = 100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def jacard_loss(y_true, y_pred, smooth = 100):
    return 1 - jacard_similarity(y_true, y_pred)

def average_precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def mean_average_precision(y_true, y_pred):
    average_precisions = K.map_fn(lambda x: average_precision(x[0], x[1]), (y_true, y_pred), dtype=K.floatx())
    return K.mean(average_precisions)

def dice_loss(y_true, y_pred, smooth=100):
    loss = 1 - dice_coef(y_true, y_pred, smooth)
    return loss

def bce_dice_loss(y_true, y_pred, smooth=100):
    loss = binary_crossentropy(y_true, y_pred) + \
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


def encoder(input, filters, conv_layers, kernel_size, activation, pool_size):
    for _ in range(conv_layers):
        conv = Conv2D(filters, kernel_size=kernel_size, activation=activation, padding='same')(input)
        input = conv
    pool = MaxPooling2D(pool_size=pool_size)(input)
    return pool, input


def decoder(conv1, conv2, filters, kernel_size, conv_layers):
    up = concatenate([Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(conv2), conv1], axis=3)
    for _ in range(conv_layers):
        input = Conv2D(filters, kernel_size=kernel_size, activation='relu', padding='same')(up)
        up = input
    return up


def bridge(input, count_layers, filters):
    for _ in range(count_layers):
        conv = Conv2D(filters, (3, 3), activation='relu', padding='same')(input)
        input = conv
    return input


def unet_md(loss, shape=(512, 512, 3)):
    input_layer = Input(shape=shape)

    pool1, conv1 = encoder(input_layer, 32, 2, (3, 3), 'relu', (2, 2))
    pool2, conv2 = encoder(pool1, 64, 2, (3, 3), 'relu', (2, 2))
    pool3, conv3 = encoder(pool2, 128, 2, (3, 3), 'relu', (2, 2))
    pool4, conv4 = encoder(pool3, 256, 2, (3, 3), 'relu', (2, 2))

    conv5 = bridge(pool4, 2, 512)

    conv6 = decoder(conv4, conv5, 256, (3, 3), 2)
    conv7 = decoder(conv3, conv6, 128, (3, 3), 2)
    conv8 = decoder(conv2, conv7, 64, (3, 3), 2)
    conv9 = decoder(conv1, conv8, 32, (3, 3), 2)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(input_layer, conv10)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=loss ,
                  metrics=['accuracy', Precision(), MeanIoU(num_classes=2), Recall(), dice_coeff, MeanAbsoluteError()])

    return model


def unet_plus_plus(loss, input_shape=(512, 512, 3), nb_filter = [32, 64, 128, 256, 512]):
    tf.keras.backend.clear_session()
    inputs = Input(input_shape)

    c1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.5)(c1)
    c1 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    c1 = Dropout(0.5)(c1)
    p1 = MaxPooling2D((2, 2), strides=(2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.5)(c2)
    c2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
    c2 = Dropout(0.5)(c2)
    p2 = MaxPooling2D((2, 2), strides=(2, 2))(c2)

    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(c2)
    conv1_2 = concatenate([up1_2, c1], name='merge12', axis=3)
    c3 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv1_2)
    c3 = Dropout(0.5)(c3)
    c3 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
    c3 = Dropout(0.5)(c3)

    conv3_1 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    conv3_1 = Dropout(0.5)(conv3_1)
    conv3_1 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv3_1)
    conv3_1 = Dropout(0.5)(conv3_1)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, c2], name='merge22', axis=3)  # x10
    conv2_2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv2_2)
    conv2_2 = Dropout(0.5)(conv2_2)
    conv2_2 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv2_2)
    conv2_2 = Dropout(0.5)(conv2_2)

    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, c1, c3], name='merge13', axis=3)
    conv1_3 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv1_3)
    conv1_3 = Dropout(0.5)(conv1_3)
    conv1_3 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv1_3)
    conv1_3 = Dropout(0.5)(conv1_3)

    conv4_1 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(pool3)
    conv4_1 = Dropout(0.5)(conv4_1)
    conv4_1 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv4_1)
    conv4_1 = Dropout(0.5)(conv4_1)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=3)  # x20
    conv3_2 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv3_2)
    conv3_2 = Dropout(0.5)(conv3_2)
    conv3_2 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv3_2)
    conv3_2 = Dropout(0.5)(conv3_2)

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, c2, conv2_2], name='merge23', axis=3)
    conv2_3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv2_3)
    conv2_3 = Dropout(0.5)(conv2_3)
    conv2_3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv2_3)
    conv2_3 = Dropout(0.5)(conv2_3)

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, c1, c3, conv1_3], name='merge14', axis=3)
    conv1_4 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv1_4)
    conv1_4 = Dropout(0.5)(conv1_4)
    conv1_4 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv1_4)
    conv1_4 = Dropout(0.5)(conv1_4)

    conv5_1 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(pool4)
    conv5_1 = Dropout(0.5)(conv5_1)
    conv5_1 = Conv2D(512, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv5_1)
    conv5_1 = Dropout(0.5)(conv5_1)

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=3)  # x30
    conv4_2 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv4_2)
    conv4_2 = Dropout(0.5)(conv4_2)
    conv4_2 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv4_2)
    conv4_2 = Dropout(0.5)(conv4_2)

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=3)
    conv3_3 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv3_3)
    conv3_3 = Dropout(0.5)(conv3_3)
    conv3_3 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv3_3)
    conv3_3 = Dropout(0.5)(conv3_3)

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, c2, conv2_2, conv2_3], name='merge24', axis=3)
    conv2_4 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv2_4)
    conv2_4 = Dropout(0.5)(conv2_4)
    conv2_4 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv2_4)
    conv2_4 = Dropout(0.5)(conv2_4)

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, c1, c3, conv1_3, conv1_4], name='merge15', axis=3)
    conv1_5 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv1_5)
    conv1_5 = Dropout(0.5)(conv1_5)
    conv1_5 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(conv1_5)
    conv1_5 = Dropout(0.5)(conv1_5)

    nestnet_output_4 = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_normal', name='output_4',
                              padding='same')(conv1_5)

    model = Model([inputs], [nestnet_output_4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=loss ,
                  metrics=['accuracy', Precision(), MeanIoU(num_classes=2), Recall(), dice_coeff, MeanAbsoluteError()])

    return model

def train_model(model, X_train, Y_train, X_val, Y_val, epochs = 10, batch_size = 16):
    model.fit(X_train, Y_train, validation_data= (X_val, Y_val), batch_size=batch_size, epochs=epochs)

def compute_metrics(model, model_name, X, Y):

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

        res.save("D:/floath/unet_tests/" + model_name + f"/{ind}.png")

    print(metrics)
    print()
    print("---------------------------------------------------")

def main():
    # dice_loss jacard_loss
    losses = {
        "bce_dice_loss": bce_dice_loss,
        "focal_loss": focal_loss,
        "bce": tf.keras.losses.BinaryCrossentropy(),
        "dice": dice_loss,
        "jacard": jacard_loss
    }
    '''
    images_path = "/home/mrumjanceva/ssl/data/dataset/image"
    masks_path = "/home/mrumjanceva/ssl/data/dataset/labels"

    '''
    images_path = "D:/floath/new_dataset/dataset/image"
    masks_path = "D:/floath/new_dataset/dataset/labels"


    X_train, X_val, Y_train, Y_val = Dataset(images_path=images_path, masks_path=masks_path, train_val_split=0.2,
                                             shape=(512, 512)).get_splitted()

    print(X_train[0].shape)

    epochs = 10
    batch_size = 16

    print(Y_val.shape, Y_train.shape)

    for loss_name in losses.keys():
        loss = losses[loss_name]

        print("----------------------------------------------------------------------")
        print(loss_name, "Unet")
        print("----------------------------------------------------------------------")
        unet = unet_md(loss)
        train_model(unet, X_train, Y_train, X_val, Y_val, epochs=epochs, batch_size=batch_size)

        compute_metrics(unet, loss_name, X_val, Y_val)

        unet.save(f"/home/mrumjanceva/ssl/barlow_twins/unet/weights_losses/unet_{loss_name}.keras")

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