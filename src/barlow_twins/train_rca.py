import random

import tensorflow as tf

import functools
import argparse
import numpy as np
import cv2 as cv2
from keras.models import Model
from keras import backend as K
from keras.regularizers import l2

from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import Adam

from shapely import hausdorff_distance
from tensorflow._api.v2.compat.v1 import keras
from data_downloader import Dataset

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


'''
from keras.src.layers import *
from keras.src.metrics import *
from keras.src.callbacks import *
from keras.src.optimizers import *
'''

'''

Global Augmentation Bools

'''

crop_bool = True
rotation_bool = True
color_jitter_bool = True
color_drop_bool = True
solarize_bool = True

'''

Metrics 

'''


def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    value = 0.
    for batch in range(batch_size):
        value = value + IoU(y_true_in[batch], y_pred_in[batch])
    return value / batch_size


def my_iou_metric(label, pred):
    metric_value = tf.py_function(iou_metric_batch, [label, pred], tf.float32)
    return metric_value


def my_iou_metric_loss(label, pred):
    loss = 1 - tf.py_function(iou_metric_batch, [label, pred], tf.float32)
    # loss = -tf.map_fn(my_iou_metric_loss(label, pred), tf.range(tf.shape(pred)[0]))
    loss.set_shape((None,))

    return loss


def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


def haud_dist(y_true, y_pred):
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    return hausdorff_distance(y_true, y_pred)


def haud_dist_batch(y_true, y_pred):
    if len(y_true.shape) == 2:
        return haud_dist(y_true, y_pred)
    else:
        batch_size = y_true.shape[0]
    hd = 0.
    for batch in range(batch_size):
        hd = hd + haud_dist(y_true[batch], y_pred[batch])
    return hd / batch_size


def my_haud_dist(label, pred):
    metric_value = tf.py_function(haud_dist_batch, [label, pred], tf.float32)
    return metric_value


'''

Barlow Twins Part

'''


def off_diagonal(x):
    n = tf.shape(x)[0]
    flattened = tf.reshape(x, [-1])[:-1]
    off_diagonals = tf.reshape(flattened, (n - 1, n + 1))[:, 1:]
    return tf.reshape(off_diagonals, [-1])


def normalize_repr(z):
    z_norm = (z - tf.reduce_mean(z, axis=0)) / tf.math.reduce_std(z, axis=0)
    return z_norm


def compute_loss(z_a, z_b, lambd):
    batch_size = tf.cast(tf.shape(z_a)[0], z_a.dtype)
    repr_dim = tf.shape(z_a)[1]

    z_a_norm = normalize_repr(z_a)
    z_b_norm = normalize_repr(z_b)

    c = tf.matmul(z_a_norm, z_b_norm, transpose_a=True) / batch_size
    on_diag = tf.linalg.diag_part(c) + (-1)
    on_diag = tf.reduce_sum(tf.pow(on_diag, 2))
    off_diag = off_diagonal(c)
    off_diag = tf.reduce_sum(tf.pow(off_diag, 2))
    loss = on_diag + (lambd * off_diag)
    return loss


class BarlowLoss(tf.keras.losses.Loss):
    def __init__(self, batch_size: int):
        super().__init__()
        self.lambda_amt = 5e-3
        self.batch_size = batch_size

    def get_off_diag(self, c: tf.Tensor) -> tf.Tensor:
        zero_diag = tf.zeros(c.shape[-1])
        return tf.linalg.set_diag(c, zero_diag)

    def cross_corr_matrix_loss(self, c: tf.Tensor) -> tf.Tensor:
        c_diff = tf.pow(tf.linalg.diag_part(c) - 1, 2)
        off_diag = tf.pow(self.get_off_diag(c), 2) * self.lambda_amt
        loss = tf.reduce_sum(c_diff) + tf.reduce_sum(off_diag)

        return loss

    def normalize(self, output: tf.Tensor) -> tf.Tensor:
        return (output - tf.reduce_mean(output, axis=0)) / tf.math.reduce_std(
            output, axis=0
        )

    def cross_corr_matrix(self, z_a_norm: tf.Tensor, z_b_norm: tf.Tensor) -> tf.Tensor:
        return (tf.transpose(z_a_norm) @ z_b_norm) / self.batch_size

    def call(self, z_a: tf.Tensor, z_b: tf.Tensor) -> tf.Tensor:
        z_a_norm, z_b_norm = self.normalize(z_a), self.normalize(z_b)
        c = self.cross_corr_matrix(z_a_norm, z_b_norm)
        loss = self.cross_corr_matrix_loss(c)
        return loss


class BarlowTwins(tf.keras.Model):
    def __init__(self, model_encoder, lambd=5e-3):
        super(BarlowTwins, self).__init__()
        self.model_encoder = model_encoder
        self.lambd = lambd
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        ds_one, ds_two = data

        with tf.GradientTape() as tape:
            z_a, z_b = self.model_encoder(ds_one, training=True), self.model_encoder(ds_two, training=True)
            loss = compute_loss(z_a, z_b, self.lambd)

        gradients = tape.gradient(loss, self.model_encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model_encoder.trainable_variables))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def get_config(self):
        base_config = super().get_config()
        config = {
            'model_encoder' : self.model_encoder,
            'lambd' : self.lambd,
            'loss_tracker' : self.loss_tracker,
        }
        return {**base_config, **config}


def random_resize_crop(image, image_size):
    rand_size = tf.random.uniform(
        shape=[],
        minval=int(0.75 * image_size),
        maxval=1 * image_size,
        dtype=tf.int32,
    )

    crop = tf.image.random_crop(image, (rand_size, rand_size, 3))
    crop_resize = tf.image.resize(crop, (image_size, image_size))
    return crop_resize


@tf.function
def flip_random_crop(image):
    image = tf.image.random_flip_left_right(image)
    image = random_resize_crop(image, image.shape[0])
    return image


@tf.function
def float_parameter(level, maxval):
    return tf.cast(level * maxval / 10.0, tf.float32)


@tf.function
def sample_level(n):
    return tf.random.uniform(shape=[1], minval=0.1, maxval=n, dtype=tf.float32)


@tf.function
def rotation(image):
    augmented_image = tf.image.rot90(image)
    return augmented_image


@tf.function
def solarize(image, level=6):
    threshold = float_parameter(sample_level(level), 1)
    return tf.where(image < threshold, image, 255 - image)


@tf.function
def color_jitter(x, strength=0.5):
    x = tf.image.random_brightness(x, max_delta=0.8 * strength)
    x = tf.image.random_contrast(
        x, lower=1 - 0.8 * strength, upper=1 + 0.8 * strength
    )
    x = tf.image.random_saturation(
        x, lower=1 - 0.8 * strength, upper=1 + 0.8 * strength
    )
    x = tf.image.random_hue(x, max_delta=0.2 * strength)
    x = tf.clip_by_value(x, 0, 255)
    return x


@tf.function
def color_drop(x):
    x = tf.image.rgb_to_grayscale(x)
    x = tf.tile(x, [1, 1, 3])
    return x


def random_apply(func, x, p):
    if random.random() < p:
        return func(x)
    else:
        return x


def custom_augment(image):
    global crop_bool, rotation_bool, color_jitter_bool, color_drop_bool, solarize_bool
    image = tf.cast(image, tf.float32)
    print(image)
    image = flip_random_crop(image)  if crop_bool else image
    image = random_apply(rotation, image, p=0.2)  if rotation_bool else image

    image = random_apply(color_jitter, image, p=0.2)  if color_jitter_bool else image

    image = random_apply(color_drop, image, p=0.2) if color_drop_bool else image

    image = random_apply(solarize, image, p=0.2) if solarize_bool else image
    return image


@tf.autograph.experimental.do_not_convert
def get_ssl_ds(X_train):
    input_height = IMG_HEIGHT = X_train[0].shape[0]
    input_width = IMG_WIDTH = X_train[0].shape[1]
    IMG_CHANNELS = X_train[0].shape[2]
    AUTO = tf.data.experimental.AUTOTUNE
    CROP_TO = IMG_HEIGHT
    SEED = 42
    BATCH_SIZE = 4
    ssl_ds_one = tf.data.Dataset.from_tensor_slices(X_train)
    ssl_ds_one = (
        ssl_ds_one.shuffle(1024, seed=SEED)
            .map(custom_augment, num_parallel_calls=AUTO)
            .batch(BATCH_SIZE)
            .prefetch(AUTO)
    )

    ssl_ds_two = tf.data.Dataset.from_tensor_slices(X_train)
    ssl_ds_two = (
        ssl_ds_two.shuffle(1024, seed=SEED)
            .map(custom_augment, num_parallel_calls=AUTO)
            .batch(BATCH_SIZE)
            .prefetch(AUTO)
    )

    return tf.data.Dataset.zip((ssl_ds_one, ssl_ds_two))


class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Implements an LR scheduler that warms up the learning rate for some training steps
    (usually at the beginning of the training) and then decays it
    with CosineDecay (see https://arxiv.org/abs/1608.03983)
    """

    def __init__(
            self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")
        learning_rate = (
                0.5
                * self.learning_rate_base
                * (
                        1
                        + tf.cos(
                    self.pi
                    * (tf.cast(step, tf.float32) - self.warmup_steps)
                    / float(self.total_steps - self.warmup_steps)
                )
                )
        )

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                            self.learning_rate_base - self.warmup_learning_rate
                    ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )


def train_barlow_twins(ssl_ds, lr_decayed_fn, model_encoder, EPOCHS=100, BATCH_SIZE=16):
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_decayed_fn, momentum=0.9)
    # optimizer=keras.optimizers.RMSprop(lr=0.001)
    barlow_twins = BarlowTwins(model_encoder)
    loss = BarlowLoss(BATCH_SIZE)
    barlow_twins.compile(optimizer=optimizer, loss=loss)

    print("Training")
    barlow_twins.model_encoder.get_weights()[0]

    history = barlow_twins.fit(ssl_ds, epochs=EPOCHS)
    return barlow_twins


def expend_as(tensor, rep):
    return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                  arguments={'repnum': rep})(tensor)


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


'''

UNET Encoder Part

'''


def projection_head(x, WEIGHT_DECAY=5e-4, hidden_dim=128):
    """Constructs the projection head."""
    for i in range(2):
        x = Dense(
            hidden_dim,
            name=f"projection_layer_{i}",
            kernel_regularizer=l2(WEIGHT_DECAY),
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    outputs = Dense(hidden_dim, name="projection_output")(x)
    return outputs


def build_encoder(shape, hidden_dim=128):
    inputs = Input(shape)
    s = keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
    # s = inputs
    x, skip_1 = encoder(s)

    x = bottleneck(x)

    # Projections
    trunk_output = GlobalAvgPool2D()(x)
    projection_outputs = projection_head(trunk_output, hidden_dim=hidden_dim)

    model = Model(inputs, projection_outputs)
    return model, skip_1


def compile_model(model, optimizer, metrics):
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=optimizer,
        metrics=metrics
    )


def train_model(model, keyname, X, Y, X_val, Y_val):
    print("Training")
    model_checkpoint = ModelCheckpoint('model_' + keyname + '.hdf5', monitor='loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='loss', verbose=1, patience=20)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
    results = model.fit(X, Y, validation_data=(X_val, Y_val), batch_size=32, epochs=6,
                        callbacks=[model_checkpoint, reduce_lr,
                                   early_stopping])
    return model


def main():
    global crop_bool, rotation_bool, color_jitter_bool, color_drop_bool, solarize_bool

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('images', type=str, help='path to images')
    parser.add_argument('masks', type=str, help='path to masks')

    parser.add_argument('height', type=int, help='height of images')
    parser.add_argument('width', type=int, help='width of images')
    parser.add_argument('size', type=int, help='size of training BT batch')
    parser.add_argument('epochs', type=int, help='count of epochs')

    parser.add_argument('cropperb', type=bool, help='cropping augm.')
    parser.add_argument('rotatedb', type=bool, help='rotation augm.')
    parser.add_argument('jitterb', type=bool, help='jitter augm.')
    parser.add_argument('droppedb', type=bool, help='color drop augm.')
    parser.add_argument('solarizedb', type=bool, help='solarize augm.')

    parser.add_argument('checkpoint', type=str, help='checkpoint name')
    print(parser.parse_args().__dict__)
    args = parser.parse_args()

    images_path = args.images
    masks_path = args.masks
    target_height = args.height
    target_width = args.width
    batch_size = args.size
    EPOCHS = args.epochs
    crop_bool = args.cropperb
    rotation_bool = args.rotatedb
    color_jitter_bool = args.jitterb
    color_drop_bool = args.droppedb
    solarize_bool = args.solarizedb
    keyname = args.checkpoint

    X_train, X_val, Y_train, Y_val = Dataset(images_path, masks_path,
                                             0.2, (target_height, target_width)).get_splitted()

    ssl_ds = get_ssl_ds(X_train)
    PROJECT_DIM = X_train[0].shape[1] / 2
    BATCH_SIZE = batch_size
    WEIGHT_DECAY = 5e-4
    val_split = 0.2

    STEPS_PER_EPOCH = len(X_train) // BATCH_SIZE
    TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS
    WARMUP_EPOCHS = int(EPOCHS * 0.1)
    WARMUP_STEPS = int(WARMUP_EPOCHS * STEPS_PER_EPOCH)

    lr_decayed_fn = WarmUpCosine(
        learning_rate_base=1e-3,
        total_steps=EPOCHS * STEPS_PER_EPOCH,
        warmup_learning_rate=0.0,
        warmup_steps=WARMUP_STEPS
    )

    unet_enc, skip_connetions = build_encoder((X_train[0].shape[1], X_train[0].shape[0], X_train[0].shape[2]),
                                              hidden_dim=PROJECT_DIM)

    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_decayed_fn, momentum=0.9)
    # optimizer=keras.optimizers.RMSprop(lr=0.001)
    barlow_twins = BarlowTwins(unet_enc)
    loss = BarlowLoss(BATCH_SIZE)
    barlow_twins.compile(optimizer=optimizer, loss=loss)

    print("Training")
    barlow_twins.model_encoder.get_weights()[0]

    history = barlow_twins.fit(ssl_ds, epochs=EPOCHS)

    weights_path = "PATH_TO_YOUR_WEIGHTS_FOLDER"
    if crop_bool:
        weights_path += "_cropped"
    if rotation_bool:
        weights_path += "_rotationed"
    if color_jitter_bool:
        weights_path += "_jittered"
    if solarize_bool:
        weights_path += "_solarized"
    if color_drop_bool:
        weights_path += "_colordropped"
    weights_path += ".h5"

    barlow_twins.save_weights(weights_path)

    backbone = tf.keras.Model(
        barlow_twins.model_encoder.input, barlow_twins.model_encoder.layers[-9].output
    )

    new_skip_connections = [backbone.get_layer(index=11).output,
                            backbone.get_layer(index=22).output,
                            backbone.get_layer(index=33).output,
                            backbone.get_layer(index=44).output]

    backbone.trainable = False
    x = backbone.output
    # x = bottleneck(x)
    x = decoder(x, new_skip_connections)
    outputs = output(x)
    print(np.array(X_train).shape, np.array(Y_train).shape)
    model = Model(barlow_twins.model_encoder.input, outputs)

    print(model.summary())

    optimizer = Adam()
    metrics = ['accuracy', Precision(), MeanIoU(num_classes=2), Recall(), dice_coeff,
               MeanAbsoluteError()]  # ], my_iou_metric]
    compile_model(model, optimizer, metrics)

    X_train, X_val, Y_train, Y_val = np.array(X_train), np.array(X_val), np.array(Y_train), np.array(Y_val)

    model_checkpoint = ModelCheckpoint('model_' + keyname + '.hdf5', monitor='loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='loss', verbose=1, patience=20)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
    results = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=32, epochs=6,
                        callbacks=[model_checkpoint, reduce_lr,
                                   early_stopping])


if __name__ == "__main__":
    main()
