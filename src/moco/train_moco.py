import random

import tensorflow as tf
from keras import Model
from keras.src.layers import *
from keras.src.regularizers import l2
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras import backend as K
import keras

batchsize = 8
epochs = 50

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



config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


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
        return np.array(self.train_images).astype(np.float32), \
               np.array(self.val_images).astype(np.float32), \
               np.array(self.train_masks).astype(np.float32), \
               np.array(self.val_masks).astype(np.float32)


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
    #print(image)
    image = flip_random_crop(image)
    image = random_apply(rotation, image, p=0.2)

    # image = random_apply(color_jitter, image, p=0.2)

    # image = random_apply(color_drop, image, p=0.2)

    # image = random_apply(solarize, image, p=0.2)
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


class DuplicatedCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        return self.transforms(img), self.transforms(img)


class RandomResizedCrop(tf.keras.layers.Layer):
    # taken from
    # https://keras.io/examples/vision/nnclr/#random-resized-crops
    def __init__(self, scale, ratio, crop_shape):
        super(RandomResizedCrop, self).__init__()
        self.scale = scale
        self.log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))
        self.crop_shape = crop_shape

    def call(self, images):
        batch_size = tf.shape(images)[0]

        random_scales = tf.random.uniform(
            (batch_size,),
            self.scale[0],
            self.scale[1]
        )
        random_ratios = tf.exp(tf.random.uniform(
            (batch_size,),
            self.log_ratio[0],
            self.log_ratio[1]
        ))

        new_heights = tf.clip_by_value(
            tf.sqrt(random_scales / random_ratios),
            0,
            1,
        )
        new_widths = tf.clip_by_value(
            tf.sqrt(random_scales * random_ratios),
            0,
            1,
        )
        height_offsets = tf.random.uniform(
            (batch_size,),
            0,
            1 - new_heights,
        )
        width_offsets = tf.random.uniform(
            (batch_size,),
            0,
            1 - new_widths,
        )

        bounding_boxes = tf.stack(
            [
                height_offsets,
                width_offsets,
                height_offsets + new_heights,
                width_offsets + new_widths,
            ],
            axis=1,
        )
        images = tf.image.crop_and_resize(
            images,
            bounding_boxes,
            tf.range(batch_size),
            self.crop_shape,
        )
        return images


transform = DuplicatedCompose(tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.05, 'constant'),
    RandomResizedCrop(scale=(0.9, 1.1), ratio=(0.9, 1.1), crop_shape=(512, 512)),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
]))


def normalize(x, eps=1e-4):
    return x / tf.reshape(tf.math.maximum(tf.reduce_sum(x ** 2, axis=-1), eps), (-1, 1))


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


def expend_as(tensor, rep):
    return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                  arguments={'repnum': rep})(tensor)





def build_encoder(shape, hidden_dim=128):
    inputs = Input(shape)
    s = Rescaling(1.0 / 255)(inputs)
    # s = inputs
    x, skip_1 = encoder(s)

    x = bottleneck(x)

    # Projections
    trunk_output = GlobalAvgPool2D()(x)
    projection_outputs = projection_head(trunk_output, hidden_dim=hidden_dim)

    model = Model(inputs, projection_outputs)
    return model, skip_1


images_path = ""
masks_path = ""

X_train, X_val, Y_train, Y_val = Dataset(images_path, masks_path,
                                         0.2, (512, 512)).get_splitted()

PROJECT_DIM = X_train[0].shape[1] / 2
model_q, skip_connetions = build_encoder((X_train[0].shape[1], X_train[0].shape[0], X_train[0].shape[2]),
                                         hidden_dim=PROJECT_DIM)
model_k = tf.keras.models.clone_model(model_q)
optimizer = tf.keras.optimizers.SGD()


def queue_data(data, k):
    #print(data, k)
    return tf.concat([data, k], axis=0)


def dequeue_data(data, K=4096):
    if len(data) > K:
        return data[-K:]
    else:
        return data


def initialize_queue(model_k, train_loader):
    queue = tf.zeros((0, 256), dtype=tf.float32)

    for batch_idx, (data, target) in enumerate(train_loader):
        x_k = transform(data)[0]
        k = model_k(x_k)
        k = tf.stop_gradient(k)
        queue = queue_data(queue, k)
        queue = dequeue_data(queue, K=10)
        break
    return queue


#print((X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
x_train = tf.reshape(X_train,(X_train.shape[0],X_train.shape[1],X_train.shape[2],3))
x_test = tf.reshape(X_val,(X_val.shape[0],X_val.shape[1],X_val.shape[2],3))
train_mnist = tf.data.Dataset.from_tensor_slices((x_train,Y_train))
test_mnist = tf.data.Dataset.from_tensor_slices((x_test,Y_val))
train_loader = train_mnist.batch(batchsize)
test_loader = test_mnist.batch(batchsize)


queue = initialize_queue(model_k, train_loader)

class MOCO(tf.keras.Model):
    def __init__(self, model_q, model_k, train_loader):
        super(MOCO, self).__init__()
        self.model_q = model_q
        self.model_k = keras.models.clone_model(model_q)
        self.queue = initialize_queue(model_k, train_loader)
        self.temp = 0.99999999

    def compile(self, optimizer, loss_fn):
        super(MOCO, self).compile(run_eagerly=True)
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, data_):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        data, target = data_
        x_q, x_k = transform(data)
        k = self.model_k(x_k)
        k = tf.stop_gradient(k)
        N = tf.shape(data)[0]
        K = tf.shape(self.queue)[0]
        with tf.GradientTape() as tape:
            q = self.model_q(x_q)
            a = tf.reshape(q, (N, 1, -1))
            b = tf.reshape(k, (N, -1, 1))
            l_pos = tf.matmul(a, b)
            l_neg = tf.matmul(tf.reshape(q, (N, -1)), tf.reshape(tf.transpose(self.queue), (-1, K)))
            logits = tf.concat([tf.reshape(l_pos, (N, 1)), l_neg], axis=1)
            labels = tf.zeros(N, dtype=tf.float32)
            loss = self.loss_fn(labels, logits / self.temp)
        grads = tape.gradient(loss, self.model_q.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.model_q.trainable_weights))

        for i in range(len(self.model_q.layers)):
            if len(self.model_q.layers[i].get_weights()) > 0:
                arr = []
                for ind in range(len(self.model_q.layers[i].get_weights())):
                    arr.append(
                        0.1 * self.model_k.layers[i].get_weights()[ind] + (1 - 0.1) * self.model_q.layers[i].get_weights()[ind])

                self.model_k.layers[i].set_weights(arr)

        self.queue = queue_data(self.queue, k)
        self.queue = dequeue_data(self.queue)

        return {"loss": loss}


moco = MOCO(model_q, model_k, train_loader)
moco.compile(optimizer, tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
moco.fit(train_loader, epochs = 10)

moco.model_k.save("")
moco.model_q.save("")
