import tensorflow as tf
import os
import config as cfg
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D


def build(size):
    model = Sequential()

    # model.add(BatchNormalization(input_shape=(size, size, 3)))
    model.add(Conv2D(24, (5, 5), padding='same',
              input_shape=(size, size, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(Conv2D(36, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),
              strides=(2, 2), padding='valid'))

    model.add(Conv2D(48, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),
              strides=(2, 2), padding='valid'))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),
              strides=(2, 2), padding='valid'))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(1000, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(136))
    return model


def compile_model(load_latest=False, size=cfg.CROP_SIZE, checkpoint_path=cfg.CHECKPOINT_PATH):
    model = build(size)
    model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

    if load_latest:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(latest)

    return model


def load_model(path):
    model = tf.keras.models.load_model(path)
    return model


def evaluate_model(model, x, y):
    loss, acc = model.evaluate(x, y, verbose=2)
    print(f'Model, accuracy: {acc*100:5.2f}%')


def save_model(model, path):
    model.save(path)
