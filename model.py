import tensorflow as tf
import config as cfg
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D

"""Here lies everything connected to model itself
Archirecture is as it is - fiddle around with it:
    add more layers, change 'magical numbers', change the layers - its all here
And lots of utulity-simplification functions for saving/loading/validating model

And yeah... model has to be compiled so I don't remember why compilation and building are split..."""


def build(image_size=cfg.IMAGE_SIZE, outputs=cfg.OUTPUTS):
    """Well, you yourself should not really use this - use compile_model"""
    input_shape = (image_size, image_size, 3)

    model = Sequential()
    model.add(Conv2D(32, (5, 5), padding='same',
              input_shape=input_shape))
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    model.add(Conv2D(64, (5, 5)))
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),
              strides=(2, 2), padding='valid'))

    model.add(Conv2D(48, (5, 5)))
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),
              strides=(2, 2), padding='valid'))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),
              strides=(2, 2), padding='valid'))

    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(500, activation='relu'))
    model.add(Dense(outputs))
    return model


def compile_model(image_size=cfg.IMAGE_SIZE):
    """Compiles the model with a given {input_size}"""
    model = build(image_size)

    model.compile(
        loss=tf.keras.losses.mean_squared_error,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=['mse'])

    return model


def load_model(path=cfg.MODEL_PATH):
    """Load the model from provided {path}"""
    model = tf.keras.models.load_model(path)
    return model


def load_weights(model, path):
    """Loads saved set of weights for already existing {model} from {path}
    There is no way to save the weight in my(stupid) framework, so why is it needed I wonder..."""
    model.load_weights(path)
    return model


def evaluate_model(model, x, y):
    """Evaluates given {model} on a given sets ({x}, {y})"""
    loss, acc = model.evaluate(x, y, verbose=2)
    print(f'Model accuracy: {acc*100:5.2f}%')


def save_model(model, path):
    """Saves {model} to a given {path}"""
    model.save(path)
