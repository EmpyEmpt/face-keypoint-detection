import tensorflow as tf
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import Reshape


def build(image_size: int, outputs: tuple = (68, 2)):
    """Builds a model"""
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
    model.add(Dense(outputs[0] * outputs[1]))
    model.add(Reshape(outputs))
    return model


def compile_model(image_size: int):
    """Compiles the model with a given {input_size}"""
    model = build(image_size)

    model.compile(
        loss=tf.keras.losses.mean_squared_error,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        metrics=['mse'])

    return model


def load_model(path: str):
    """Load the model from given {path}"""
    model = tf.keras.models.load_model(path)
    return model


def evaluate_model(model, x, y):
    """Evaluates given {model} on a given sets ({x}, {y})"""
    _, acc = model.evaluate(x, y, verbose=2)
    print(f'Model accuracy: {acc*100:5.2f}%')


def save_model(model, path: str):
    """Saves {model} to a given {path}"""
    model.save(path)
