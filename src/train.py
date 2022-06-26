import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from wandb.keras import WandbCallback

from src.data_tests import pass_tests_before_fitting
from src.model import compile_model
from src.dataset import fetch_ds


def train(config):
    tf.random.set_seed(config['random_seed'])

    train_dataset, test_dataset = fetch_ds(config, 'train')

    # model
    model = compile_model(
        input_shape=config['img_shape'], output_shape=config['kp_shape'])

    callbacks = [EarlyStopping(**config['callbacks']['EarlyStopping']),
                 ReduceLROnPlateau(**config['callbacks']['ReduceLROnPlateau']),
                 WandbCallback(**config['callbacks']['WandbCallback'])]

    # data tests (pre-fitting)
    pass_tests_before_fitting(
        data=train_dataset, img_shape=config['img_shape'], keypoint_shape=config['kp_shape'])
    pass_tests_before_fitting(
        data=test_dataset, img_shape=config['img_shape'],  keypoint_shape=config['kp_shape'])

    # training
    history = model.fit(
        train_dataset, epochs=config['train']['epochs'], validation_data=test_dataset, callbacks=callbacks)

    return history, model
