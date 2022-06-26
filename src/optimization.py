import tempfile
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from wandb.keras import WandbCallback


from src.data_tests import pass_tests_before_fitting
from src.model import compile_model
from src.dataset import fetch_ds


def prune(config, model):
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    # Compute end step to finish pruning after 2 epochs.
    tf.random.set_seed(config['random_seed'])

    train_dataset, test_dataset = fetch_ds(config, 'optimize')

    # Define model for pruning.
    # TODO: WHAT ARE THOOOOOSE
    end_step = np.ceil(
        config['amount'] / config['optimize']['batch_size']).astype(np.int32) * config['optimize']['epochs']
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                 final_sparsity=0.80,
                                                                 begin_step=0,
                                                                 end_step=end_step)
    }

    model_for_pruning = prune_low_magnitude(model, **pruning_params)

    # `prune_low_magnitude` requires a recompile.
    model_for_pruning = compile_model(
        input_shape=config['img_shape'], output_shape=config['kp_shape'], model=model)

    # model_for_pruning.summary()
    logdir = tempfile.mkdtemp()
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
    ]

    #  TODO: wandbcallback with tfds???
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep(),
                 tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
                 EarlyStopping(**config['callbacks']['EarlyStopping']),
                 ReduceLROnPlateau(**config['callbacks']['ReduceLROnPlateau']),
                 WandbCallback(**config['callbacks']['WandbCallback'])]

    # data tests (pre-fitting)
    pass_tests_before_fitting(
        data=train_dataset, img_shape=config['img_shape'], keypoint_shape=config['kp_shape'])
    pass_tests_before_fitting(
        data=test_dataset, img_shape=config['img_shape'],  keypoint_shape=config['kp_shape'])

    # training
    history = model_for_pruning.fit(
        train_dataset, epochs=config['optimize']['epochs'], validation_data=test_dataset, callbacks=callbacks)

    return history, model_for_pruning


def strip_pruning(model, save_path):
    # TODO: use path from config
    model_for_export = tfmot.sparsity.keras.strip_pruning(model)

    _, pruned_keras_file = tempfile.mkstemp(save_path + 'pruned.h5')
    tf.keras.models.save_model(
        model_for_export, pruned_keras_file, include_optimizer=False)

    print('Saved pruned Keras model to:', pruned_keras_file)

    return model_for_export


def quantize(model, save_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_and_pruned_tflite_model = converter.convert()

    _, quantized_and_pruned_tflite_file = tempfile.mkstemp(
        save_path + 'quantized.tflite')

    with open(quantized_and_pruned_tflite_file, 'wb') as f:
        f.write(quantized_and_pruned_tflite_model)
