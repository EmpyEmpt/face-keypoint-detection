import tempfile
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from wandb.keras import WandbCallback


from src.data_tests import pass_tests_before_fitting
from src.model import compile_model
from src.dataset import fetch_ds


def save_model(model) -> None:
    _, keras_file = tempfile.mkstemp('.h5')
    tf.keras.models.save_model(model, keras_file, include_optimizer=False)
    print('Saved baseline model to:', keras_file)


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


def strip_pruning(config, model):
    model_for_export = tfmot.sparsity.keras.strip_pruning(model)

    _, pruned_keras_file = tempfile.mkstemp('.h5')
    tf.keras.models.save_model(
        model_for_export, pruned_keras_file, include_optimizer=False)

    print('Saved pruned Keras model to:', pruned_keras_file)

    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    pruned_tflite_model = converter.convert()
    _, pruned_tflite_file = tempfile.mkstemp('.tflite')

    with open(pruned_tflite_file, 'wb') as f:
        f.write(pruned_tflite_model)

    print('Saved pruned TFLite model to:', pruned_tflite_file)


def quantize(config, model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_and_pruned_tflite_model = converter.convert()

    _, quantized_and_pruned_tflite_file = tempfile.mkstemp('.tflite')

    with open(quantized_and_pruned_tflite_file, 'wb') as f:
        f.write(quantized_and_pruned_tflite_model)

# TODO: this needs thinking
# def evaluate_model(interpreter, test_images, test_labels):
#     input_index = interpreter.get_input_details()[0]["index"]
#     output_index = interpreter.get_output_details()[0]["index"]

#     # Run predictions on ever y image in the "test" dataset.
#     prediction_digits = []
#     for i, test_image in enumerate(test_images):
#         if i % 1000 == 0:
#             print('Evaluated on {n} results so far.'.format(n=i))
#         # Pre-processing: add batch dimension and convert to float32 to match with
#         # the model's input data format.
#         test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
#         interpreter.set_tensor(input_index, test_image)

#         # Run inference.
#         interpreter.invoke()

#         # Post-processing: remove batch dimension and find the digit with highest
#         # probability.
#         output = interpreter.tensor(output_index)
#         digit = np.argmax(output()[0])
#         prediction_digits.append(digit)

#     print('\n')
#     # Compare prediction results with ground truth labels to calculate accuracy.
#     prediction_digits = np.array(prediction_digits)
#     accuracy = (prediction_digits == test_labels).mean()
#     return accuracy


# def evaluate_tflite(model):
#     interpreter = tf.lite.Interpreter(model_content=model)
#     interpreter.allocate_tensors()
#     test_accuracy = evaluate_model(interpreter)
#     print('Pruned and quantized TFLite test_accuracy:', test_accuracy)
#     # print('Pruned TF test accuracy:', model_for_pruning_accuracy)


def compare_model(baseline, pruned, test_x, test_y):
    _, baseline_accuracy = baseline.evaluate(
        test_x, test_y, verbose=0)

    _, pruned_accuracy = pruned.evaluate(
        test_x, test_y, verbose=0)

    print('Baseline test accuracy:', baseline_accuracy)
    print('Pruned test accuracy:', pruned_accuracy)
