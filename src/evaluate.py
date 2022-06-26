import tensorflow as tf
import numpy as np


def compare_baseline_to_pruned(baseline, pruned, test_x, test_y):
    _, baseline_accuracy = baseline.evaluate(
        test_x, test_y, verbose=0)

    _, pruned_accuracy = pruned.evaluate(
        test_x, test_y, verbose=0)

    print('Baseline test accuracy:', baseline_accuracy)
    print('Pruned test accuracy:', pruned_accuracy)


def compare_baseline_to_tflite(baseline, quantized_model):
    # TODO: pls make this work
    interpreter = tf.lite.Interpreter(model_content=quantized_model)
    interpreter.allocate_tensors()
    test_accuracy = evaluate_model(interpreter)
    print('Pruned and quantized TFLite test_accuracy:', test_accuracy)
    # print('Pruned TF test accuracy:', model_for_pruning_accuracy)


def evaluate_model(interpreter, test_images, test_labels):
    # TODO: this needs rework
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Run predictions on ever y image in the "test" dataset.
    prediction_digits = []
    for i, test_image in enumerate(test_images):
        if i % 1000 == 0:
            print('Evaluated on {n} results so far.'.format(n=i))
        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input data format.
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, test_image)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        output = interpreter.tensor(output_index)
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)

    print('\n')
    # Compare prediction results with ground truth labels to calculate accuracy.
    prediction_digits = np.array(prediction_digits)
    accuracy = (prediction_digits == test_labels).mean()
    return accuracy
