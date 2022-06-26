from src.optimization import prune, strip_pruning, quantize
from src.evaluate import compare_baseline_to_pruned, compare_baseline_to_tflite


def optimize(config, model):
    _, pruned_model = prune(config, model)
    # compare_baseline_to_pruned(model, pruned_model)
    pruned_model = strip_pruning(pruned_model, 'SOME_SAVE_PATH_FROM_CONFIG')
    _, quantized_model = quantize(pruned_model, 'SOME_SAVE_PATH_FROM_CONFIG')
    # compare_baseline_to_tflite(model, quantized_model)
