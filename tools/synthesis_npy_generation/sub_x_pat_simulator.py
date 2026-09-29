import numpy as np
import matplotlib.pyplot as plt
import sys
import importlib
import os

# heat_maps_path = "heat_maps/npy_matrix/8bit_resnet_20"

def get_prob_matrix(heat_maps_path):
    heat_map_files = sorted([f for f in os.listdir(heat_maps_path) if f.endswith(".npy")])
    cumulative_heatmap = None
    for f in heat_map_files:
        matrix = np.load(os.path.join(heat_maps_path, f))
        matrix = matrix.sum(axis=0)
        if cumulative_heatmap is None:
            cumulative_heatmap = matrix.copy()
        else:
            cumulative_heatmap += matrix
    prob_matrix = cumulative_heatmap / np.sum(cumulative_heatmap)
    return prob_matrix

def compute_adapted_mult(i: int, y: int, bit_width: int, sub_x_pat_multiplier) -> int:
    """
    If bit_width < 8, trunc the (8 - bit_width) LSB in input
    and insert zeros (left shift of 2 * shift) on the result.
    If bit_width == 8, directly call the multiplier.
    """
    shift = 8 - bit_width
    if shift > 0:
        i_hw = i >> shift
        y_hw = y >> shift
        res_hw = sub_x_pat_multiplier.approx_mult(i_hw, y_hw)
        return int(res_hw << (2 * shift))
    elif shift == 0:
        return int(sub_x_pat_multiplier.approx_mult(i, y))
    else:
        # If bit_width > 8, perform padding with shift
        ext = bit_width - 8
        return int(sub_x_pat_multiplier.approx_mult(i << ext, y << ext) >> (2 * ext))

def multiplier_test(bit_width, filename, sub_x_pat_multiplier):
    scrumbled_res = np.zeros((256, 256), dtype=np.int64)
    for i in range(256):
        for y in range(256):
            scrumbled_res[i, y] = compute_adapted_mult(i, y, bit_width, sub_x_pat_multiplier)
    np.save(filename, scrumbled_res)

def get_mult_caracteristics(bit_width, sub_x_pat_multiplier, heat_maps_path):
    max_error = 0
    mean_ae = 0.0
    mean_ae_cnn = 0.0
    prob_matrix = get_prob_matrix(heat_maps_path)

    for i in range(256):
        for y in range(256):
            scrumbled = compute_adapted_mult(i, y, bit_width, sub_x_pat_multiplier)
            exact = i * y
            diff = abs(scrumbled - exact)
            
            mean_ae += diff
            mean_ae_cnn += diff * prob_matrix[i, y]
            if diff > max_error:
                max_error = diff

    mean_ae = mean_ae / (256 * 256)
    return mean_ae, mean_ae_cnn, max_error

def execute_save(bit_width, filename, heat_maps_path):

    try:
        import tools.synthesis_npy_generation.sub_x_pat_multiplier as sub_x_pat_multiplier
    except ImportError:
        import sub_x_pat_multiplier

    importlib.invalidate_caches()
    sub_x_pat_multiplier = importlib.reload(sub_x_pat_multiplier)

    mean_ae, mean_ae_cnn, max_error = get_mult_caracteristics(bit_width, sub_x_pat_multiplier, heat_maps_path)
    multiplier_test(bit_width, filename, sub_x_pat_multiplier)
    return mean_ae, mean_ae_cnn, max_error