import numpy as np
from scipy.ndimage import zoom


def zoom_lil_matrix(matrix, big_shape, lil_shape):
    ratio = big_shape[0] / lil_shape[0]
    new_lil_matrix = zoom(matrix, (ratio, ratio), mode='reflect', order=5)
    return new_lil_matrix
def dezoom_ref_matrix(matrix, big_shape, lil_shape):
    ratio = lil_shape[0] / big_shape[0]
    new_ref_matrix = zoom(matrix, (ratio, ratio), mode='reflect', order=5)
    return new_ref_matrix


def error(ref_matrix, error_matrix):
    diff = ref_matrix - error_matrix
    norm = np.linalg.norm(diff)/np.linalg.norm(ref_matrix)
    return norm
