import numpy as np


def extract_state_vector(lambda_val, beta_val, A_val, coupling_matrix_g, mi_timeseries, L=None):
    if L is None:
        L = np.eye(coupling_matrix_g.shape[0])
    param_coords = [float(lambda_val), float(beta_val), float(A_val)]
    g_norm = float(np.linalg.norm(coupling_matrix_g))
    smoothness = float(np.trace(coupling_matrix_g @ L @ coupling_matrix_g.T))
    mi_range = float(np.max(mi_timeseries) - np.min(mi_timeseries)) if len(mi_timeseries) else 0.0
    return np.array(param_coords + [g_norm, smoothness, mi_range], dtype=float)
