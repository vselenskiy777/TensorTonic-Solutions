import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    y_pred_n = np.asarray(y_pred, dtype=float)
    y_true_n = np.asarray(y_true, dtype=float)

    loss = np.mean((y_pred_n - y_true_n) ** 2)
    return float(loss)
