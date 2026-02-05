import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    error = y_true - y_pred
    abs_error = np.abs(error)
    
    huber_loss = np.where(abs_error>delta, (abs_error - delta/2)*delta, abs_error*abs_error / 2)
    
    return np.mean(huber_loss)