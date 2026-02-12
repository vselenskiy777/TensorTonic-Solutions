import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    yn_true = np.asarray(y_true, dtype=int)
    yn_pred = np.asarray(y_pred, dtype=float)

    indices = np.arange(len(y_true), dtype=int)
    probs_correct = yn_pred[indices, yn_true]

    return -np.mean(np.log(probs_correct))