import numpy as np

def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    predictions = np.asarray(predictions, dtype=float)
    k = len(predictions)
    q = np.full_like(predictions, epsilon / k)
    q[target] = (1 - epsilon) + epsilon / k
    return -float(np.sum(q*np.log(predictions)))