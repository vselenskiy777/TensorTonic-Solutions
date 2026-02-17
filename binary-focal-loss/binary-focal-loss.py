import numpy as np

def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    targets = np.asarray(targets, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    pt = np.where(targets==1, predictions, 1-predictions)
    print(pt)
    return float(-alpha * np.mean((1 - pt)**gamma * np.log(pt)))