import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    pn = np.asarray(p, dtype=float)
    yn = np.asarray(y, dtype=float)
    
    dice = 2 * np.sum(pn*yn) / (eps + np.sum(pn+yn))
    if np.sum(pn+yn)==0:
        return 0.
    return float(1 - dice)