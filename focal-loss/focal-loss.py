import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    eps = 1e-15
    pp = np.asarray(p, dtype=float)
    yy = np.asarray(y, dtype=float)
    
    p_pclipped = np.clip(pp, eps, 1 - eps)
    p1 = (1-p_pclipped)**gamma * yy * np.log(p_pclipped)
    p2 = p_pclipped**gamma * (1-yy) * np.log(1 - p_pclipped)
    return -(p1+p2).mean()