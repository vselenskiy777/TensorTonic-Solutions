import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    yy = np.asarray(y, dtype=float)
    if len(aa.shape)>1:
        di = np.linalg.norm(aa - bb, axis=1)
        li = yy*di**2 + (1 - yy) * np.maximum(0, margin-di)**2
        if reduction=="mean":
            return np.mean(li)
        return np.sum(li)
    di = np.linalg.norm(aa - bb)
    li = yy*di**2 + (1 - yy) * np.maximum(0, margin-di)**2
    return li[0]