import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.sum(a*a)==0.:
        return 0.
    if np.sum(b*b)==0.:
        return 0.
    return float(np.sum(a*b) / np.sum(a*a)**0.5 / np.sum(b*b)**0.5)