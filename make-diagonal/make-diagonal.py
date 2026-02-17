import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """
    n = len(v)
    d = np.zeros((n, n))
    for i in range(n):
        d[i][i] = v[i]
    return d
