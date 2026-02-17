import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    a = np.asarray(A, dtype=float)
    trace = 0
    for i in range(a.shape[0]):
        trace += a[i][i]
    return trace
