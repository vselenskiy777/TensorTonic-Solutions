import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    a = np.asarray(A, dtype=int)
    m, n = a.shape
    #print(m, n)
    b = np.zeros((n, m), dtype=int)
    for i in range(m):
        for j in range(n):
            b[j][i] = a[i][j]
    return b
    
