import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    z1n = np.asarray(Z1, dtype=float)
    z2n = np.asarray(Z2, dtype=float)
    
    A = z1n @ z2n.T / temperature
    row_max = np.max(A, axis=1, keepdims=True)
    # print(A, row_max)
    A_stable = A - row_max
    A_exp = np.exp(A_stable)
    # print(np.log(A_exp.sum(axis=1)))
    loss =  np.mean(np.log(A_exp.sum(axis=1)) - np.diagonal(A_stable))
    return loss