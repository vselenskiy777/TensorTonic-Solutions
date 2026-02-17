import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    pp = np.asarray(p, dtype=float)
    qq = np.asarray(q, dtype=float)
    
    qq = qq + eps
    pp = np.where(pp==0, qq, pp)
    
    return float(np.sum(pp*np.log(pp/qq)))