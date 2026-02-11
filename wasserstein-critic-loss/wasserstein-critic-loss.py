import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein Critic Loss for WGAN.
    """
    rs = np.asarray(real_scores, dtype=float)
    fs = np.asarray(fake_scores, dtype=float)

    return float(np.mean(fs) - np.mean(rs))