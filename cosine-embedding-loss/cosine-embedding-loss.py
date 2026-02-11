def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    xx12 = sum([a*b for a, b in zip(x1, x2)])
    x1_norm = sum([a*a for a in x1])
    x2_norm = sum([a*a for a in x2])
    cos_x = xx12 / x1_norm**0.5 / x2_norm**0.5
    if label == 1:
        return 1 - cos_x
    return max(0, cos_x - margin)