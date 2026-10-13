import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    n = len(y)
    all_labels = np.unique(y, return_counts=True)
    labels = all_labels[0]
    counts_l = all_labels[1]

    h = 0.
    for c in counts_l:
        p = c / n
        h -= p * np.log(p)
        
    return h / np.log(2)