def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    n = len(set(recommended[:k])&set(relevant))
    precision = n / k
    recall = n / len(relevant)
    return [precision, recall]