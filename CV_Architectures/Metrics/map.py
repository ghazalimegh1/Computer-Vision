def n_points_technique(precision, recall, n=11):
    if len(precision) != len(recall):
        raise ValueError(
            "precision and recall must have the same length"
        )
    """
    Computes the n-point interpolated average precision.

    Args:
        precision (list): List of precision values.
        recall (list): List of recall values.
    """
    interpolated_precision = []
    for r in [i / (n - 1) if n > 1 else 0 for i in range(n)]:
        p = max([p for p, r_ in zip(precision, recall) if r_ >= r], default=0.0)
        interpolated_precision.append(p)
    return sum(interpolated_precision) / n


def mean_average_precision(precisions, recalls, n=11):
    """
    Computes the mean average precision (mAP).

    Args:

        precision (list): List of precision values.
        recall (list): List of recall values.
        n (int): Number of points for interpolation.
    """
    aps = []
    for precision, recall in zip(precisions, recalls):
        ap = n_points_technique(precision, recall, n)
        aps.append(ap)
    return sum(aps) / len(aps) if aps else 0.0
    


