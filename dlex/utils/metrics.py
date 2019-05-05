"""Evaluation metrics"""

def ser(predicted, ground_truth, pass_ids):
    """Segment error rate.
    
    Args:
        predicted (list):
        ground_truth (list):
        pass_ids (list): list of id which does not start a new tag (inside)

    Returns:
        correct (int):
        count (int):
    """
    count = 0
    correct = 0
    is_correct = False
    for i, _pr in enumerate(predicted):
        _gt = ground_truth[i]
        if _gt not in pass_ids:
            count += 1
            if _pr not in pass_ids:
                correct += 1 if is_correct else 0
            is_correct = _pr == _gt
        if _gt != _pr:
            is_correct = False

    if is_correct:
        correct += 1
    return correct, count
