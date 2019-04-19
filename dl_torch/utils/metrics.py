"""Evaluation metrics"""

def ser(predicted, ground_truth, pass_ids):
    """Segment error rate."""
    count = 0
    correct = 0
    is_correct = False
    for i, _pr in enumerate(predicted):
        _gt = ground_truth[i]
        if _gt not in pass_ids:
            count += 1
            if _pr not in pass_ids:
                if is_correct:
                    correct += 1
            is_correct = _pr == _gt
        if _gt != _pr:
            is_correct = False

    if is_correct:
        correct += 1
    if count == 0:
        print(ground_truth)
        print(predicted)
    return correct / count
