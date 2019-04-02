def ser(pr, gt, start_ids):
    count = 0
    correct = 0
    is_correct = False
    for i in range(len(pr)):
        if gt[i] in start_ids:
            count += 1
            if pr[i] in start_ids:
                if is_correct: correct += 1
            is_correct = pr[i] == gt[i]
        if gt[i] != pr[i]: is_correct = False

    if is_correct: correct += 1
    return correct / count
