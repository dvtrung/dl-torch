def ser(pr, gt, pass_ids):
    count = 0
    correct = 0
    is_correct = False
    for i in range(len(pr)):
        if gt[i] not in pass_ids:
            count += 1
            if pr[i] not in pass_ids:
                if is_correct: correct += 1
            is_correct = pr[i] == gt[i]
        if gt[i] != pr[i]: is_correct = False

    if is_correct: correct += 1
    if count == 0:
        print(gt)
        print(pr)
    return correct / count
