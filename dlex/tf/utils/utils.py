from typing import List, Any


def pad_sequence(data: List[List[Any]], padding_value=0):
    max_len = max([len(seq) for seq in data])

    i = 0
    while len(data[i]) == 0:
        i += 1
        if i == len(data):
            raise ValueError("Empty input.")
    if isinstance(data[i][0], list) or isinstance(data[i][0], tuple):
        padding_value = [padding_value for _ in range(len(data[i][0]))]

    lengths = [max(len(seq), 1) for seq in data]
    data = [seq + [padding_value] * (max_len - len(seq)) for seq in data]
    return data, lengths