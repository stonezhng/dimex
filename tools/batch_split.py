import numpy as np


def get_split(total_list, batch_size):
    split = []
    l = len(total_list)
    for i in range(l / batch_size):
        if i < l / batch_size - 1:
            split.append(total_list[i * batch_size: (i+1) * batch_size])
        else:
            split.append(total_list[i * batch_size:])
    return split
