import numpy as np
import matplotlib.pyplot as plt


def cluster_errrate(pred_labels, gt, num_labels):
    sum_of_err = 0

    for i in range(num_labels):
        idx = np.where(pred_labels == i)[0]

        gt_label = np.array([gt[i] for i in idx])
        (values, counts) = np.unique(gt_label, return_counts=True)
        correct = np.max(counts)
        sum_of_err += (len(idx) - correct)

    return sum_of_err * 1.0 / pred_labels.shape[0]


def cluster_imgs(images, pred_labels, num_labels):
    max_cl = 0

    input_shape = images[0].shape

    clustered_img = []
    for i in range(num_labels):
        idx = np.where(pred_labels == i)[0]
        if len(idx) == 0:
            clustered_img.append(None)
            continue
        if len(idx) > max_cl:
            max_cl = len(idx)
        line_img = np.concatenate([images[i] for i in idx], axis=2)
        # print line_img.shape
        clustered_img.append(line_img)

    zero_pad = np.zeros((input_shape[0], input_shape[1] * num_labels, input_shape[2] * max_cl))
    for i in range(num_labels):
        cm = clustered_img[i]
        if cm is None:
            continue
        zero_pad[:, i*cm.shape[1]: (i+1)*cm.shape[1], :cm.shape[2]] = cm

    zero_pad = zero_pad.squeeze()
    return zero_pad


def code_visualize(code_list, labels):
    """
    :param code_list: (B, 64) code
    :param labels: (B,) labels
    :return:
    """
    label_type = np.unique(labels)
