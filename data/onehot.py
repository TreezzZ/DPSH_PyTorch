#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
import numpy as np


class Onehot(object):
    """one-hot编码transform
    """

    def __init__(self):
        pass

    def __call__(self, sample, num_classes=10):
        target_onehot = torch.zeros(num_classes)
        target_onehot[sample] = 1

        return target_onehot


def encode_onehot(labels):
    """one-hot编码labels

    Parameters
        labels: ndarray
        标签

    Returns
        onehot_labels: ndarray
        onehot编码后的标签
    """
    onehot_labels = np.zeros((len(labels), 10))

    for i in range(len(labels)):
        onehot_labels[i, labels[i]] = 1

    return onehot_labels


if __name__ == "__main__":
    t = Onehot()
    sample = 2

    print(t(sample))

