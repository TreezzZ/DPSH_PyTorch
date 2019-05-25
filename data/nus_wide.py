# -*- coding:utf-8 -*-


import numpy as np
import os


def load_data(path, train=True):
    """加载NUS-WIDE数据集

    Parameters
        path: str
        数据集路径

        train: bool
        True: 加载训练数据; False: 加载测试数据

    Returns
        data: ndarray
        数据

        tags: ndarray
        标签
    """
    if train:
        data = np.load(os.path.join(path, 'nus_wide_train_data.npy'))
        tags = np.load(os.path.join(path, 'nus_wide_train_tags.npy'))
    else:
        data = np.load(os.path.join(path, 'nus_wide_test_data.npy'))
        tags = np.load(os.path.join(path, 'nus_wide_test_tags.npy'))

    return data, tags
