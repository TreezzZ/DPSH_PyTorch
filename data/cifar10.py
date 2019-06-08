# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio
import torch.utils.data as data
from PIL import Image

import os
import sys
import pickle


def load_data_gist(path, train=True):
    """加载对cifar10使用gist提取的数据

    Parameters
        path: str
        数据路径

        train: bools
        True，加载训练数据; False，加载测试数据

    Returns
        data: ndarray
        数据

        labels: ndarray
        标签
    """
    mat_data = sio.loadmat(path)

    if train:
        data = mat_data['traindata']
        labels = mat_data['traingnd'].astype(np.int)
    else:
        data = mat_data['testdata']
        labels = mat_data['testgnd'].astype(np.int)

    return data, labels


class CIFAR10(data.Dataset):
    """加载官网下载的CIFAR10数据集"""
    data = []
    targets = []
    query_data = []
    query_targets = []
    train_data = []
    train_targets = []
    all_data = []
    all_targets = []

    @staticmethod
    def init_load_data(root,
                       num_train=5000, num_query=1000,
                       ):
        data_list = ['data_batch_1',
                     'data_batch_2',
                     'data_batch_3',
                     'data_batch_4',
                     'data_batch_5',
                     'test_batch',
                     ]
        base_folder = 'cifar-10-batches-py'

        for file_name in data_list:
            file_path = os.path.join(root, base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                CIFAR10.data.append(entry['data'])
                if 'labels' in entry:
                    CIFAR10.targets.extend(entry['labels'])
                else:
                    CIFAR10.targets.extend(entry['fine_labels'])

        CIFAR10.data = np.vstack(CIFAR10.data).reshape(-1, 3, 32, 32)
        CIFAR10.data = CIFAR10.data.transpose((0, 2, 3, 1))  # convert to HWC
        CIFAR10.targets = np.array(CIFAR10.targets)

        # 类别平衡划分
        # 按类别排序，聚集在一起
        sort_index = CIFAR10.targets.argsort()
        CIFAR10.data = CIFAR10.data[sort_index, :]
        CIFAR10.targets = CIFAR10.targets[sort_index]

        query_index = np.random.permutation(6000)[0:num_query//10]
        for i in range(1, 10):
            query_index = np.concatenate((query_index, np.random.permutation(6000)[0:num_query//10] + i * 6000))

        rest_index = set(range(60000)) - set(query_index)
        rest_index = np.array(list(rest_index))

        rest_index.sort()

        train_index = np.random.permutation(5900)[0:num_train//10]
        for i in range(1, 10):
            train_index = np.concatenate((train_index, np.random.permutation(5900)[0:num_train//10] + i * 5900))
        train_index = rest_index[train_index]

        # perm_index = np.random.permutation(CIFAR10.data.shape[0])
        # query_index = perm_index[:num_query]
        # train_index = perm_index[num_query: num_query + num_train]

        CIFAR10.query_data = CIFAR10.data[query_index, :]
        CIFAR10.query_targets = CIFAR10.targets[query_index]

        CIFAR10.train_data = CIFAR10.data[train_index, :]
        CIFAR10.train_targets = CIFAR10.targets[train_index]

        CIFAR10.all_data = CIFAR10.data
        CIFAR10.all_targets = CIFAR10.targets

    def __init__(self, mode='train',
                 transform=None, target_transform=None,
                 ):
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform

        if mode == 'train':
            self.data = self.train_data
            self.targets = self.train_targets
        elif mode == 'query':
            self.data = self.query_data
            self.targets = self.query_targets
        elif mode == 'all':
            self.data = self.all_data
            self.targets = self.all_targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)
