#!/usr/bin/env python
# -*- coding: utf-8 -*-

import data.cifar10 as cifar10
import data.nus_wide as nus_wide
from data.onehot import Onehot
from data.onehot import encode_onehot

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import numpy as np


def dataloader(path, train=True, batch_size=64, num_workers=4, download=False):
    """返回训练数据加载器

    Parameters
        path: str
        训练数据集文件路径

        train: bool
        True: 返回训练数据加载器; False: 返回测试数据加载器

        batch_size: int
        batch size

        num_workers: int
        number of dataloader workers

        download: bool
        True: 下载数据集; False: 不下载数据集

    Returns
        dataloader: DataLoader
        数据加载器
    """
    dataset = CIFAR10(root=path,
                      train=train,
                      transform=_transform(),
                      target_transform=Onehot(),
                      download=download,
                      )
    return DataLoader(dataset,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      )


def load_data(path,
              dataset,
              train=True,
              normalized=True,
              one_hot=True,
              batch_size=64,
              num_workers=4,
              ):
    """返回数据和标签

    Parameters
        path: str
        数据集文件路径

        dataset: str
        数据集名称

        train: bool
        True: 加载训练数据; False: 加载测试数据

        normalized: bool
        True: 数据归一化处理; False: 数据不归一化处理

        one_hot: bool
        True: labels one-hot编码; False: labels不使用one-hot编码

        batch_size: int
        batch size

        num_workers: int
        number of dataloader workers

    Returns
        data: Tensor
        数据

        labels: Tensor
        标签
    """
    if dataset == 'cifar10':
        # 手动加载cifar10数据
        data, labels = cifar10.load_data(path, train=train)
    elif dataset == 'cifar10_pytorch':
        # 使用pytorch提供的cifar10函数
        cifar10_train_dataset = cifar10.CIFAR10(path,
                                                train=True,
                                                transform=_transform(),
                                                target_transform=Onehot(),
                                                )
        cifar10_train_dataloader = DataLoader(cifar10_train_dataset,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              )
        cifar10_query_dataset = cifar10.CIFAR10(path,
                                                train=False,
                                                transform=_transform(),
                                                target_transform=Onehot(),
                                                )
        cifar10_query_dataloader = DataLoader(cifar10_query_dataset,
                                              batch_size=batch_size,
                                              num_workers=num_workers,
                                              )
        return cifar10_train_dataloader, cifar10_query_dataloader
    elif dataset == 'cifar10_gist':
        data, labels = cifar10.load_data_gist(path, train=train)
    elif dataset == 'nus_wide':
        data, labels = nus_wide.load_data(path, train=train)

    if normalized:
        data = normalization(data)

    if one_hot:
        labels = encode_onehot(labels).astype(np.int)

    return torch.from_numpy(data).float(), torch.from_numpy(labels).float()


def normalization(data):
    """归一化数据 (data - mean) / std

    Parameters
        data: ndarray
        数据

    Returns
        normalized_data: ndarray
        归一化后数据
    """
    if data.dtype != np.float:
        data = data.astype(np.float)
    return (data - data.mean()) / data.std()


def _transform():
    """返回transform

    Returns:
        transform: transforms
        图像变换
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform


def partition(data, labels, num_query):
    """划分数据集

    Parameters
        data: Tensor
        数据

        labels: Tensor
        标签

        num_query: int
        查询集数量

    Returns
        train_data: Tensor
        训练集

        train_labels: Tensor
        训练标签

        query_data: Tensor
        查询集

        query_labels: Tensor
        查询标签
    """
    perm_index = torch.randperm(data.shape[0])
    sample_index = perm_index[:num_query]
    other_index = perm_index[num_query:]

    query_data = data[sample_index, :]
    query_labels = labels[sample_index, :]

    train_data = data[other_index, :]
    train_labels = labels[other_index, :]

    return train_data, train_labels, query_data, query_labels


if __name__ == "__main__":
    train_data, train_labels = cifar10.load_data(path='/data3/zhaoshu/data/ImageRetrieval/pytorch_cifar10',
                                                 train=True)
    test_data, test_labels = cifar10.load_data(path='/data3/zhaoshu/data/ImageRetrieval/pytorch_cifar10',
                                               train=False)
    data = np.vstack((train_data, test_data))
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    labels = np.concatenate((train_labels, test_labels))
    num_query = 1000
    perm_index = np.random.permutation(data.shape[0])
    sample_index = perm_index[:num_query]
    other_index = perm_index[num_query:]

    query_data = data[sample_index, :]
    query_labels = labels[sample_index]

    train_data = data[other_index, :]
    train_labels = labels[other_index]

    import scipy.io as sio
    sio.savemat('cifar10.mat', {'train_data': train_data,
                                'train_labels': train_labels,
                                'query_data': query_data,
                                'query_labels': query_labels})
