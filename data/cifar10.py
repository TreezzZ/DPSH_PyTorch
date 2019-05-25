# -*- coding: utf-8 -*-


import numpy as np
import scipy.io as sio
import torch.utils.data as data
from PIL import Image

import os
import sys
import pickle


def load_data(path, train=True):
    """加载CIFAR 10数据集

    Parameters
        path: str
        数据集路径

        train: bool
        True: 加载训练数据; False: 加载测试数据

    Returns
        data: ndarray
        数据

        labels: ndarray
        标签
    """
    base_folder = 'cifar-10-batches-py'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    if train:
        train_data = []
        train_labels = []
        for fentry in train_list:
            f = fentry[0]
            file = os.path.join(os.path.expanduser(path), base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            train_data.append(entry['data'])
            if 'labels' in entry:
                train_labels += entry['labels']
            else:
                train_labels += entry['fine_labels']
            fo.close()

        train_data = np.concatenate(train_data)
        train_data = train_data.reshape((50000, 3, 32, 32))
        train_data = train_data.transpose((0, 2, 3, 1))  # convert to HWC

        return train_data, train_labels
    else:
        f = test_list[0][0]
        file = os.path.join(os.path.expanduser(path), base_folder, f)
        fo = open(file, 'rb')
        if sys.version_info[0] == 2:
            entry = pickle.load(fo)
        else:
            entry = pickle.load(fo, encoding='latin1')
        test_data = entry['data']
        if 'labels' in entry:
            test_labels = entry['labels']
        else:
            test_labels = entry['fine_labels']
        fo.close()
        test_data = test_data.reshape((10000, 3, 32, 32))
        test_data = test_data.transpose((0, 2, 3, 1))  # convert to HWC

        return test_data, test_labels


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
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 ):
        data = sio.loadmat(root)
        self.transform = transform
        self.target_transform = target_transform

        if train:
            self.data = data['train_data']
            self.targets = data['train_labels']
        else:
            self.data = data['query_data']
            self.targets = data['query_labels']
        self.targets = self.targets.squeeze()

    def __getitem__(self, index):
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


class _CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    data_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
        ['test_batch', '40351d587109b95175f43aff81a1287e'],

    ]

    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 num_query=1000,
                 ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        self.data = []
        self.targets = []

        # 获取query不需要再加载一遍，因为后面使用了random，这样train和query可能有重叠
        # 我的方法是，只在第一次加载一遍数据，query直接访问第一次获得的dataset，从里面取
        if not train:
            return

        # now load the picked numpy arrays
        for file_name, checksum in self.data_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.targets = np.array(self.targets)

        # 划分数据
        perm_index = np.random.permutation(self.data.shape[0])
        sample_index = perm_index[:num_query]
        other_index = perm_index[num_query:]

        self.query_data = self.data[sample_index, :]
        self.query_labels = self.targets[sample_index]

        train_data = self.data[other_index, :]
        train_labels = self.targets[other_index]

        self.data = train_data
        self.targets = train_labels

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
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
