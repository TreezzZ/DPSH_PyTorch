#!/usr/bin/env python
# -*- coding: utf-8 -*-

import DPSH
import data.dataloader as DataLoader
import data.cifar10 as cifar10

from loguru import logger

import argparse
import torch


def run_dpsh(opt):
    """运行DLFH算法

    Parameters
        opt: parser
        程序运行参数

    Returns
        None
    """
    logger.info("hyper-parameters: code_length: {}, lr: {}, batch_size:{}, eta:{}"
                .format(opt.code_length,
                        opt.lr,
                        opt.batch_size,
                        opt.eta),
                )

    onehot = False if opt.dataset == 'nus_wide' else True

    # 加载数据，pytorch版本的ciar10直接返回DataLoader类型，要分开对待，以后可能修改代码统一形式
    kwargs = {'num_workers': opt.num_workers,
              'batch_size': opt.batch_size,
              'num_train': 5000,
              'num_query': 1000,
              }
    # init
    cifar10.CIFAR10.init_load_data(opt.data_path,
                                   kwargs['num_train'],
                                   kwargs['num_query'],
                                   )
    train_dataloader = DataLoader.load_data(path=opt.data_path,
                                            dataset=opt.dataset,
                                            mode='train',
                                            **kwargs,
                                            )
    query_dataloader = DataLoader.load_data(path=opt.data_path,
                                            dataset=opt.dataset,
                                            mode='query',
                                            **kwargs,
                                            )
    database_dataloader = DataLoader.load_data(path=opt.data_path,
                                               dataset=opt.dataset,
                                               mode='all',
                                               **kwargs,
                                               )

    # DLFH算法
    DPSH.dpsh(opt,
              train_dataloader,
              query_dataloader,
              database_dataloader,
              )


def load_parse():
    """加载程序参数

    Parameters
        None

    Returns
        opt: parser
        程序参数
    """
    parser = argparse.ArgumentParser(description='DPSH_PyTorch')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='dataset used to train (default: cifar10)')
    parser.add_argument('--data-path', default='/data3/zhaoshu/data/ImageRetrieval/cifar10.mat', type=str,
                        help='path of cifar10 dataset')
    parser.add_argument('--num-query', default='1000', type=int,
                        help='number of query(default: 1000)')
    parser.add_argument('--num-train', default='5000', type=int,
                        help='number of train(default: 5000)')
    parser.add_argument('--code-length', default='12', type=int,
                        help='hyper-parameter: binary hash code length (default: 12)')

    parser.add_argument('--model', default='alexnet', type=str,
                        help='CNN model(default: alexnet')
    parser.add_argument('--gpu', default='0', type=int,
                        help='use gpu(default: 0. -1: use cpu)')
    parser.add_argument('--lr', default='1e-3', type=float,
                        help='learning rate(default: 1e-3)')
    parser.add_argument('--batch-size', default='64', type=int,
                        help='batch size(default: 64)')
    parser.add_argument('--epochs', default='50', type=int,
                        help='epochs(default:64)')
    parser.add_argument('--num-workers', default='4', type=int,
                        help='number of workers(default: 4)')
    parser.add_argument('--eta', default='50', type=float,
                        help='hyper-parameter: regularization term (default: 50)')

    return parser.parse_args()


if __name__ == "__main__":
    opt = load_parse()
    logger.add('logs/file_{time}.log')

    if opt.gpu == -1:
        opt.device = torch.device("cpu")
    else:
        opt.device = torch.device("cuda:%d" % opt.gpu)

    run_dpsh(opt)


    # import random
    # used_value = set()
    # for it in range(20):
    #     lr = lrs[random.randint(0, len(lrs)-1)]
    #     batch_size = batch_sizes[random.randint(0, len(batch_sizes)-1)]
    #     eta = etas[random.randint(0, len(etas)-1)]
    #
    #     hyper_params = (lr, batch_size, eta)
    #     if hyper_params in used_value:
    #         continue
    #     used_value.add(hyper_params)
    #
    #     opt.lr = lr
    #     opt.batch_size = batch_size
    #     opt.eta = eta
    #
    #     run_dlfh(opt)


    # run_dlfh(opt)
