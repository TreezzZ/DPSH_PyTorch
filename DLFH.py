#!/usr/bin/env python
# -*- coding: utf-8 -*-

import models.modelloader as modelloader
import models.loss.dlfh_loss as dlfh_loss
from utils.calc_map import calc_map
from utils.cal_similarity_matrix import cal_similarity_matrix
from utils.visualizer import Visualizer
from data.onehot import encode_onehot

import torch
import torch.optim as optim
from tqdm import tqdm
from loguru import logger
import os


def dlfh(opt,
         train_dataloader,
         query_dataloader,
         ):
    """DLFH algorithm
    
    Parameters
        opt: Parser
        配置

        train_dataloader: DataLoader
        训练数据

        query_data: DataLoader
        查询数据

    Returns
        U: ndarray
        学到的hash code

        meanAP: float
        Mean Average Precision
    """
    # 预处理数据
    train_labels = torch.FloatTensor(encode_onehot(train_dataloader.dataset.targets)).to(opt.device)

    # 定义网络
    model = modelloader.load_model(opt.model, num_classes=opt.code_length)
    model.to(opt.device)
    criterion = dlfh_loss.DLFHLoss(opt.eta)
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # 初始化B
    N = len(train_dataloader.dataset)
    B = torch.randn(N, opt.code_length).sign().to(opt.device)
    U = torch.randn(N, opt.code_length).to(opt.device)

    # 算法开始
    vis = Visualizer(env='DLFH')
    train_tqdm = tqdm(range(opt.epochs))
    best_map = 0.0
    last_model = None
    model.train()
    for epoch in train_tqdm:
        train_tqdm.set_description("train")

        scheduler.step()
        # CNN
        total_loss = 0.0
        for data, labels, index in train_dataloader:
            data = data.to(opt.device)
            labels = labels.to(opt.device)

            optimizer.zero_grad()

            S = cal_similarity_matrix(labels, train_labels)
            outputs = model(data)
            U[index, :] = outputs.data
            B[index, :] = outputs.sign()

            loss = criterion(S, outputs, U)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        meanAP = evaluate(model, opt.device, query_dataloader, train_labels, B)

        # 保存当前最好结果
        if best_map < meanAP:
            if last_model:
                os.remove(os.path.join('result', last_model))
            best_map = meanAP
            last_model = 'model_{:.4f}.t'.format(best_map)
            torch.save(model, os.path.join('result', last_model))

        # 可视化，日志
        vis.plot('loss_{}_{}_{}'.format(opt.lr, opt.eta, opt.batch_size), total_loss)
        vis.plot('map_{}_{}_{}'.format(opt.lr, opt.eta, opt.batch_size), meanAP)
        logger.info('epoch: {}, loss: {:.4f}, map: {:.4f}'.format(epoch+1, total_loss, meanAP))
    return B, meanAP


def evaluate(model, device, query_dataloader, train_labels, B):
    """评估算法

    Parameters
        model: model
        学得的CNN模型

        device: device
        gpu or cpu

        query_dataloader: DataLoader
        测试数据

        train_labels: Tensor
        训练标签

        B: ndarray
        学到的hash code

    Returns
        meanAP: float
        mean Average precision
    """
    # CNN作为out-of-sampling extension
    model.eval()
    with torch.no_grad():
        N = len(query_dataloader.dataset)
        query_code = torch.randn(N, B.shape[1]).sign()
        for data, labels, index in query_dataloader:
            query_data = data.to(device)
            outputs = model(query_data)
            query_code[index, :] = outputs.sign().cpu()

        # query labels
        query_labels = encode_onehot(query_dataloader.dataset.targets)

        # 计算map
        meanAP = calc_map(query_code.cpu().numpy(),
                          B.cpu().numpy(),
                          query_labels,
                          train_labels.cpu().numpy(),
                          )

    return meanAP
