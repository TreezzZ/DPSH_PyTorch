#!/usr/bin/env python
# -*- coding: utf-8 -*-

import models.modelloader as modelloader
import models.loss.dlfh_loss as dlfh_loss
from utils.calc_map import calc_map
from data.transform import encode_onehot

import torch
import torch.optim as optim
from loguru import logger
import os


def dpsh(opt,
         train_dataloader,
         query_dataloader,
         database_dataloader,
         ):
    """DPSH_PyTorch algorithm
    
    Parameters
        opt: Parser
        配置

        train_dataloader: DataLoader
        训练数据

        query_data: DataLoader
        查询数据

        database_dataloader: DataLoader
        整个数据集数据

    Returns
        None
    """
    # 标签onehot处理
    if opt.dataset == 'cifar10':
        train_labels = torch.FloatTensor(encode_onehot(train_dataloader.dataset.targets)).to(opt.device)
    elif opt.dataset == 'nus-wide':
        train_labels = torch.FloatTensor(train_dataloader.dataset.tags).to(opt.device)

    # 定义网络,optimizer,loss
    model = modelloader.load_model(opt.model, num_classes=opt.code_length)
    if opt.multi_gpu:
        model = torch.nn.DataParallel(model)
    model.to(opt.device)
    criterion = dlfh_loss.DLFHLoss(opt.eta)

    # 不知道为什么，加momentum无法收敛！！！
    # 不知道为什么，SGD不用zeros初始化U无法收敛！！！
    optimizer = optim.RMSprop(model.parameters(),
                              lr=opt.lr,
                              weight_decay=10**-5,
                              )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # 初始化
    N = len(train_dataloader.dataset)
    B = torch.zeros(N, opt.code_length).to(opt.device)
    U = torch.zeros(N, opt.code_length).to(opt.device)

    # 算法开始
    best_map = 0.0
    last_model = None
    for epoch in range(opt.epochs):
        scheduler.step()
        # CNN
        total_loss = 0.0
        model.train()
        for data, labels, index in train_dataloader:
            data = data.to(opt.device)
            labels = labels.to(opt.device)

            optimizer.zero_grad()

            S = (labels @ train_labels.t() > 0).float()

            outputs = model(data)

            U[index, :] = outputs.data
            B[index, :] = outputs.clone().sign()

            U = U.sign()

            loss = criterion(S, outputs, U)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        if epoch % opt.evaluate_freq == opt.evaluate_freq - 1:
            meanAP = evaluate(model, query_dataloader, train_labels, B, opt)

            # 保存当前最好结果
            if best_map < meanAP:
                if last_model:
                    os.remove(os.path.join('result', last_model))
                best_map = meanAP
                last_model = 'model_{:.4f}.t'.format(best_map)
                torch.save(model, os.path.join('result', last_model))

            logger.info('code_length: {}, epoch: {}, loss: {:.4f}, map: {:.4f}'.format(opt.code_length, epoch+1, total_loss, meanAP))

    # 加载性能最好模型，对整个数据集产生hash code进行evaluate
    model = torch.load(os.path.join('result', last_model))
    database_code = generate_code(model, database_dataloader, opt).to(opt.device)
    if opt.dataset == 'cifar10':
        database_labels = torch.FloatTensor(encode_onehot(database_dataloader.dataset.targets)).to(opt.device)
    elif opt.dataset == 'nus-wide':
        database_labels = torch.FloatTensor(database_dataloader.dataset.tags).to(opt.device)
    final_map = evaluate(model,
                         query_dataloader,
                         database_labels,
                         database_code,
                         opt,
                         )
    logger.info('code_length: {}, final_map: {:.4f}'.format(opt.code_length, final_map))


def evaluate(model,
             query_dataloader,
             train_labels,
             B,
             opt,
             ):
    """评估算法

    Parameters
        model: model
        学得的CNN模型

        query_dataloader: DataLoader
        测试数据

        train_labels: Tensor
        训练标签

        B: ndarray
        学到的hash code

        topk: int
        计算前topk个map

        opt: Parser
        参数

    Returns
        meanAP: float
        mean Average precision
    """
    model.eval()
    # CNN作为out-of-sampling extension
    query_code = generate_code(model, query_dataloader, opt).to(opt.device)

    # query labels
    if opt.dataset == 'cifar10':
        query_labels = torch.FloatTensor(encode_onehot(query_dataloader.dataset.targets)).to(opt.device)
    elif opt.dataset == 'nus-wide':
        query_labels = torch.FloatTensor(query_dataloader.dataset.tags).to(opt.device)

    # 计算map
    meanAP = calc_map(query_code,
                      B,
                      query_labels,
                      train_labels,
                      opt.device,
                      opt.topk,
                      )
    model.train()

    return meanAP


def generate_code(model, dataloader, opt):
    """产生hash code

    Parameters
        model: Model
        模型

        dataloader: DataLoader
        数据加载器

        opt: Parser
        参数

    Returns
        code: Tensor
        hash code
    """
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, opt.code_length])
        for data, _, index in dataloader:
            data = data.to(opt.device)
            outputs = model(data)
            code[index, :] = outputs.sign().cpu()

    return code
