#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils.calc_hamming_dist import calc_hamming_dist

import numpy as np


def calc_map(query_code,
             database_code,
             query_labels,
             database_labels,
             enable_fast=True,
             ):
    """计算mAP

    Parameters
        query_code: ndarray, {-1, +1}^{m * Q}
        query的hash code

        database_code: ndarray, {-1, +1}^{n * Q}
        database的hash code

        query_labels: ndarray, {0, 1}^{m * n_classes}
        query的label，onehot编码

        database_labels: ndarray, {0, 1}^{n * n_classes}
        database的label，onehot编码

        enable_fast: bool
        是否启用加速

    Returns
        meanAP: float
        Mean Average Precision
    """
    num_query = query_labels.shape[0]
    mean_AP = 0.0

    # 内存够大使用下面的代码预计算可以加速
    if enable_fast:
        pre_calc_retrieval = (query_labels @ database_labels.T > 0).astype(np.float32)
        pre_calc_retrieval_cnt = pre_calc_retrieval.sum(axis=1)
        pre_calc_hamming_dist = 0.5 * (query_code.shape[1] - query_code @ database_code.T)

    for i in range(num_query):
        # 加速
        if enable_fast:
            retrieval = pre_calc_retrieval[i, :]
            retrieval_cnt = pre_calc_retrieval_cnt[i]
            if retrieval_cnt == 0:
                continue
            hamming_dist = pre_calc_hamming_dist[i, :]
        else:
            # 检索
            retrieval = (query_labels[i, :] @ database_labels.T > 0).astype(np.float32)

            # 检索到数量
            retrieval_cnt = retrieval.sum()

            # 未检索到
            if retrieval_cnt == 0:
                continue

            # hamming distance
            hamming_dist = calc_hamming_dist(query_code[i, :], database_code)

        # 根据hamming distance安排检索结果位置
        retrieval = retrieval[np.argsort(hamming_dist)]

        # 每个位置打分
        score = np.linspace(1, retrieval_cnt, retrieval_cnt)

        # 检索到的下标位置
        index = np.asarray(np.where(retrieval == 1)) + 1.0

        mean_AP += np.mean(score / index)

    mean_AP = mean_AP / num_query
    return mean_AP


