#!/usr/bin/env python
# -*- coding: utf-8 -*-


def cal_similarity_matrix(labels1, labels2, form='label'):
    """计算similarity matrix

    Parameters
        labels1, labels2: Tensor
        标签

        form: str
        'label': 单标签; 'tags': 多标签

    Returns
        similarity_matrix: Tensor
        相似度矩阵
    """
    similarity_matrix = labels1 @ labels2.t()
    if form == 'label':
        return similarity_matrix
    elif form == 'tags':
        similarity_matrix[similarity_matrix > 0] = 1
        return similarity_matrix
