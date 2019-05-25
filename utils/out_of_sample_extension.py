#!/usr/bin/env python
# -*- coding: utf-8 -*-


def linear_extension(pre_calc, U):
    """linear out-of-sample extension

    Parameters
        pre_calc: Tensor
        预先计算的部分

        U: Tensor
        学到的hash code

    Returns
        W: ndarray
        权重矩阵
    """
    return pre_calc @ U
