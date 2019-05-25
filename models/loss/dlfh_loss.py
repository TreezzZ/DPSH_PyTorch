# -*- coding:utf-8 -*-

import torch.nn as nn
import torch


class DLFHLoss(nn.Module):
    def __init__(self, eta):
        super(DLFHLoss, self).__init__()
        self.eta = eta

    def forward(self, S, outputs, U):
        """
        前向传播

        Parameters
            S: Tensor
            相似矩阵

            outputs: Tensor
            CNN outputs

            U: Tensor
            Relaxation hash code

        Returns
            loss: Tensor
            损失
        """
        theta = outputs @ U.t() / 2

        # 防止exp溢出
        theta = torch.clamp(theta, min=-100, max=50)

        loss = torch.log(1 + torch.exp(theta)).sum() - (S * theta).sum()

        # 正则化项
        reg_term = (outputs.sign() - outputs).pow(2).sum()

        loss = loss + self.eta * reg_term
        loss = loss / S.numel()

        return loss
