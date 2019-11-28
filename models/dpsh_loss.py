import torch.nn as nn
import torch


class DPSHLoss(nn.Module):
    def __init__(self, eta):
        super(DPSHLoss, self).__init__()
        self.eta = eta

    def forward(self, U_cnn, U, S):
        theta = U_cnn @ U.t() / 2

        # Prevent overflow
        theta = torch.clamp(theta, min=-100, max=50)

        pair_loss = (torch.log(1 + torch.exp(theta)) - S * theta).mean()
        regular_term = (U_cnn - U_cnn.sign()).pow(2).mean()
        loss = pair_loss + self.eta * regular_term

        return loss
