import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from collections import OrderedDict
from torch.optim.optimizer import Optimizer


class homoscedastic_uncertainty_loss(nn.Module):
    """
    Based on paper <Multi-task learning using uncertainty to weigh losses for scene geometry and semantics>
    and paper <Auxiliary tasks in multi-task learning>
    pytorch implementation
    @ https://github.com/Mikoto10032/AutomaticWeightedLoss
    """
    def __init__(self, num=3):
        super(homoscedastic_uncertainty_loss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0           
        for i, loss in enumerate(x):
            #loss_sum += 0.5 / (torch.tensor(1).float().cuda() ** 2) * loss + torch.log(1 + torch.tensor(1).float().cuda() ** 2).cuda()
            loss_sum += 0.5 / (self.params[i].float().cuda() ** 2) * loss + torch.log(1 + self.params[i].float().cuda() ** 2).cuda()
        return loss_sum.cuda()


class Region_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice

def weighted_cross_entropy_loss(inputs, label):
    """
    Based on paper <Richer Convolutional Features for Edge Detection> RCF 
    pytorch implementation
    @ https://github.com/meteorshowers/RCF-pytorch/tree/master
    """
    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask > 0.5).float()).float() # ==1.
    num_negative = torch.sum((mask == 0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0
    inputs= torch.sigmoid(inputs)
    cost = torch.nn.functional.binary_cross_entropy(
            inputs.float(),label.float(), weight=mask, reduce=False)

    return cost.mean() #1.*torch.sum(cost)/10000

class Boundary_Loss:
    def __init__(self):
        self.device = ''
    def __call__(self, outputs, targets):
        criterion = sum([weighted_cross_entropy_loss(outpu, targets) for outpu in outputs])
        return criterion