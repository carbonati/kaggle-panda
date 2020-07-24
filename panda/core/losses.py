import torch
import torch.nn as nn
from torch.nn.modules.loss import MSELoss, _Loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def cauchy_loss(y_pred, y_true, c=1.0, reduction='mean'):
    x = y_pred - y_true
    loss = torch.log(0.5 * (x / c) ** 2 + 1)
    if reduction == 'sum':
        loss = loss.sum()
    if reduction == 'mean':
        loss = loss.mean()
    return loss


class CauchyLoss(_Loss):
    def __init__(self, c=1.0, reduction='mean', ignore_index=None):
        super(CauchyLoss, self).__init__(reduction=reduction)
        self.c = c
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            target = target[mask]
            input = input[mask]

        if not len(target):
            return torch.tensor(0.).to(input.device)

        return cauchy_loss(input, target.float(), self.c, self.reduction)

def clip_regression(x, y, v_min=0, v_max=4):
    min_mask = (y == v_min) & (x <= v_min)
    max_mask = (y == v_max) & (x >= v_max)
    x = x.masked_fill(min_mask, v_min)
    x = x.masked_fill(max_mask, v_max)
    return x, y


class ClippedMSELoss(MSELoss):
    def __init__(self, min=0, max=4, size_average=None, reduce=None, reduction='mean', ignore_index=None):
        super(ClippedMSELoss, self).__init__(size_average, reduce, reduction)
        self.min = min
        self.max = max
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            target = target[mask]
            input = input[mask]

        if not len(target):
            return torch.tensor(0.).to(input.device)

        input, target = clip_regression(input, target.float(), self.min, self.max)
        return super().forward(input, target.float())

