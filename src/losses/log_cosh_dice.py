import torch
import torch.nn.functional as F

def log_cosh_dice(pr, gt, eps=1.):
    pr = torch.sigmoid(pr)
    tp = torch.sum(gt * pr, axis=(-2,-1))
    fp = torch.sum(pr, axis=(-2,-1)) - tp
    fn = torch.sum(gt, axis=(-2,-1)) - tp
    loss = torch.log(torch.cosh(1. - (2.*tp + eps) / (2.*tp + fn + fp + eps)))
    return torch.mean(loss)
