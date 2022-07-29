import torch
import torch.nn as F

def bce(pr, gt):
    loss = F.CrossEntropyLoss()
    #input (torch.Tensor): input data tensor with shape :math:`(B, *)`.
    #target (torch.Tensor): the target tensor with shape :math:`(B, *)`.
    return loss(pr, gt)