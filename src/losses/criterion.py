import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)
def bce(pr, gt):
    #input (torch.Tensor): input data tensor with shape :math:`(B, *)`.
    #target (torch.Tensor): the target tensor with shape :math:`(B, *)`.
    return F.binary_cross_entropy_with_logits(pr, gt)
    
def criterion(pr, gt):
    return 0.5*bce(pr, gt) + 0.5*TverskyLoss(pr, gt)