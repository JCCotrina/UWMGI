import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from .bce import bce 

TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)

def criterion(pr, gt):

    return 0.5*bce(pr, gt) + 0.5*TverskyLoss(pr, gt)