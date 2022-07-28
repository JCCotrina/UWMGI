import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from .bce import bce 

TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()

def criterion(pr, gt):

    return F.binary_cross_entropy_with_logits(pr, gt)